/**
 * The editing surface.
 *
 * Playback: we create and own the `<audio>` element and hand it to WaveSurfer as
 * `media`. This is not incidental. Passing `peaks` + `duration` without a media
 * element puts WaveSurfer into its "render from precomputed peaks, skip
 * decoding" path -- which also skips creating any media element at all, so the
 * waveform draws perfectly and nothing ever plays, silently. Owning the element
 * keeps the instant render (peaks still come from the server) while making
 * playback ours to control, and makes load failures observable instead of
 * vanishing into a swallowed promise.
 *
 * Everything interactive is a custom absolutely-positioned overlay rather than
 * WaveSurfer's regions plugin: regions model a span with drag handles, which
 * fits loops but not point cues, and the plugin owns its own hit-testing, which
 * makes "double-click empty space to add a cue on the nearest downbeat" awkward.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { useStore } from '../state/store'
import {
  CUE_KIND_LABELS, SECTION_COLORS, fmtTime, hex, rgba,
  type CueKind, type CuePoint, type Loop,
} from '../api/types'

type Drag =
  | { kind: 'cue'; id: string }
  | { kind: 'loop-move'; id: string; grabOffset: number }
  | { kind: 'loop-start'; id: string }
  | { kind: 'loop-end'; id: string }
  | { kind: 'section-start'; index: number }
  | { kind: 'section-end'; index: number }
  | null

export function WaveformEditor() {
  const {
    trackId, analysis, prep, peaks, duration, selectedId, selectedSection,
    select, selectSection, moveCue, moveLoop, addCue, deleteCue, moveSectionEdge,
  } = useStore()

  const containerRef = useRef<HTMLDivElement>(null)
  const waveRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WaveSurfer | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const [playhead, setPlayhead] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [ready, setReady] = useState(false)
  const [audioError, setAudioError] = useState<string | null>(null)
  const [volume, setVolume] = useState(0.85)
  const [zoom, setZoom] = useState(1)
  const [drag, setDrag] = useState<Drag>(null)
  const [addKind, setAddKind] = useState<CueKind>('custom')
  const [hoverT, setHoverT] = useState<number | null>(null)
  const [waveH, setWaveH] = useState(200)

  // Let the waveform fill the pane instead of sitting in a fixed 128 px strip
  // with dead space under it. A DJ reads structure off the envelope, so vertical
  // resolution is not decoration.
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const measure = () => {
      const h = el.clientHeight - 46 /* ribbon */ - 26 /* section lane */ - 16
      setWaveH(Math.max(120, Math.min(520, h)))
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // --- WaveSurfer + audio lifecycle ----------------------------------------
  useEffect(() => {
    if (!waveRef.current || !trackId || peaks.length === 0) return
    wsRef.current?.destroy()
    setReady(false)
    setAudioError(null)

    const audio = new Audio()
    audio.preload = 'auto'
    audio.volume = volume
    audio.src = `/api/tracks/${trackId}/audio`
    audioRef.current = audio

    const onCanPlay = () => setReady(true)
    const onErr = () => {
      const codes: Record<number, string> = {
        1: 'loading was aborted', 2: 'network error',
        3: 'the file could not be decoded',
        4: 'this audio format is not supported by your browser',
      }
      setAudioError(codes[audio.error?.code ?? 0] ?? 'audio failed to load')
    }
    audio.addEventListener('canplay', onCanPlay)
    audio.addEventListener('error', onErr)

    const ws = WaveSurfer.create({
      container: waveRef.current,
      media: audio,             // <- ours, so playback actually happens
      height: waveH,
      waveColor: '#46536c',
      progressColor: '#6b7f9f',
      cursorColor: 'transparent',
      normalize: true,
      interact: false,          // the overlay owns all pointer interaction
      peaks: [peaks],
      duration,
    })
    ws.on('timeupdate', (t: number) => setPlayhead(t))
    ws.on('play', () => setPlaying(true))
    ws.on('pause', () => setPlaying(false))
    ws.on('finish', () => setPlaying(false))
    wsRef.current = ws

    return () => {
      audio.removeEventListener('canplay', onCanPlay)
      audio.removeEventListener('error', onErr)
      ws.destroy()
      audio.pause()
      audio.src = ''
      wsRef.current = null
      audioRef.current = null
    }
    // volume is applied separately; re-creating on volume change would be wrong
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trackId, peaks, duration, waveH])

  useEffect(() => {
    if (audioRef.current) audioRef.current.volume = volume
  }, [volume])

  const togglePlay = useCallback(async () => {
    const a = audioRef.current
    if (!a) return
    try {
      if (a.paused) await a.play()
      else a.pause()
    } catch (e) {
      // Autoplay policy, or a decode failure surfacing late. Say so rather than
      // leaving a dead button.
      setAudioError(e instanceof Error ? e.message : 'playback was blocked')
    }
  }, [])

  // --- coordinate transform -------------------------------------------------
  const width = (containerRef.current?.clientWidth ?? 1000) * zoom
  const toX = useCallback((t: number) => (t / Math.max(duration, 1e-6)) * width,
    [duration, width])
  const toT = useCallback((x: number) => (x / Math.max(width, 1)) * duration,
    [duration, width])

  const localX = (e: { clientX: number }): number => {
    const el = containerRef.current
    if (!el) return 0
    return e.clientX - el.getBoundingClientRect().left + el.scrollLeft
  }

  const seek = useCallback((t: number) => {
    const a = audioRef.current
    if (a && duration > 0 && Number.isFinite(t)) {
      a.currentTime = Math.max(0, Math.min(duration - 0.05, t))
      setPlayhead(a.currentTime)
    }
  }, [duration])

  // --- bar / phrase grid ----------------------------------------------------
  const grid = useMemo(() => {
    if (!analysis) return { bars: [] as number[], phrases: [] as number[] }
    const g = analysis.beat_grid
    const barSec = (60 / g.bpm) * g.beats_per_bar
    const bars: number[] = []
    if (toX(barSec) > 9) {          // only draw bar lines when they're readable
      for (let t = g.anchor_sec; t < duration; t += barSec) bars.push(t)
    }
    return { bars, phrases: analysis.phrase_grid.boundaries_sec }
  }, [analysis, duration, toX])

  // --- drag handling --------------------------------------------------------
  useEffect(() => {
    if (!drag) return
    const onMove = (e: PointerEvent) => {
      const t = toT(localX(e))
      if (drag.kind === 'cue') moveCue(drag.id, t)
      else if (drag.kind === 'loop-start') {
        const l = prep?.loops.find((x) => x.id === drag.id)
        if (l) moveLoop(l.id, t, l.end_sec)
      } else if (drag.kind === 'loop-end') {
        const l = prep?.loops.find((x) => x.id === drag.id)
        if (l) moveLoop(l.id, l.start_sec, t)
      } else if (drag.kind === 'loop-move') {
        const l = prep?.loops.find((x) => x.id === drag.id)
        if (l) {
          const len = l.end_sec - l.start_sec
          moveLoop(l.id, t - drag.grabOffset, t - drag.grabOffset + len)
        }
      } else if (drag.kind === 'section-start') {
        moveSectionEdge(drag.index, 'start', t)
      } else if (drag.kind === 'section-end') {
        moveSectionEdge(drag.index, 'end', t)
      }
    }
    const onUp = () => setDrag(null)
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
  }, [drag, prep, moveCue, moveLoop, moveSectionEdge, toT])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null
      if (el && /^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName)) return
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedId
          && prep?.cues.some((c) => c.id === selectedId)) {
        e.preventDefault()
        deleteCue(selectedId)
      }
      if (e.code === 'Space') { e.preventDefault(); void togglePlay() }
      if (e.key === 'ArrowLeft') seek(playhead - (e.shiftKey ? 10 : 2))
      if (e.key === 'ArrowRight') seek(playhead + (e.shiftKey ? 10 : 2))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selectedId, prep, deleteCue, togglePlay, seek, playhead])

  if (!analysis || !prep) {
    return <div className="editor empty">Upload a track to begin.</div>
  }
  const g = analysis.beat_grid

  return (
    <div className="editor">
      <div className="editor-toolbar">
        <button className="btn primary play" onClick={() => void togglePlay()}
                disabled={!!audioError}>
          {playing ? '❚❚' : '▶'}
        </button>
        <span className="mono time">{fmtTime(playhead)} <i>/ {fmtTime(duration)}</i></span>
        {!ready && !audioError && <span className="loading">buffering…</span>}

        <label className="vol" title="Volume">
          <span aria-hidden="true">🔊</span>
          <input type="range" min={0} max={1} step={0.01} value={volume}
                 onChange={(e) => setVolume(Number(e.target.value))} />
        </label>

        <span className="chip">{g.bpm.toFixed(2)} BPM</span>
        <span className="chip">{analysis.phrase_grid.phrase_bars}-bar phrases</span>
        {analysis.tempo_changes.length > 0 && (
          <span className="chip warn-chip"
                title="The constant-tempo grid is only valid up to the first change">
            {analysis.tempo_changes.length} tempo change
            {analysis.tempo_changes.length > 1 ? 's' : ''}
          </span>
        )}

        <div className="spacer" />

        <label className="zoom">
          zoom
          <input type="range" min={1} max={12} step={0.5} value={zoom}
                 onChange={(e) => setZoom(Number(e.target.value))} />
        </label>
        <label className="addkind" title="Which cue type a double-click creates">
          Double-click adds
          <select value={addKind} onChange={(e) => setAddKind(e.target.value as CueKind)}>
            {(Object.keys(CUE_KIND_LABELS) as CueKind[])
              .filter((k) => k !== 'initial')
              .map((k) => <option key={k} value={k}>{CUE_KIND_LABELS[k]}</option>)}
          </select>
        </label>
      </div>

      {audioError && (
        <div className="audio-error">
          Audio didn’t load — {audioError}. The analysis below is still valid.
        </div>
      )}

      <div className="editor-scroll" ref={containerRef}
           onPointerMove={(e) => setHoverT(toT(localX(e)))}
           onPointerLeave={() => setHoverT(null)}>
        <div className="editor-canvas" style={{ width }}>

          {/* section bands — click to select, drag edges to move boundaries */}
          <div className="lane sections">
            {analysis.sections.map((s, i) => {
              const x = toX(s.start_sec)
              const w = Math.max(3, toX(s.end_sec) - x)
              const col = SECTION_COLORS[s.label]
              return (
                <div key={i}
                     className={`section-band${selectedSection === i ? ' selected' : ''}`}
                     style={{ left: x, width: w, background: col + '33', borderColor: col }}
                     onPointerDown={(e) => { e.stopPropagation(); selectSection(i) }}
                     title={`${s.label} #${s.occurrence} · ${s.length_bars.toFixed(0)} bars`
                       + ` · ${(s.confidence * 100).toFixed(0)}%`
                       + (s.ambiguous_with ? ` · could also be ${s.ambiguous_with}` : '')
                       + (s.is_manual ? ' · edited by you' : '')}>
                  <div className="sec-handle left"
                       onPointerDown={(e) => {
                         e.stopPropagation(); selectSection(i)
                         setDrag({ kind: 'section-start', index: i })
                       }} />
                  <span className="section-label" style={{ color: col }}>
                    {s.label}{s.occurrence > 1 ? ` ${s.occurrence}` : ''}
                    {s.ambiguous_with && <b className="amb" title={`Close call with ${s.ambiguous_with}`}>?</b>}
                    {s.is_manual && <b className="amb edited" title="edited by you">✎</b>}
                  </span>
                  <div className="sec-handle right"
                       onPointerDown={(e) => {
                         e.stopPropagation(); selectSection(i)
                         setDrag({ kind: 'section-end', index: i })
                       }} />
                </div>
              )
            })}
          </div>

          <div className="wave-wrap" style={{ height: waveH + 22 }}>
            <div ref={waveRef} className="wave" />

            <div className="grid-layer">
              {grid.bars.map((t, i) => (
                <div key={`b${i}`} className="gridline bar" style={{ left: toX(t) }} />
              ))}
              {grid.phrases.map((t, i) => (
                <div key={`p${i}`} className="gridline phrase" style={{ left: toX(t) }} />
              ))}
              {analysis.tempo_changes.map((tc, i) => (
                <div key={`t${i}`} className="gridline tempo" style={{ left: toX(tc.time_sec) }}
                     title={`Tempo ${tc.bpm_before} → ${tc.bpm_after} BPM`} />
              ))}
            </div>

            {analysis.vocals.map((v, i) => (
              <div key={`v${i}`} className="vocal-span"
                   style={{ left: toX(v.start_sec),
                            width: Math.max(2, toX(v.end_sec) - toX(v.start_sec)) }}
                   title={`vocal · ${(v.confidence * 100).toFixed(0)}%`} />
            ))}

            {prep.loops.map((l: Loop) => {
              const x = toX(l.start_sec)
              const w = Math.max(6, toX(l.end_sec) - x)
              return (
                <div key={l.id}
                     className={`loop-region${selectedId === l.id ? ' selected' : ''}`
                       + (l.accepted === false ? ' rejected' : '')}
                     style={{ left: x, width: w, background: rgba(l.color, 0.16),
                              borderColor: hex(l.color) }}
                     onPointerDown={(e) => {
                       e.stopPropagation(); select(l.id)
                       setDrag({ kind: 'loop-move', id: l.id,
                                 grabOffset: toT(localX(e)) - l.start_sec })
                     }}
                     title={`${l.label} · ${l.length_bars} bars`}>
                  <div className="loop-handle left"
                       onPointerDown={(e) => { e.stopPropagation(); select(l.id); setDrag({ kind: 'loop-start', id: l.id }) }} />
                  <span className="loop-label" style={{ color: hex(l.color) }}>
                    ⟲ {l.length_bars}
                  </span>
                  <div className="loop-handle right"
                       onPointerDown={(e) => { e.stopPropagation(); select(l.id); setDrag({ kind: 'loop-end', id: l.id }) }} />
                </div>
              )
            })}

            {/* `detail` is 0 on pointerdown in every spec-compliant browser, so
                click-count must not be read there — double-click gets its own
                handler. */}
            <div className="click-layer"
                 onPointerDown={(e) => { select(null); seek(toT(localX(e))) }}
                 onDoubleClick={(e) => { e.stopPropagation(); addCue(toT(localX(e)), addKind) }} />

            {prep.cues.map((c: CuePoint) => (
              <div key={c.id}
                   className={`cue${selectedId === c.id ? ' selected' : ''}`
                     + (c.accepted === false ? ' rejected' : '')
                     + (c.kind === 'initial' ? ' initial' : '')}
                   style={{ left: toX(c.time_sec) }}
                   onPointerDown={(e) => { e.stopPropagation(); select(c.id); setDrag({ kind: 'cue', id: c.id }) }}
                   onDoubleClick={(e) => { e.stopPropagation(); seek(c.time_sec) }}
                   title={`${c.label} · ${fmtTime(c.time_sec)}`
                     + (c.confidence ? ` · ${(c.confidence * 100).toFixed(0)}%` : ' · manual')}>
                <div className="cue-flag" style={{ background: hex(c.color) }}>
                  {c.label}
                  {c.source !== 'ai' && <span className="cue-edited">✎</span>}
                </div>
                <div className="cue-stem" style={{ background: hex(c.color) }} />
              </div>
            ))}

            <div className="playhead" style={{ left: toX(playhead) }} />
            {hoverT !== null && <div className="hoverline" style={{ left: toX(hoverT) }} />}
          </div>

          <EnergyRibbon width={width} />
        </div>
      </div>

      <div className="editor-hint">
        Click to seek · <kbd>Space</kbd> play/pause · <kbd>←</kbd><kbd>→</kbd> nudge
        · double-click to add a <b>{CUE_KIND_LABELS[addKind]}</b> · drag markers and
        section edges to move them · <kbd>Del</kbd> removes the selected cue
      </div>
    </div>
  )
}

/** Energy, low-band and vocal curves under the waveform. */
function EnergyRibbon({ width }: { width: number }) {
  const analysis = useStore((s) => s.analysis)
  if (!analysis) return null
  const e = analysis.energy
  const n = e.times_sec.length
  if (!n) return null
  const H = 46

  const path = (vals: number[]): string => {
    if (!vals.length) return ''
    const lo = Math.min(...vals)
    const span = (Math.max(...vals) - lo) || 1
    return vals.map((v, i) => {
      const x = (i / (n - 1)) * width
      const y = H - ((v - lo) / span) * H
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
    }).join(' ')
  }

  return (
    <svg className="ribbon" width={width} height={H} viewBox={`0 0 ${width} ${H}`}
         preserveAspectRatio="none">
      <path d={path(e.rms_db)} className="curve rms" />
      <path d={path(e.low)} className="curve low" />
      {e.vocal_likelihood.length > 0 && (
        <path d={path(e.vocal_likelihood)} className="curve vocal" />
      )}
    </svg>
  )
}
