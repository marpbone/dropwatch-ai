/**
 * What the user wants the analyser to look for.
 *
 * Rendered from the `AnalysisConfig` the backend serves, so adding a cue type on
 * the server makes a new row appear here with no frontend change. Every control
 * writes back through the store, which calls `/recommend` — milliseconds, not a
 * re-analysis — so the waveform updates as you toggle.
 *
 * Rows whose section type this particular track does not contain are dimmed and
 * explained rather than hidden: a Chorus checkbox that can never fire is worse
 * than one that tells you why.
 */
import { useStore } from '../state/store'
import {
  CUE_KIND_LABELS, CUE_REQUIRES_SECTION, LOOP_KIND_LABELS, fromHex, hex,
  type CueKind, type LoopLength, type LoopSelection, type MixOffset,
  type SnapMode, type VocalMode,
} from '../api/types'

const MIX_CHOICES: MixOffset[] = ['4', '8', '16', '32']

export function ConfigPanel() {
  const { config, analysis, setCueEnabled, setCueColor, setCueField, patchLoops,
          setLoopLocation, setConfig, scaleBpm, busy } = useStore()
  if (!config) return <aside className="panel"><div className="panel-section">Loading…</div></aside>

  const present = new Set(analysis?.label_summary.present ?? [])
  const hasAnalysis = !!analysis

  /** A cue type is unavailable when the section it depends on isn't in this track. */
  const unavailable = (kind: CueKind): string | null => {
    if (!hasAnalysis) return null
    const needs = CUE_REQUIRES_SECTION[kind]
    if (needs && !present.has(needs)) return `no ${needs} found in this track`
    return null
  }

  const mixRow = (kind: 'mix_in' | 'mix_out') => {
    const field = kind === 'mix_in' ? 'mix_in_bars' : 'mix_out_bars'
    const value = config.mix[field]
    const where = kind === 'mix_in' ? 'before the intro ends' : 'after the outro starts'
    return (
      <div className="mix-offset">
        <span className="mix-label">bars {where}</span>
        <div className="seg">
          {MIX_CHOICES.map((b) => (
            <button key={b} type="button"
                    className={`seg-btn${value === b ? ' on' : ''}`}
                    onClick={() => setConfig({ ...config, mix: { ...config.mix, [field]: b } })}>
              {b}
            </button>
          ))}
        </div>
      </div>
    )
  }

  return (
    <aside className="panel">
      <section className="panel-section">
        <h2>Cue points</h2>
        <div className="cue-rows">
          {config.cues.map((c) => {
            const why = unavailable(c.kind)
            const locked = c.kind === 'initial'
            return (
              <div key={c.kind}>
                <div className={`cue-row${c.enabled && !why ? '' : ' off'}`}>
                  <label className="row-main">
                    <input type="checkbox" checked={c.enabled} disabled={locked || !!why}
                           onChange={(e) => setCueEnabled(c.kind, e.target.checked)} />
                    <span className="swatch" style={{ background: hex(c.color) }} />
                    <span className="row-name">{CUE_KIND_LABELS[c.kind as CueKind]}</span>
                  </label>
                  <div className="row-controls">
                    <input type="color" value={hex(c.color)} title="Marker colour"
                           onChange={(e) => setCueColor(c.kind, fromHex(e.target.value))} />
                    {!locked && (
                      <>
                        <select value={c.snap} title="Snap to…"
                                onChange={(e) => setCueField(c.kind, 'snap', e.target.value as SnapMode)}>
                          <option value="beat">beat</option>
                          <option value="downbeat">bar</option>
                          <option value="phrase">phrase</option>
                        </select>
                        <label className="all-toggle"
                               title="Mark every occurrence, not just the most confident one">
                          <input type="checkbox" checked={c.select_all}
                                 onChange={(e) => setCueField(c.kind, 'select_all', e.target.checked)} />
                          all
                        </label>
                      </>
                    )}
                  </div>
                </div>
                {why && <div className="row-why">{why}</div>}
                {c.enabled && !why && (c.kind === 'mix_in' || c.kind === 'mix_out')
                  && mixRow(c.kind)}
              </div>
            )
          })}
        </div>
      </section>

      <section className="panel-section">
        <h2>BPM fix</h2>
        <div className="bpm-fix">
          <span className="bpm-now mono">
            {analysis ? analysis.beat_grid.bpm.toFixed(2) : '—'}
          </span>
          <button className="btn" disabled={!analysis || busy}
                  onClick={() => void scaleBpm(2)}>×2</button>
          <button className="btn" disabled={!analysis || busy}
                  onClick={() => void scaleBpm(0.5)}>÷2</button>
        </div>
      </section>

      <section className="panel-section">
        <h2>
          <label className="inline">
            <input type="checkbox" checked={config.loops.enabled}
                   onChange={(e) => patchLoops({ enabled: e.target.checked })} />
            Loops
          </label>
        </h2>
        <fieldset disabled={!config.loops.enabled} className="loop-config">
          <div className="sub">Locations</div>
          <div className="loop-locations">
            {config.loops.locations.map((l) => (
              <label key={l.kind} className="chk">
                <input type="checkbox" checked={l.enabled}
                       onChange={(e) => setLoopLocation(l.kind, e.target.checked)} />
                {LOOP_KIND_LABELS[l.kind]}
              </label>
            ))}
          </div>

          <div className="sub">Length</div>
          <div className="radio-row">
            {(['4', '8', '16', 'auto'] as LoopLength[]).map((v) => (
              <label key={v} className="rad">
                <input type="radio" name="looplen" checked={config.loops.length === v}
                       onChange={() => patchLoops({ length: v })} />
                {v === 'auto' ? 'Auto' : `${v} bars`}
              </label>
            ))}
          </div>

          <div className="sub">How many</div>
          <div className="radio-row">
            {([['first', 'First match'], ['top_n', 'Best N'], ['all', 'All found']] as
              [LoopSelection, string][]).map(([v, label]) => (
              <label key={v} className="rad">
                <input type="radio" name="loopsel" checked={config.loops.selection === v}
                       onChange={() => patchLoops({ selection: v })} />
                {label}
              </label>
            ))}
            {config.loops.selection === 'top_n' && (
              <input className="narrow" type="number" min={1} max={8} value={config.loops.top_n}
                     onChange={(e) => patchLoops({ top_n: Number(e.target.value) })} />
            )}
          </div>

          <div className="row-controls spread">
            <label className="inline">
              Colour
              <input type="color" value={hex(config.loops.color)}
                     onChange={(e) => patchLoops({ color: fromHex(e.target.value) })} />
            </label>
            <label className="inline" title="Reject loops whose boundary lands inside a sung phrase">
              <input type="checkbox" checked={config.loops.avoid_vocal_cuts}
                     onChange={(e) => patchLoops({ avoid_vocal_cuts: e.target.checked })} />
              Don’t cut vocals
            </label>
          </div>
        </fieldset>
      </section>

      <section className="panel-section">
        <h2>Analysis</h2>
        <label className="field">
          Vocal detection
          <select value={config.vocal_mode}
                  onChange={(e) => setConfig({ ...config, vocal_mode: e.target.value as VocalMode })}>
            <option value="off">Off</option>
            <option value="heuristic">Fast</option>
            <option value="separation">Accurate</option>
          </select>
        </label>

        <label className="field">
          Minimum spacing between cues
          <span className="inline">
            <input type="range" min={0} max={8} step={1} value={config.min_cue_spacing_bars}
                   onChange={(e) => setConfig({ ...config, min_cue_spacing_bars: Number(e.target.value) })} />
            <span className="mono">{config.min_cue_spacing_bars} bars</span>
          </span>
        </label>
      </section>
    </aside>
  )
}
