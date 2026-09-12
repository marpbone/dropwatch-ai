/**
 * Application state.
 *
 * One store, three slices of state with genuinely different lifecycles:
 *
 *   config      -- what the user asked for. Cheap to change, drives /recommend.
 *   analysis    -- what the DSP found. Expensive, immutable once computed.
 *   preparation -- what the DJ has decided. The editable document, and the only
 *                  thing that gets exported.
 *
 * Keeping them apart is what allows toggling a checkbox to cost 40 ms instead of
 * 14 s, and it is why user edits survive a re-run of the recommender: edited and
 * hand-made markers are merged back over fresh AI output rather than being
 * replaced by it (see `applyRecommendation`).
 */
import { create } from 'zustand'
import { api } from '../api/client'
import type {
  AnalysisConfig, CueKind, CuePoint, ExportFormat, Loop, RGB, SectionEdit,
  SectionLabel, TrackAnalysis, TrackPreparation,
} from '../api/types'

interface PendingFeedback {
  marker_id: string
  marker_class: 'cue' | 'loop'
  kind: string
  action: 'accepted' | 'rejected' | 'moved' | 'deleted' | 'recoloured'
  original_time: number | null
  final_time: number | null
  confidence: number | null
}

interface State {
  trackId: string | null
  filename: string
  duration: number
  config: AnalysisConfig | null
  analysis: TrackAnalysis | null
  prep: TrackPreparation | null
  formats: ExportFormat[]
  peaks: number[]
  status: string
  progress: number
  busy: boolean
  error: string | null
  selectedId: string | null
  selectedSection: number | null
  pendingFeedback: PendingFeedback[]

  bootstrap: () => Promise<void>
  uploadAndAnalyze: (file: File) => Promise<void>
  loadTrack: (trackId: string) => Promise<void>
  setConfig: (c: AnalysisConfig) => void
  reRecommend: () => Promise<void>
  setCueEnabled: (kind: CueKind, enabled: boolean) => void
  setCueColor: (kind: CueKind, color: RGB) => void
  setCueField: <K extends keyof AnalysisConfig['cues'][number]>(
    kind: CueKind, field: K, value: AnalysisConfig['cues'][number][K]) => void
  patchLoops: (patch: Partial<AnalysisConfig['loops']>) => void
  setLoopLocation: (kind: string, enabled: boolean) => void

  select: (id: string | null) => void
  selectSection: (i: number | null) => void
  moveSectionEdge: (index: number, which: 'start' | 'end', t: number) => void
  relabelSection: (index: number, label: SectionLabel) => void
  deleteSection: (index: number) => void
  splitSection: (index: number, at: number) => void
  applySectionEdits: (edits: SectionEdit[]) => Promise<void>
  scaleBpm: (factor: 0.5 | 2) => Promise<void>
  moveCue: (id: string, t: number) => void
  deleteCue: (id: string) => void
  setCueAccepted: (id: string, accepted: boolean | null) => void
  recolorCue: (id: string, color: RGB) => void
  renameCue: (id: string, label: string) => void
  addCue: (t: number, kind: CueKind) => void
  moveLoop: (id: string, start: number, end: number) => void
  deleteLoop: (id: string) => void
  setLoopBars: (id: string, bars: number) => void
  save: () => Promise<void>
}

const snapTo = (t: number, a: TrackAnalysis | null, mode: 'beat' | 'downbeat' | 'phrase'): number => {
  if (!a) return t
  const g = a.beat_grid
  const beat = 60 / g.bpm
  if (mode === 'beat') return g.anchor_sec + Math.round((t - g.anchor_sec) / beat) * beat
  if (mode === 'phrase' && a.phrase_grid.boundaries_sec.length) {
    let best = a.phrase_grid.boundaries_sec[0]
    for (const b of a.phrase_grid.boundaries_sec) if (Math.abs(b - t) < Math.abs(best - t)) best = b
    return best
  }
  const rel = (t - g.anchor_sec) / beat - g.downbeat_offset
  const bar = Math.round(rel / g.beats_per_bar)
  return g.anchor_sec + (g.downbeat_offset + bar * g.beats_per_bar) * beat
}

// Boundary drags fire on every pointer move; persisting each one would hammer
// the API, so the last one wins after a pause.
let sectionCommitTimer: ReturnType<typeof setTimeout> | null = null
let pendingEdit: SectionEdit | null = null

export const useStore = create<State>((set, get) => ({
  trackId: null, filename: '', duration: 0,
  config: null, analysis: null, prep: null, formats: [], peaks: [],
  status: 'Ready', progress: 0, busy: false, error: null,
  selectedId: null, selectedSection: null, pendingFeedback: [],

  async bootstrap() {
    try {
      const [config, formats] = await Promise.all([api.defaultConfig(), api.exportFormats()])
      set({ config, formats })
    } catch (e) {
      set({ error: String(e) })
    }
  },

  async uploadAndAnalyze(file) {
    const cfg = get().config
    if (!cfg) return
    set({ busy: true, error: null, status: 'Uploading…', progress: 0.02,
          analysis: null, prep: null, peaks: [] })
    try {
      const up = await api.upload(file)
      set({ trackId: up.track_id, filename: up.filename, duration: up.duration_sec })
      const job = await api.analyze(up.track_id, cfg)
      for (;;) {
        const s = await api.job(job.id)
        set({ status: s.stage || s.status, progress: s.progress })
        if (s.status === 'done') break
        if (s.status === 'error') throw new Error(s.error ?? 'analysis failed')
        await new Promise((r) => setTimeout(r, 400))
      }
      await get().loadTrack(up.track_id)
    } catch (e) {
      set({ error: String(e), status: 'Failed' })
    } finally {
      set({ busy: false })
    }
  },

  async loadTrack(trackId) {
    set({ busy: true, status: 'Loading…' })
    try {
      const [analysis, prep, wf] = await Promise.all([
        api.analysis(trackId), api.preparation(trackId), api.waveform(trackId, 2400),
      ])
      set({ trackId, analysis, prep, peaks: wf.peaks, duration: wf.duration_sec,
            filename: analysis.meta.filename, status: 'Ready', progress: 1 })
    } catch (e) {
      set({ error: String(e) })
    } finally {
      set({ busy: false })
    }
  },

  setConfig(config) { set({ config }); void get().reRecommend() },

  async reRecommend() {
    const { trackId, config, prep } = get()
    if (!trackId || !config || !prep) return
    set({ status: 'Updating recommendations…' })
    try {
      const fresh = await api.recommend(trackId, config)
      // Preserve everything the DJ touched. AI output is a proposal; a marker the
      // user made, moved or judged is a decision, and re-running the recommender
      // must never discard a decision.
      const keptCues = prep.cues.filter((c) => c.source !== 'ai' || c.accepted !== null)
      const keptIds = new Set(keptCues.map((c) => c.id))
      const keptLoops = prep.loops.filter((l) => l.source !== 'ai' || l.accepted !== null)
      const keptLoopIds = new Set(keptLoops.map((l) => l.id))
      set({
        prep: {
          ...fresh,
          cues: [...fresh.cues.filter((c) => !keptIds.has(c.id)), ...keptCues]
            .sort((a, b) => a.time_sec - b.time_sec),
          loops: [...fresh.loops.filter((l) => !keptLoopIds.has(l.id)), ...keptLoops]
            .sort((a, b) => a.start_sec - b.start_sec),
        },
        status: 'Ready',
      })
    } catch (e) {
      set({ error: String(e), status: 'Ready' })
    }
  },

  setCueEnabled(kind, enabled) {
    const c = get().config
    if (!c) return
    get().setConfig({ ...c, cues: c.cues.map((x) => (x.kind === kind ? { ...x, enabled } : x)) })
  },

  setCueColor(kind, color) {
    const c = get().config
    if (!c) return
    // Recolour existing markers of this kind immediately: colour is presentation,
    // not a recommendation, so it should not wait for a round trip.
    const prep = get().prep
    if (prep) {
      set({ prep: { ...prep, cues: prep.cues.map((q) => (q.kind === kind ? { ...q, color } : q)) } })
    }
    set({ config: { ...c, cues: c.cues.map((x) => (x.kind === kind ? { ...x, color } : x)) } })
  },

  setCueField(kind, field, value) {
    const c = get().config
    if (!c) return
    get().setConfig({
      ...c, cues: c.cues.map((x) => (x.kind === kind ? { ...x, [field]: value } : x)),
    })
  },

  patchLoops(patch) {
    const c = get().config
    if (!c) return
    if ('color' in patch && patch.color) {
      const prep = get().prep
      if (prep) set({ prep: { ...prep, loops: prep.loops.map((l) => ({ ...l, color: patch.color! })) } })
    }
    get().setConfig({ ...c, loops: { ...c.loops, ...patch } })
  },

  setLoopLocation(kind, enabled) {
    const c = get().config
    if (!c) return
    get().setConfig({
      ...c,
      loops: {
        ...c.loops,
        locations: c.loops.locations.map((l) => (l.kind === kind ? { ...l, enabled } : l)),
      },
    })
  },

  select(id) { set({ selectedId: id }) },
  selectSection(i) { set({ selectedSection: i }) },

  /** Move one section boundary. Snaps to a bar; overlap with a neighbour is
   *  allowed, because a hand-made overlap is a statement about the music that
   *  the detector cannot make on its own. */
  moveSectionEdge(index, which, t) {
    const { analysis } = get()
    if (!analysis) return
    const snapped = snapTo(t, analysis, 'downbeat')
    const bar = (60 / analysis.beat_grid.bpm) * analysis.beat_grid.beats_per_bar
    let moved: { start: number; end: number } | undefined
    const sections = analysis.sections.map((s, i) => {
      if (i !== index) return s
      const start = which === 'start' ? Math.max(0, snapped) : s.start_sec
      const end = which === 'end' ? snapped : s.end_sec
      if (end - start < bar) return s          // never collapse below one bar
      moved = { start, end }
      return {
        ...s, start_sec: start, end_sec: end, is_manual: true, confidence: 1,
        length_bars: Math.round(((end - start) / bar) * 100) / 100,
        start_bar: Math.round((start - analysis.beat_grid.anchor_sec) / bar),
      }
    })
    const m = moved as { start: number; end: number } | undefined
    if (!m) return
    set({ analysis: { ...analysis, sections } })
    // Drags fire on every pointer move; commit once the user stops.
    pendingEdit = { index, start_sec: m.start, end_sec: m.end, is_manual: true }
    if (sectionCommitTimer) clearTimeout(sectionCommitTimer)
    sectionCommitTimer = setTimeout(() => {
      const e = pendingEdit
      pendingEdit = null
      if (e) void get().applySectionEdits([e])
    }, 600)
  },

  relabelSection(index, label) {
    const { analysis } = get()
    if (!analysis) return
    // Optimistic local update so the dropdown responds instantly.
    set({
      analysis: {
        ...analysis,
        sections: analysis.sections.map((s, i) => (
          i === index
            ? { ...s, label, is_manual: true, confidence: 1,
                ambiguous_with: null, ambiguity_margin: null }
            : s)),
      },
    })
    void get().applySectionEdits([{ index, label, is_manual: true }])
  },

  deleteSection(index) {
    set({ selectedSection: null })
    void get().applySectionEdits([{ index, delete: true }])
  },

  /** Split a section in two. The quickest way to fix an under-segmented track
   *  without hunting for boundary handles. */
  splitSection(index, at) {
    const { analysis } = get()
    if (!analysis) return
    const src = analysis.sections[index]
    if (!src) return
    const bar = (60 / analysis.beat_grid.bpm) * analysis.beat_grid.beats_per_bar
    const cut = snapTo(at, analysis, 'downbeat')
    if (cut <= src.start_sec + bar || cut >= src.end_sec - bar) return
    void get().applySectionEdits([
      { index, end_sec: cut, is_manual: true },
      { start_sec: cut, end_sec: src.end_sec, label: src.label },
    ])
  },

  /** Send only what actually changed.
   *
   *  Sending the whole section list on every edit is the obvious thing and it is
   *  wrong: the server marks everything it receives as hand-edited, so
   *  relabelling one section would brand all of them as yours and make them
   *  immune to re-analysis. Targeted edits keep "the user decided this" meaning
   *  what it says.
   */
  async applySectionEdits(edits) {
    const { trackId } = get()
    if (!trackId || edits.length === 0) return
    try {
      const fresh = await api.editSections(trackId, edits)
      set({ analysis: fresh })
      await get().reRecommend()
    } catch (e) {
      set({ error: String(e) })
    }
  },

  /** Halve or double the tempo when the detector picked the wrong octave. */
  async scaleBpm(factor) {
    const { trackId } = get()
    if (!trackId) return
    set({ busy: true, status: factor === 2 ? 'Doubling tempo…' : 'Halving tempo…' })
    try {
      const fresh = await api.scaleGrid(trackId, factor)
      set({ analysis: fresh })
      const prep = await api.preparation(trackId)
      set({ prep, status: `Grid rebuilt at ${fresh.beat_grid.bpm.toFixed(2)} BPM` })
    } catch (e) {
      set({ error: String(e), status: 'Ready' })
    } finally {
      set({ busy: false })
    }
  },

  moveCue(id, t) {
    const { prep, analysis, config } = get()
    if (!prep) return
    const cue = prep.cues.find((c) => c.id === id)
    if (!cue) return
    const mode = config?.cues.find((c) => c.kind === cue.kind)?.snap ?? 'downbeat'
    const snapped = Math.max(0, snapTo(t, analysis, mode))
    const g = analysis?.beat_grid
    set({
      prep: {
        ...prep,
        cues: prep.cues.map((c) => (c.id === id ? {
          ...c, time_sec: snapped,
          source: c.source === 'ai' ? 'ai_edited' : c.source,
          bar_index: g ? Math.round(((snapped - g.anchor_sec) / (60 / g.bpm) - g.downbeat_offset) / g.beats_per_bar) : c.bar_index,
        } : c)),
      },
      pendingFeedback: [...get().pendingFeedback, {
        marker_id: id, marker_class: 'cue', kind: cue.kind, action: 'moved',
        original_time: cue.original_time_sec, final_time: snapped,
        confidence: cue.confidence,
      }],
    })
  },

  deleteCue(id) {
    const { prep } = get()
    if (!prep) return
    const cue = prep.cues.find((c) => c.id === id)
    set({
      prep: { ...prep, cues: prep.cues.filter((c) => c.id !== id) },
      selectedId: get().selectedId === id ? null : get().selectedId,
      pendingFeedback: cue ? [...get().pendingFeedback, {
        marker_id: id, marker_class: 'cue', kind: cue.kind, action: 'deleted',
        original_time: cue.original_time_sec, final_time: null, confidence: cue.confidence,
      }] : get().pendingFeedback,
    })
  },

  setCueAccepted(id, accepted) {
    const { prep } = get()
    if (!prep) return
    const cue = prep.cues.find((c) => c.id === id)
    set({
      prep: { ...prep, cues: prep.cues.map((c) => (c.id === id ? { ...c, accepted } : c)) },
      pendingFeedback: cue && accepted !== null ? [...get().pendingFeedback, {
        marker_id: id, marker_class: 'cue', kind: cue.kind,
        action: accepted ? 'accepted' : 'rejected',
        original_time: cue.original_time_sec, final_time: cue.time_sec,
        confidence: cue.confidence,
      }] : get().pendingFeedback,
    })
  },

  recolorCue(id, color) {
    const { prep } = get()
    if (!prep) return
    set({ prep: { ...prep, cues: prep.cues.map((c) => (c.id === id ? { ...c, color } : c)) } })
  },

  renameCue(id, label) {
    const { prep } = get()
    if (!prep) return
    set({ prep: { ...prep, cues: prep.cues.map((c) => (c.id === id ? { ...c, label } : c)) } })
  },

  addCue(t, kind) {
    const { prep, analysis, config } = get()
    if (!prep) return
    const tc = config?.cues.find((c) => c.kind === kind)
    const snapped = Math.max(0, snapTo(t, analysis, tc?.snap ?? 'downbeat'))
    const g = analysis?.beat_grid
    const cue: CuePoint = {
      id: Math.random().toString(16).slice(2, 14),
      label: tc?.label_template || kind.replace('_', ' ').replace(/\b\w/g, (m) => m.toUpperCase()),
      color: tc?.color ?? { r: 148, g: 163, b: 184 },
      source: 'user', confidence: null, reasons: [], accepted: true,
      original_time_sec: null, time_sec: snapped, kind,
      beat_index: g ? Math.round((snapped - g.anchor_sec) / (60 / g.bpm)) : null,
      bar_index: g ? Math.round(((snapped - g.anchor_sec) / (60 / g.bpm) - g.downbeat_offset) / g.beats_per_bar) : null,
      on_downbeat: true, on_phrase: (tc?.snap ?? 'downbeat') === 'phrase',
    }
    set({ prep: { ...prep, cues: [...prep.cues, cue].sort((a, b) => a.time_sec - b.time_sec) },
          selectedId: cue.id })
  },

  moveLoop(id, start, end) {
    const { prep, analysis } = get()
    if (!prep) return
    const s = Math.max(0, snapTo(start, analysis, 'downbeat'))
    const e = Math.max(s + 0.1, snapTo(end, analysis, 'downbeat'))
    const bar = analysis ? (60 / analysis.beat_grid.bpm) * analysis.beat_grid.beats_per_bar : 1
    set({
      prep: {
        ...prep,
        loops: prep.loops.map((l) => (l.id === id ? {
          ...l, start_sec: s, end_sec: e,
          length_bars: Math.round(((e - s) / bar) * 100) / 100,
          source: l.source === 'ai' ? 'ai_edited' : l.source,
        } : l)),
      },
    })
  },

  setLoopBars(id, bars) {
    const { prep, analysis } = get()
    if (!prep || !analysis) return
    const bar = (60 / analysis.beat_grid.bpm) * analysis.beat_grid.beats_per_bar
    set({
      prep: {
        ...prep,
        loops: prep.loops.map((l) => (l.id === id ? {
          ...l, end_sec: l.start_sec + bars * bar, length_bars: bars,
          source: l.source === 'ai' ? 'ai_edited' : l.source,
        } : l)),
      },
    })
  },

  deleteLoop(id) {
    const { prep } = get()
    if (!prep) return
    set({ prep: { ...prep, loops: prep.loops.filter((l) => l.id !== id) } })
  },

  async save() {
    const { trackId, prep, pendingFeedback } = get()
    if (!trackId || !prep) return
    set({ status: 'Saving…' })
    try {
      await api.savePreparation(trackId, prep)
      if (pendingFeedback.length) {
        await api.feedback(trackId, pendingFeedback)
        set({ pendingFeedback: [] })
      }
      set({ status: 'Saved' })
    } catch (e) {
      set({ error: String(e), status: 'Save failed' })
    }
  },
}))

export type { Loop, CuePoint }
