/**
 * Types mirroring the backend's Pydantic models.
 *
 * These are hand-written here for readability, but the backend's OpenAPI schema
 * is the source of truth: `scripts/gen_types.sh` regenerates this file from
 * `/openapi.json`, so a change to `djprep/models/config.py` propagates here
 * rather than being duplicated by hand. There is exactly one definition of what
 * the user is allowed to ask for.
 */

export interface RGB { r: number; g: number; b: number }

export type CueKind =
  | 'initial' | 'intro' | 'mix_in' | 'buildup' | 'pre_drop' | 'drop' | 'breakdown'
  | 'chorus' | 'vocal_in' | 'vocal_out' | 'mix_out' | 'outro' | 'tempo_change'
  | 'fx' | 'custom'

export type LoopKind = 'intro' | 'outro' | 'breakdown' | 'pre_drop' | 'drop' | 'vocal' | 'custom'
export type SectionLabel =
  | 'intro' | 'verse' | 'buildup' | 'drop' | 'chorus' | 'breakdown'
  | 'bridge' | 'outro' | 'unknown'
export type Source = 'ai' | 'ai_edited' | 'user'
export type SnapMode = 'beat' | 'downbeat' | 'phrase'
export type LoopLength = '1' | '2' | '4' | '8' | '16' | '32' | 'auto'
export type LoopSelection = 'first' | 'all' | 'top_n'
export type MixOffset = '4' | '8' | '16' | '32'
export type VocalMode = 'off' | 'heuristic' | 'separation'

export interface Reason { code: string; text: string; weight: number; value: number | null }

export interface CueTypeConfig {
  kind: CueKind
  enabled: boolean
  color: RGB
  max_count: number
  select_all: boolean
  min_confidence: number
  snap: SnapMode
  offset_bars: number
  label_template: string
}

export interface LoopLocationConfig { kind: LoopKind; enabled: boolean }

export interface LoopConfig {
  enabled: boolean
  locations: LoopLocationConfig[]
  length: LoopLength
  color: RGB
  selection: LoopSelection
  top_n: number
  min_confidence: number
  avoid_vocal_cuts: boolean
}

export interface MixConfig {
  mix_in_bars: MixOffset
  mix_out_bars: MixOffset
}

export interface AnalysisConfig {
  cues: CueTypeConfig[]
  loops: LoopConfig
  mix: MixConfig
  vocal_mode: VocalMode
  bpm_hint: number | null
  bpm_range: [number, number]
  beats_per_bar: number
  phrase_bars_hint: number | null
  min_cue_spacing_bars: number
  min_label_confidence: number
  max_total_cues: number
}

export interface BeatGrid {
  bpm: number; anchor_sec: number; beats_per_bar: number
  downbeat_offset: number; n_beats: number; is_constant: boolean
  confidence: number; downbeat_confidence: number
}

export interface PhraseGrid {
  phrase_bars: number; phase_bars: number; boundaries_sec: number[]
  confidence: number; alt_lengths: Record<string, number>
}

export interface TempoChange {
  time_sec: number
  bpm_before: number
  bpm_after: number
  confidence: number
}

export interface LabelSummary {
  present: string[]
  absent: string[]
  suppressed: string[]
  counts: Record<string, number>
  notes: string[]
}

export interface Section {
  start_sec: number; end_sec: number; start_bar: number; length_bars: number
  label: SectionLabel; confidence: number; cluster_id: number
  energy_pct: number; low_energy_pct: number; high_energy_pct: number
  onset_density_pct: number; brightness_pct: number; energy_slope: number
  vocal_ratio: number; label_scores: Record<string, number>
  ambiguous_with: string | null
  ambiguity_margin: number | null
  is_manual: boolean
  occurrence: number
}

export interface SectionEdit {
  index?: number | null
  start_sec?: number | null
  end_sec?: number | null
  label?: string | null
  is_manual?: boolean | null
  delete?: boolean
}

export interface VocalSegment { start_sec: number; end_sec: number; confidence: number }

export interface EnergyCurve {
  times_sec: number[]; rms_db: number[]; low: number[]; mid: number[]
  high: number[]; onset_density: number[]; vocal_likelihood: number[]
}

export interface TrackMeta {
  track_id: string; filename: string; duration_sec: number
  sample_rate: number; channels: number; title: string | null; artist: string | null
}

export interface TrackAnalysis {
  meta: TrackMeta; beat_grid: BeatGrid; phrase_grid: PhraseGrid
  sections: Section[]; tempo_changes: TempoChange[]; label_summary: LabelSummary
  vocals: VocalSegment[]; energy: EnergyCurve
  key: string | null; analysis_version: string; timings_ms: Record<string, number>
}

export interface CuePoint {
  id: string; label: string; color: RGB; source: Source
  confidence: number | null; reasons: Reason[]; accepted: boolean | null
  original_time_sec: number | null
  time_sec: number; kind: CueKind
  beat_index: number | null; bar_index: number | null
  on_downbeat: boolean; on_phrase: boolean
}

export interface Loop {
  id: string; label: string; color: RGB; source: Source
  confidence: number | null; reasons: Reason[]; accepted: boolean | null
  original_time_sec: number | null
  start_sec: number; end_sec: number; length_bars: number; kind: LoopKind
  start_beat_index: number | null; start_bar_index: number | null
}

export interface TrackPreparation {
  track_id: string; cues: CuePoint[]; loops: Loop[]
  cue_priority: string[]; notes: string
}

export interface ExportFormat {
  format_id: string; display_name: string; file_extension: string
  max_hot_cues: number | null; supports_loops: boolean; supports_cue_names: boolean
  color_mode: string; palette_size: number; palette: RGB[]; notes: string
}

export interface LoweringNote {
  severity: 'info' | 'adjusted' | 'dropped'
  code: string; message: string; marker_id: string | null
}

export interface LoweringReport {
  target: string; lossless: boolean
  counts: Record<string, number>; notes: LoweringNote[]
}

export interface Job {
  id: string; kind: string; status: 'queued' | 'running' | 'done' | 'error'
  progress: number; stage: string; error: string | null; elapsed_sec: number
  result?: unknown
}

export const hex = (c: RGB): string =>
  '#' + [c.r, c.g, c.b].map((v) => v.toString(16).padStart(2, '0')).join('').toUpperCase()

export const fromHex = (s: string): RGB => ({
  r: parseInt(s.slice(1, 3), 16),
  g: parseInt(s.slice(3, 5), 16),
  b: parseInt(s.slice(5, 7), 16),
})

export const rgba = (c: RGB, a: number): string => `rgba(${c.r},${c.g},${c.b},${a})`

export const CUE_KIND_LABELS: Record<CueKind, string> = {
  initial: 'Start (always on)', intro: 'Intro', mix_in: 'Mix In',
  buildup: 'Build-up', pre_drop: 'Pre-Drop', drop: 'Drop',
  breakdown: 'Breakdown', chorus: 'Chorus', vocal_in: 'Vocal In',
  vocal_out: 'Vocal Out', mix_out: 'Mix Out', outro: 'Outro',
  tempo_change: 'Tempo Change', fx: 'FX Point', custom: 'Custom Cue',
}

/** Which section label a cue type depends on, for greying out what can't apply. */
export const CUE_REQUIRES_SECTION: Partial<Record<CueKind, SectionLabel>> = {
  intro: 'intro', buildup: 'buildup', pre_drop: 'buildup', drop: 'drop',
  breakdown: 'breakdown', chorus: 'chorus', outro: 'outro',
}

export const SECTION_LABELS: SectionLabel[] = [
  'intro', 'verse', 'buildup', 'drop', 'chorus', 'breakdown', 'bridge',
  'outro', 'unknown',
]

export const LOOP_KIND_LABELS: Record<LoopKind, string> = {
  intro: 'Intro', outro: 'Outro', breakdown: 'Breakdown',
  pre_drop: 'Pre-drop', drop: 'Drop', vocal: 'Vocal section', custom: 'Custom',
}

export const SECTION_COLORS: Record<SectionLabel, string> = {
  intro: '#16A34A', verse: '#0EA5E9', buildup: '#EAB308', drop: '#3B82F6',
  chorus: '#06B6D4', breakdown: '#A855F7', bridge: '#8B5CF6', outro: '#EA580C',
  unknown: '#64748B',
}

export const fmtTime = (t: number): string => {
  const m = Math.floor(t / 60)
  const s = t - m * 60
  return `${m}:${s.toFixed(2).padStart(5, '0')}`
}
