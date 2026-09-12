import type {
  AnalysisConfig, ExportFormat, Job, LoweringReport, SectionEdit, TrackAnalysis,
  TrackPreparation,
} from './types'

const BASE = '/api'

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${body.slice(0, 300)}`)
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T)
}

export const api = {
  health: () => req<{ status: string; version: string }>('/health'),

  defaultConfig: () => req<AnalysisConfig>('/config/default'),

  exportFormats: () => req<ExportFormat[]>('/export/formats'),

  listTracks: () => req<Array<{ id: string; filename: string; duration_sec: number }>>('/tracks'),

  async upload(file: File): Promise<{ track_id: string; filename: string; duration_sec: number }> {
    const fd = new FormData()
    fd.append('file', file)
    const res = await fetch(`${BASE}/tracks`, { method: 'POST', body: fd })
    if (!res.ok) throw new Error(`upload failed: ${res.status} ${await res.text()}`)
    return res.json()
  },

  analyze: (trackId: string, config: AnalysisConfig) =>
    req<Job>(`/tracks/${trackId}/analyze`, { method: 'POST', body: JSON.stringify(config) }),

  job: (jobId: string) => req<Job>(`/jobs/${jobId}`),

  analysis: (trackId: string) => req<TrackAnalysis>(`/tracks/${trackId}/analysis`),

  preparation: (trackId: string) => req<TrackPreparation>(`/tracks/${trackId}/preparation`),

  savePreparation: (trackId: string, prep: TrackPreparation) =>
    req<TrackPreparation>(`/tracks/${trackId}/preparation`, {
      method: 'PUT', body: JSON.stringify(prep),
    }),

  /** Cheap: re-runs recommendation only, no DSP. Called on every config change. */
  recommend: (trackId: string, config: AnalysisConfig) =>
    req<TrackPreparation>(`/tracks/${trackId}/recommend`, {
      method: 'POST', body: JSON.stringify(config),
    }),

  waveform: (trackId: string, points = 2000) =>
    req<{ peaks: number[]; duration_sec: number }>(
      `/tracks/${trackId}/waveform?points=${points}`),

  audioUrl: (trackId: string) => `${BASE}/tracks/${trackId}/audio`,

  exportPreview: (trackId: string, formatId: string, prep: TrackPreparation) =>
    req<{ bytes: number; preview: string; report: LoweringReport }>(
      `/tracks/${trackId}/export/preview?format_id=${formatId}`,
      { method: 'POST', body: JSON.stringify(prep) }),

  async download(trackId: string, formatId: string, prep: TrackPreparation): Promise<void> {
    const res = await fetch(`${BASE}/tracks/${trackId}/export?format_id=${formatId}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(prep),
    })
    if (!res.ok) throw new Error(`export failed: ${res.status}`)
    const blob = await res.blob()
    const cd = res.headers.get('Content-Disposition') ?? ''
    const name = /filename="(.+?)"/.exec(cd)?.[1] ?? `export${formatId}`
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = name
    a.click()
    URL.revokeObjectURL(url)
  },

  /** Halve or double the detected tempo. Fast: reuses cached spectral features. */
  scaleGrid: (trackId: string, factor: 0.5 | 2) =>
    req<TrackAnalysis>(`/tracks/${trackId}/grid/scale?factor=${factor}`,
      { method: 'POST' }),

  /** Apply hand edits to the detected sections. */
  editSections: (trackId: string, edits: SectionEdit[]) =>
    req<TrackAnalysis>(`/tracks/${trackId}/sections`, {
      method: 'PUT', body: JSON.stringify(edits),
    }),

  feedback: (trackId: string, items: unknown[]) =>
    req<{ logged: number }>(`/tracks/${trackId}/feedback`, {
      method: 'POST', body: JSON.stringify(items),
    }),
}
