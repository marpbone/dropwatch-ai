import { useEffect, useRef, useState } from 'react'
import logo from './assets/dropwatch.png'
import { ConfigPanel } from './components/ConfigPanel'
import { CueList } from './components/CueList'
import { ExportPanel } from './components/ExportPanel'
import { WaveformEditor } from './components/WaveformEditor'
import { useStore } from './state/store'

export default function App() {
  const { bootstrap, uploadAndAnalyze, loadTrack, save, status, progress, busy,
          error, analysis, filename, prep } = useStore()
  const fileRef = useRef<HTMLInputElement>(null)
  const [showExport, setShowExport] = useState(false)

  useEffect(() => {
    void bootstrap()
    void fetch('/api/tracks').then((r) => r.json()).then((ts) => {
      if (Array.isArray(ts) && ts.length > 0) void loadTrack(ts[0].id)
    }).catch(() => {})
  }, [bootstrap, loadTrack])

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img src={logo} alt="" width={34} height={34} className="logo" />
          <h1>Dropwatch&nbsp;AI</h1>
        </div>

        <div className="track-info">
          {analysis ? (
            <>
              <strong title={filename}>{filename}</strong>
              <span className="chip">{analysis.beat_grid.bpm.toFixed(2)} BPM</span>
              <span className="chip">{analysis.sections.length} sections</span>
              <span className="chip">{prep?.cues.length ?? 0} cues</span>
              <span className="chip">{prep?.loops.length ?? 0} loops</span>
            </>
          ) : <span className="muted">No track loaded</span>}
        </div>

        <div className="actions">
          <input ref={fileRef} type="file" accept=".mp3,.wav,.flac,.aiff,.m4a,.ogg"
                 style={{ display: 'none' }}
                 onChange={(e) => { const f = e.target.files?.[0]; if (f) void uploadAndAnalyze(f) }} />
          <button className="btn" onClick={() => fileRef.current?.click()} disabled={busy}>
            Upload track
          </button>
          <button className="btn" onClick={() => void save()} disabled={!prep}>Save</button>
          <button className="btn primary" onClick={() => setShowExport(true)} disabled={!prep}>
            Export
          </button>
        </div>
      </header>

      {busy && (
        <div className="progress">
          <div className="progress-bar" style={{ width: `${progress * 100}%` }} />
          <span>{status}</span>
        </div>
      )}
      {error && <div className="error-bar">{error}</div>}

      <main className="layout">
        <ConfigPanel />
        <div className="center"><WaveformEditor /></div>
        <CueList />
      </main>

      {showExport && <ExportPanel onClose={() => setShowExport(false)} />}
    </div>
  )
}
