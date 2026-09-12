/**
 * Export, with the lowering report shown *before* the download.
 *
 * The interesting UI decision here: the report is not an error log, it is a
 * preview of what the target format cannot represent. A DJ who is told "your
 * eleventh cue will become a memory cue and your teal will come out turquoise"
 * before exporting can go and reorder their priorities; one who finds out
 * afterwards, in Rekordbox, cannot.
 */
import { useState } from 'react'
import { api } from '../api/client'
import { useStore } from '../state/store'
import type { LoweringReport } from '../api/types'

export function ExportPanel({ onClose }: { onClose: () => void }) {
  const { trackId, prep, formats } = useStore()
  const [formatId, setFormatId] = useState('rekordbox_xml')
  const [report, setReport] = useState<LoweringReport | null>(null)
  const [preview, setPreview] = useState('')
  const [busy, setBusy] = useState(false)
  const fmt = formats.find((f) => f.format_id === formatId)

  const active = prep?.cues.filter((c) => c.accepted !== false) ?? []
  const overflow = fmt?.max_hot_cues != null
    ? Math.max(0, active.length + (prep?.loops.length ?? 0) - fmt.max_hot_cues) : 0

  const check = async () => {
    if (!trackId || !prep) return
    setBusy(true)
    try {
      const r = await api.exportPreview(trackId, formatId, prep)
      setReport(r.report)
      setPreview(r.preview)
    } finally { setBusy(false) }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <header>
          <h2>Export</h2>
          <button className="btn ghost" onClick={onClose}>✕</button>
        </header>

        <div className="modal-body">
          <div className="format-picker">
            {formats.map((f) => (
              <label key={f.format_id} className={`format${formatId === f.format_id ? ' on' : ''}`}>
                <input type="radio" name="fmt" checked={formatId === f.format_id}
                       onChange={() => { setFormatId(f.format_id); setReport(null) }} />
                <div>
                  <strong>{f.display_name}</strong>
                  <p>{f.notes}</p>
                  <div className="caps">
                    <span>{f.max_hot_cues == null ? 'unlimited cues' : `${f.max_hot_cues} hot cues`}</span>
                    <span>{f.color_mode === 'palette' ? `${f.palette_size}-colour palette` : 'any colour'}</span>
                    <span>{f.supports_loops ? 'loops' : 'no loops'}</span>
                  </div>
                </div>
              </label>
            ))}
          </div>

          {overflow > 0 && (
            <div className="warn">
              You have {active.length} cues and {prep?.loops.length ?? 0} loops but this
              format has {fmt?.max_hot_cues} hot cue slots. {overflow} will be written as
              memory cues — reorder priority in the list to choose which.
            </div>
          )}

          {fmt?.palette && fmt.palette.length > 0 && (
            <div className="palette-row">
              <span className="sub">Target palette</span>
              <div className="palette">
                {fmt.palette.map((p, i) => (
                  <i key={i} style={{ background: `rgb(${p.r},${p.g},${p.b})` }} />
                ))}
              </div>
            </div>
          )}

          <div className="modal-actions">
            <button className="btn" onClick={check} disabled={busy}>
              {busy ? 'Checking…' : 'Check what will be lost'}
            </button>
            <button className="btn primary"
                    onClick={() => trackId && prep && api.download(trackId, formatId, prep)}>
              Download {fmt?.file_extension}
            </button>
          </div>

          {report && (
            <div className="report">
              <div className={`report-head ${report.lossless ? 'ok' : 'warnish'}`}>
                {report.lossless
                  ? 'Nothing is lost in this export.'
                  : `${report.counts.adjusted ?? 0} adjusted, ${report.counts.dropped ?? 0} dropped`}
              </div>
              <ul>
                {report.notes.map((n, i) => (
                  <li key={i} className={n.severity}>
                    <span className="sev">{n.severity}</span>{n.message}
                  </li>
                ))}
              </ul>
              {preview && <pre className="xml-preview">{preview.slice(0, 4000)}</pre>}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
