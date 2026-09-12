/**
 * The review list: every recommendation, why it was made, and accept/reject.
 *
 * This is where the "explainable" part of the project has to earn its keep. A
 * confidence number on its own is not an explanation and a DJ has no reason to
 * trust it. Each reason shown here is a term from the scoring model with its
 * actual signed contribution to the decision, sorted by magnitude -- so the list
 * is not a plausible-sounding narrative generated after the fact, it *is* the
 * computation.
 */
import { useState } from 'react'
import { SectionList } from './SectionList'
import { useStore } from '../state/store'
import { CUE_KIND_LABELS, fmtTime, fromHex, hex, type CuePoint } from '../api/types'

const confClass = (c: number | null): string =>
  c === null ? 'manual' : c >= 0.7 ? 'high' : c >= 0.45 ? 'mid' : 'low'

export function CueList() {
  const { prep, analysis, selectedId, select, setCueAccepted, deleteCue, recolorCue,
          renameCue, setLoopBars, deleteLoop } = useStore()
  const [expanded, setExpanded] = useState<string | null>(null)
  if (!prep || !analysis) return null

  const pending = prep.cues.filter((c) => c.source === 'ai' && c.accepted === null).length

  return (
    <aside className="panel right">
      <SectionList />

      <section className="panel-section">
        <h2>
          Cue points
          {pending > 0 && <span className="badge">{pending} to review</span>}
        </h2>

        <div className="cue-list">
          {prep.cues.map((c: CuePoint) => (
            <div key={c.id}
                 className={`cue-card ${confClass(c.confidence)}`
                   + (selectedId === c.id ? ' selected' : '')
                   + (c.accepted === false ? ' rejected' : '')}
                 onClick={() => { select(c.id); setExpanded(expanded === c.id ? null : c.id) }}>
              <div className="cue-card-head">
                <input type="color" value={hex(c.color)} onClick={(e) => e.stopPropagation()}
                       onChange={(e) => recolorCue(c.id, fromHex(e.target.value))} />
                <input className="cue-name" value={c.label} onClick={(e) => e.stopPropagation()}
                       onChange={(e) => renameCue(c.id, e.target.value)} />
                <span className="mono time">{fmtTime(c.time_sec)}</span>
              </div>

              <div className="cue-card-meta">
                <span className="kind">{CUE_KIND_LABELS[c.kind]}</span>
                <span className="bar">bar {c.bar_index ?? '—'}</span>
                {c.on_downbeat && <span className="tag">downbeat</span>}
                {c.on_phrase && <span className="tag">phrase</span>}
                {c.source !== 'ai' && <span className="tag edited">
                  {c.source === 'user' ? 'yours' : 'edited'}
                </span>}
                <span className="spacer" />
                {c.confidence !== null
                  ? <span className="conf">{(c.confidence * 100).toFixed(0)}%</span>
                  : <span className="conf manual">manual</span>}
              </div>

              {c.confidence !== null && (
                <div className="conf-bar"><i style={{ width: `${c.confidence * 100}%` }} /></div>
              )}

              {expanded === c.id && c.reasons.length > 0 && (
                <ul className="reasons">
                  {c.reasons.map((r, i) => (
                    <li key={i} className={r.weight >= 0 ? 'pos' : 'neg'}>
                      <span className="w mono">{r.weight >= 0 ? '+' : ''}{r.weight.toFixed(2)}</span>
                      <span>{r.text}</span>
                    </li>
                  ))}
                </ul>
              )}

              {c.source === 'ai' && (
                <div className="cue-actions" onClick={(e) => e.stopPropagation()}>
                  <button className={`btn tiny${c.accepted === true ? ' on' : ''}`}
                          onClick={() => setCueAccepted(c.id, c.accepted === true ? null : true)}>
                    ✓ Keep
                  </button>
                  <button className={`btn tiny${c.accepted === false ? ' on' : ''}`}
                          onClick={() => setCueAccepted(c.id, c.accepted === false ? null : false)}>
                    ✕ Reject
                  </button>
                  <button className="btn tiny ghost" onClick={() => deleteCue(c.id)}>Delete</button>
                </div>
              )}
              {c.source !== 'ai' && (
                <div className="cue-actions" onClick={(e) => e.stopPropagation()}>
                  <button className="btn tiny ghost" onClick={() => deleteCue(c.id)}>Delete</button>
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      {prep.loops.length > 0 && (
        <section className="panel-section">
          <h2>Loops</h2>
          <div className="cue-list">
            {prep.loops.map((l) => (
              <div key={l.id} className={`cue-card ${confClass(l.confidence)}`
                     + (selectedId === l.id ? ' selected' : '')}
                   onClick={() => { select(l.id); setExpanded(expanded === l.id ? null : l.id) }}>
                <div className="cue-card-head">
                  <span className="swatch" style={{ background: hex(l.color) }} />
                  <span className="cue-name static">{l.label}</span>
                  <span className="mono time">{fmtTime(l.start_sec)}</span>
                </div>
                <div className="cue-card-meta">
                  <span className="bar">bar {l.start_bar_index ?? '—'}</span>
                  <select value={String(l.length_bars)} onClick={(e) => e.stopPropagation()}
                          onChange={(e) => setLoopBars(l.id, Number(e.target.value))}>
                    {[1, 2, 4, 8, 16, 32].map((b) => <option key={b} value={b}>{b} bars</option>)}
                  </select>
                  <span className="spacer" />
                  {l.confidence !== null && <span className="conf">{(l.confidence * 100).toFixed(0)}%</span>}
                </div>
                {expanded === l.id && l.reasons.length > 0 && (
                  <ul className="reasons">
                    {l.reasons.map((r, i) => (
                      <li key={i} className={r.weight >= 0 ? 'pos' : 'neg'}>
                        <span className="w mono">{r.weight >= 0 ? '+' : ''}{r.weight.toFixed(2)}</span>
                        <span>{r.text}</span>
                      </li>
                    ))}
                  </ul>
                )}
                <div className="cue-actions" onClick={(e) => e.stopPropagation()}>
                  <button className="btn tiny ghost" onClick={() => deleteLoop(l.id)}>Delete</button>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </aside>
  )
}
