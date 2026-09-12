/**
 * The section review and editing list.
 *
 * Detection produces a first draft; this is where a DJ corrects it. Relabelling
 * is a dropdown rather than a modal because it is the single most common edit
 * and it should cost one click. Sections the analyser found genuinely ambiguous
 * say so in place, with the runner-up named, so the correction is a confirmation
 * rather than a guess.
 *
 * Anything edited here is marked as yours and is never overwritten by a
 * re-analysis.
 */
import { useStore } from '../state/store'
import {
  SECTION_COLORS, SECTION_LABELS, fmtTime, type SectionLabel,
} from '../api/types'

export function SectionList() {
  const { analysis, selectedSection, selectSection, relabelSection,
          deleteSection } = useStore()
  if (!analysis) return null
  const summary = analysis.label_summary

  return (
    <section className="panel-section">
      <h2>
        Sections
        <span className="badge quiet">{analysis.sections.length}</span>
      </h2>

      {summary.notes.length > 0 && (
        <ul className="summary-notes">
          {summary.notes.map((n, i) => <li key={i}>{n}</li>)}
        </ul>
      )}

      <div className="section-list">
        {analysis.sections.map((s, i) => (
          <div key={i}
               className={`section-row${selectedSection === i ? ' selected' : ''}`}
               style={{ borderLeftColor: SECTION_COLORS[s.label] }}
               onClick={() => selectSection(i)}>
            <div className="section-row-head">
              <select value={s.label} onClick={(e) => e.stopPropagation()}
                      style={{ color: SECTION_COLORS[s.label] }}
                      onChange={(e) => relabelSection(i, e.target.value as SectionLabel)}>
                {SECTION_LABELS.map((l) => <option key={l} value={l}>{l}</option>)}
              </select>
              {s.occurrence > 1 && <span className="occ">#{s.occurrence}</span>}
              <span className="spacer" />
              <span className="mono time">{fmtTime(s.start_sec)}</span>
              <button className="btn tiny ghost" title="Delete this section"
                      onClick={(e) => { e.stopPropagation(); deleteSection(i) }}>✕</button>
            </div>

            <div className="section-row-meta">
              <span>bar {s.start_bar}</span>
              <span>{s.length_bars.toFixed(0)} bars</span>
              {s.is_manual
                ? <span className="tag edited">yours</span>
                : <span className="conf">{(s.confidence * 100).toFixed(0)}%</span>}
            </div>

            {s.ambiguous_with && !s.is_manual && (
              <div className="ambiguity">
                Could also be <b>{s.ambiguous_with}</b> — pick one to settle it.
                <button className="btn tiny" onClick={(e) => {
                  e.stopPropagation()
                  relabelSection(i, s.ambiguous_with as SectionLabel)
                }}>
                  Use {s.ambiguous_with}
                </button>
              </div>
            )}
          </div>
        ))}
      </div>

      {summary.absent.length > 0 && (
        <p className="absent-note">
          Not found in this track: {summary.absent.join(', ')}.
        </p>
      )}
    </section>
  )
}
