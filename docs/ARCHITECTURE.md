# Architecture

## The three seams that matter

Most of the design is ordinary. Three boundaries carry real weight, and each one
exists because putting it elsewhere causes a specific problem.

### 1. Analysis is separate from recommendation

`TrackAnalysis` is what the DSP found. `TrackPreparation` is what should be done
about it. They are separate models produced by separate stages and stored
separately.

Analysis costs ~10 s and depends only on the audio. Recommendation costs ~40 ms
and depends on user configuration. Fusing them would mean every checkbox toggle
re-runs a spectrogram; separating them makes the configuration panel feel live.
It also means the analysis can be cached, versioned (`analysis_version`) and
re-used when the recommender changes.

Measured: `/recommend` returns in ~40 ms against a 10 s `/analyze`. A test
asserts it stays under 2 s so this cannot silently regress.

### 2. The internal model knows nothing about any DJ software

`djprep.models` has no `hot_cue_slot`, no `Num`, no palette index, no XML. A cue
has a musical meaning, a position, a free 24-bit colour and its reasons.

Everything vendor-specific lives in `djprep.export` and is applied by a
**lowering pass** — the compiler term, used deliberately: a rich representation
is lowered onto a constrained target, and every concession is recorded in a
`LoweringReport` rather than silently applied.

`ExporterCapabilities` declares the constraints as data (`max_hot_cues`,
`color_mode`, `palette`), which lets the UI warn *before* export rather than
letting the user discover the truncation in Rekordbox. Adding Serato or Engine DJ
means a capabilities declaration plus a serialiser — no changes anywhere else.

The report is the proof the decoupling is real. If the internal model were
secretly Rekordbox-shaped there would be nothing to report.

### 3. Scoring weights are data, not code

`data/cue_weights.json` holds a logistic model per cue type. The same file
format serves hand-set priors on day one and weights refitted from user feedback
on day one hundred (`scripts/refit_weights.py`). Refitting is a data update.

## Data flow

```
file ──▶ io.load ──▶ features.extract ──┬──▶ beats.build_beat_grid
        (mono 22k)   (one STFT, reused) │        └─▶ phrases.estimate_phrase_grid
                                        │              └─▶ structure.detect_boundaries
                                        │                    └─▶ merge → stats → cluster
                                        │                          └─▶ labeling (rules + Viterbi)
                                        └──▶ vocals.likelihood ──────────┘
                                                      │
                                                 TrackAnalysis
                                                      │
                                    AnalysisConfig ──▶ reco.engine.recommend
                                                      │
                                                 TrackPreparation
                                                      │
                            ┌─────────────────────────┼──────────────────────┐
                            ▼                         ▼                      ▼
                     waveform editor           export.lowering        feedback table
                     (user edits)              (+ report)             (refit weights)
```

Everything downstream of `features.extract` reads one `Features` object. No stage
recomputes a spectrogram.

## Frontend

Three state slices with different lifecycles: `config` (cheap, drives
`/recommend`), `analysis` (expensive, immutable), `preparation` (the editable
document). Re-running the recommender merges fresh AI output *under* anything the
user touched — an AI proposal can be replaced, a user decision cannot.

The editor is WaveSurfer for waveform rendering and playback with a **custom DOM
overlay** for everything interactive. WaveSurfer's regions plugin models a span
with drag handles, which fits loops but not point cues, and it owns its own
hit-testing, which makes "double-click empty space to add a cue at the nearest
downbeat" awkward. Owning the overlay means cues, loops, section bands and the
phrase grid are ordinary DOM under one coordinate transform.

The waveform renders from **server-computed peaks** (~2400 floats), so it paints
immediately; the audio file streams separately and only for playback.

## Why SQLite, and why one table is relational

Analyses and preparations are document-shaped; they are stored as JSON blobs
because a relational schema would buy migrations and nothing else.

`feedback` is deliberately relational and append-only: it is a training set, and
it wants to be queried by cue kind, by action, by time delta. `delta_bars` is
stored in bars rather than seconds so a 4-bar bias means the same thing at any
tempo.
