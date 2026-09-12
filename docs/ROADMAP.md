# Roadmap

Calibrated to ~10 hrs/week over 3–4 months. The MVP is deliberately small and the
stretch goals are deliberately labelled as such: a finished, sharp MVP beats an
unfinished ambitious one, and "here is what I would do next and why" is a good
interview answer.

**What is in this repository already** spans MVP and most of Intermediate. The
staging below is how to *present* the work and what to build next.

---

## Phase 0 — before any recommender code (week 1)

**Build the evaluation set first.** 40 tracks you know well, annotated with BPM,
one known downbeat, phrase length, section boundaries and labels, and the cue
points you would actually set. Two hours in a DAW or in Rekordbox itself.

Nothing else in this project is as cheap or as valuable. Without it you cannot
tell a working detector from a plausible one, and every later decision is guesswork.

Also: `scripts/make_test_track.py`. A synthetic track with ground truth *by
construction* catches octave errors, off-by-one downbeat phase and boundaries
drifting off the grid, in CI, in 40 seconds, with no copyrighted audio in the repo.

---

## Phase 1 — MVP (weeks 2–6)

The smallest thing that is genuinely useful to a DJ.

- [x] Upload WAV/MP3, decode to mono
- [x] Constant-tempo beat grid + downbeat phase
- [x] Phrase grid
- [x] Section boundaries + labels
- [x] Cue recommendations for **drop, breakdown, buildup, intro, outro** with
      confidence and reasons
- [x] Waveform editor: see, select, drag, delete, add
- [x] Rekordbox XML export with hot cue colours
- [x] FastAPI backend, React frontend

**Definition of done:** you can drop in one of your own tracks, get cues you would
actually use, fix the two that are wrong, export, and open it in Rekordbox.

Ship it here if you have to stop. This alone is a strong portfolio project.

---

## Phase 2 — Intermediate (weeks 7–12)

- [x] Loop detection with a real seamlessness metric
- [x] Full configuration UI (per-type colours, snap mode, counts, loop config)
- [x] Vocal detection (heuristic) + vocal cues
- [x] Export capability declarations + lowering report
- [x] Feedback logging (accept / reject / drag)
- [x] Evaluation harness
- [ ] **Demucs vocal separation** behind the existing `VocalDetector` interface —
      the one place a pretrained model clearly beats a heuristic
- [ ] **Refit the confidence model** on your own accept/reject data and report
      the before/after accuracy. This is the human-in-the-loop story; it needs
      real usage to be real
- [ ] Batch mode: prepare a folder of tracks overnight
- [ ] Undo/redo in the editor

---

## Phase 3 — Advanced (stretch, pick one or two)

Choose by what you want to *talk about*, not by feature count.

- **Real-time preview engine (C++/JUCE).** Plays the track, fires cues live,
  sample-accurate loop rolls. This is where C++ is genuinely justified: audio
  callback deadlines, lock-free buffers, no allocation on the audio thread. The
  strongest single addition for a Computer Engineering résumé.
- **Learned downbeat tracker, evaluated against the DSP one.** Small TCN on
  beat-synchronous features. The value is the comparison, not the model.
- **Variable-tempo support.** Piecewise-constant grid via changepoint detection.
  Opens up live/acoustic material, where the current constant-tempo assumption
  fails. `BeatGrid.is_constant` already exists for this.
- **Cross-track structure matching.** Given two prepared tracks, suggest
  compatible transition points (phrase-aligned, energy-compatible, key-compatible).
  This is the feature an actual DJ would want most.
- **Rekordbox database write** via `pyrekordbox`, instead of XML import. Removes
  the manual import step entirely. Higher risk: undocumented, encrypted, and it
  writes to the user's real library — implement read-only inspection first and
  back up the database.
- **Serato / Engine DJ exporters**, to prove the abstraction with a target that
  has genuinely different constraints.

---

## Deliberately not doing

Key detection and harmonic mixing · user accounts · cloud sync · playlist
management · stem separation for performance · mobile app · a plugin version.

Each is a reasonable project. None of them makes *this* project better.
