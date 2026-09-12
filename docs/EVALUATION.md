# How to tell whether the recommendations are actually good

This is the question that separates a project you can make claims about from one
you cannot, and it is the one most portfolio projects skip.

## Three levels, in increasing order of what they tell you

### 1. Synthetic fixture (automatic, in CI, 40 s)

`scripts/make_test_track.py` renders a track whose tempo, downbeat phase, phrase
length, section boundaries and vocal regions are known *by construction*. CI
asserts against them to the bar.

This cannot tell you the system works on music. It catches exactly the failures
that are catastrophic and silent: tempo octave errors, downbeat off by one,
boundaries drifting off the phrase grid, a refactor that shifts every cue by a
frame. Those are worth catching in 40 seconds rather than in a club.

Current: BPM exact, anchor 6.8 ms, downbeat correct, boundary F1 1.00, label
accuracy 99.5%, vocal F1 0.91.

### 2. Annotated real tracks (manual, the one that matters)

40 tracks you know well. For each, a JSON file (`scripts/evaluate.py` documents
the schema) with BPM, one known downbeat, phrase length, section boundaries and
labels, vocal regions, and the cue points you would set yourself.

Two hours of work. Do it **before** writing the recommender, so you are measuring
rather than rationalising.

```bash
python scripts/evaluate.py annotations/*.json --csv results.csv
```

Metrics and why each:

| Metric | What a failure means |
|---|---|
| BPM exact / octave-tolerant, reported separately | An octave error and a 2 BPM error are different bugs with different fixes |
| Downbeat phase accuracy | Binary and unforgiving, because it is: off by one ruins every phrase-aligned cue |
| Boundary P/R/F1 at ½-bar and 1-bar tolerance | Tolerance in *bars*, not seconds — musical tolerance scales with tempo |
| Frame-level label accuracy | Reported separately from boundaries: F1 0.9 on boundaries with 60% labels is still useful, and a blended number would hide it |
| Vocal F1 | Frame-level |
| Cue distance in bars, per type, plus fraction within ½ bar | Per type, because drop and vocal cues fail for unrelated reasons |

**Stratify your set.** Include material you expect to fail — live drums, tempo
changes, ambient intros, half-time sections. Reporting "F1 0.92 on
four-to-the-floor, 0.61 on anything with live drums" is a much stronger result
than a single average, because it shows you know where the boundary of the
approach is.

### 3. The DJ's own decisions (continuous, free, and the best signal)

Every accept, reject and drag in the editor is logged to the `feedback` table
with the evidence that produced the recommendation. This is a labelled dataset
that grows every time the tool is used, with no annotation effort at all.

```bash
curl localhost:8000/api/feedback/stats
python scripts/refit_weights.py --dry-run
```

Two distinct things to read from it:

**Acceptance rate per cue type.** If drop cues are kept 90% of the time and vocal
cues 40%, you know exactly where to spend your next week.

**Systematic drag offset.** The *median* bars a cue type gets moved. Scattered
residuals mean a weighting problem — refit the model. A consistent offset means
something else entirely: the detector's anchor is wrong, and no amount of
reweighting will fix it. `refit_weights.py` flags this separately for that reason.

## What "good" looks like

There is no benchmark for DJ cue placement, so define the target by what the tool
is *for*: preparing a track faster than doing it by hand.

- **The beat grid must be near-perfect.** A wrong grid makes everything
  downstream useless. Target >98% downbeat accuracy on four-to-the-floor
  material. Below that, fix this before anything else.
- **Cue placement precision beats recall.** A missing cue costs a few seconds to
  add. A confidently wrong one costs trust, and once a DJ stops believing the
  confidence numbers the feature is dead. Tune thresholds toward precision.
- **The explanation must survive scrutiny.** If a DJ reads "16-bar build-up
  preceding it" and there is no build-up, the whole premise fails. This is why
  the reasons are the model's own terms rather than generated prose.

## Reporting it honestly

Put the numbers in the README, including the bad ones, with the evaluation set
described. "Boundary F1 0.87 across 40 tracks; 0.94 on four-to-the-floor and 0.62
on tracks with live drums, which the constant-tempo grid assumption does not fit"
is a sentence that demonstrates engineering judgment. "Uses AI to find cue points"
is not.
