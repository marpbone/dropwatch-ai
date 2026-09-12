# Algorithms, and the bugs that shaped them

Each of these was arrived at by watching a simpler version fail on the test
fixture. The failures are recorded because they are the justification.

## Tempo: normalise by √n, not by n

Score a candidate grid by how much onset energy lands on it. The obvious score —
*mean* on-grid energy — is biased toward sparse grids: a slower tempo samples only
the strongest onsets and wins on average while explaining less of the signal.
Scoring by *sum* is biased the other way.

The correct quantity is the normalised correlation between the onset envelope and
a unit-norm impulse train: `sum / √n`. Halving the tempo scores 0.71×; doubling
it scores 0.71× unless the extra beats carry real energy.

> **Observed:** with mean-energy scoring, a 128 BPM track was reported as
> **85.33 BPM** = 128 × ⅔. With √n normalisation it reports 128.000.

A log-normal tempo prior centred at 126 BPM breaks remaining ties between
metrical levels. It is a tiebreaker, never strong enough to override evidence.

## Tempo search: phase folding

Testing each candidate phase separately is O(n_phases × n_beats) per tempo. Fold
every frame's time modulo the beat period and histogram its onset energy instead:
one O(T) pass, in which the peak bin *is* the best phase and its height *is* that
phase's comb score. A wrong tempo makes true beats drift across the fold and
smears the histogram flat — exactly the discrimination wanted.

Coarse-to-fine, because tempo error accumulates as phase drift proportional to
elapsed time: a coarse scan over a 90 s excerpt finds the neighbourhood, then a
0.01 BPM scan against the full track refines it. 14.4 s → 3.4 s.

## Grid anchor: coherent averaging

The onset envelope is computed on 23 ms frames, so its peak lags the true attack
by up to a frame — enough to visibly misalign a grid in DJ software.

Once the period is known, extract a window around every predicted beat, average
them all, and find the steepest rise in the averaged attack at sample resolution.
With ~600 beats this is roughly a 25× SNR improvement on the transient against
anything not locked to the grid. **26 ms → 6.8 ms.**

## Downbeats: the loudest beat is the *wrong* answer

The tempting cue — "the downbeat is the loudest beat" — is not merely weak, it is
actively wrong for most popular and dance music, because the snare or clap sits
on beats 2 and 4. Onset-envelope strength and high-frequency flux both peak on
the backbeat and vote *confidently* for the wrong phase.

> **Observed:** on the fixture, onset strength at candidate downbeats was 3.79
> for phase 1 and 2.46 for phase 0 — a confident vote for the wrong answer.

What actually resolves phase modulo the bar is bar-rate evidence:

- low-band positive flux (bass re-entering lands on beat 1)
- inter-bar chroma coherence (a wrong phase makes each "bar" straddle a chord change)
- **arrangement-change alignment** (the strongest cue — the handful of moments
  where a track genuinely turns over land on downbeats essentially without
  exception)
- bass-note-change alignment

Two further traps:

**Never z-score a cue that cannot discriminate.** Four-on-the-floor kick energy is
identical on every beat, so its standard deviation is near zero, and dividing by
it turns rounding noise into a confident ±2σ vote. Normalise by the *level*
instead, so a flat cue contributes ~0.

**Never smooth-then-peak-pick.** Convolving novelty with a boxcar turns a step
into a ramp and moves its peak by half the kernel width. With a half-bar kernel
on 4/4 material that is exactly one beat.

> **Observed:** arrangement-change events landed at bar 30.25, 110.25, 117.25 — a
> systematic +0.25 bar bias, which *is* the downbeat error. Replacing it with a
> lag-free Foote novelty (mean of the W beats before vs. the W beats after) fixed
> the phase and raised confidence from 0.24 to 0.95.

## Phrases: a rank test, not a threshold

Phrase levels nest — every 16-bar boundary is also an 8-bar boundary — so plain
boundary contrast is inflated at every finer level and the finest grid always
wins. The question must be: are the bars that *only* level L marks actually novel?

And the answer must not depend on a tuned constant. Novelty has no meaningful
absolute scale and its distribution is badly skewed, so any ratio threshold needs
retuning per track.

> **Observed:** the same audio selected phrase length 4, 8 or 16 depending only on
> whether the contrast ratio used mean or median in its denominator. That is the
> signature of a criterion measuring the wrong thing.

A one-sided Mann-Whitney U test is invariant to monotone rescaling and comes with
a significance test, which is what lets the estimator say "do not subdivide"
honestly: descend the hierarchy while the finer level's unique boundaries beat
ordinary bars at AUC ≥ 0.65, p < 0.05.

> On the fixture: level 4 → AUC 0.545, p = 0.26 (not marked, correctly rejected);
> level 8 → AUC 0.721, p = 0.0098 (marked). Answer: 8 bars.

## Sections: snap boundaries to the phrase grid

Self-similarity over delay-embedded beat-synchronous features → Foote novelty →
peak picking. Delay embedding matters: without it the matrix is dominated by which
drum hit is sounding rather than by arrangement.

Then **snap every boundary to the phrase grid, with no distance limit**. A section
in club music always begins on a phrase boundary, so a boundary detected 3 bars
off the grid is measurement error, not a discovery. (Cue snapping keeps a 2-bar
guard, because a vocal really can enter mid-phrase.)

**Merging** uses a scale-free threshold — `d < 0.5 × median(d)` over adjacent
distances — which has the right degenerate behaviour: if nothing is
over-segmented, nothing is below half the median and nothing merges.

> **Observed:** a fixed threshold of 0.55 merged nothing, because within-section
> distances were ~1–4 and true boundaries ~10–24 on this track. Order-of-magnitude
> wrong, and it would be wrong differently on every other track.

Chroma is deliberately *excluded* from the merge comparison: two halves of the
same drop often sit on different chords, but a section is defined by its
arrangement, not its harmony.

## Labels: rules for evidence, HMM for grammar

Interpretable rules produce an emission score per (section, label) from statistics
a DJ would recognise, all expressed as percentile ranks within the track so the
same thresholds work on a quiet mix and a brickwalled one. A Viterbi pass over an
arrangement grammar then picks the best globally-consistent sequence.

The transition table describes *what follows what*. On its own it also implicitly
encodes a duration model, and a wrong one:

> **Observed:** a section split into three segments paid the self-transition cost
> twice, and Viterbi relabelled the middle segment "breakdown" purely to avoid it
> — overriding an emission score of 1.00 for "drop". Boosting the diagonal and
> weighting each segment's emission by its duration separates the two concerns.

A terminal-state prior (tracks end on an outro) is the symmetric counterpart of
the initial prior, and it makes the result stable across a wide range of the
self-transition constant — 0 through 6 all give 7/7 on the fixture, which is the
sign a constant is not load-bearing.

## Confidence: calibrate automatically

Hand-set weights can only honestly express *relative* importance. Their absolute
scale is arbitrary, and left alone it saturates the sigmoid.

> **Observed:** a textbook drop scored a logit of +7.9 → "100% confident". False,
> and useless for ranking, since every good cue reported the same thing.

Normalising total absolute weight to a fixed span and deriving the bias from it
(`bias = −span/2`) spreads the output without anyone hand-tuning a bias per cue
type. A model refitted from real feedback sets `"calibrated": true` and opts out.

## Loops: seamlessness

The metric that makes loop detection more than bar-counting: cosine similarity
between the loop window's features and the equal-length window *after* it. That
literally measures "if this repeats, will it sound like the track continuing".
Combined with internal stability, phrase alignment, and a hard penalty for cutting
a vocal phrase mid-word.


## Tempo changes: decisiveness, not loudness

The main grid fit assumes a constant tempo. That assumption has to be *checked*,
not asserted, so tempo is estimated independently in overlapping 20 s windows and
sustained shifts are reported. Two guards keep it honest: a single deviant window
is ignored, and the comparison is between medians either side of a candidate
point rather than between adjacent windows.

Neither guard is enough on its own, because a drumless passage still produces a
confident-looking argmax — the comb filter will happily lock onto pad swells.

> **Observed:** a rigidly constant 128 BPM track with a 40 s quiet passage
> spliced into it reported **two tempo changes**, both inside the quiet part.

The fix is a reliability test on each window, and the useful signal turned out
not to be the obvious one. Onset *level* barely separates the cases — 0.027 in
beat-bearing windows against 0.019 in noise, far too close to threshold on.
Comb *decisiveness* — how much better the winning tempo explains the window than
a typical wrong one — separates them cleanly: **2.9 and above with beats, 1.8 and
below without.** A strength near 1 means no tempo fits better than any other,
which is the definition of "there is no beat here". Windows below 2.3 are
excluded from changepoint detection entirely, rather than down-weighted: a tempo
estimate from a window with no beats is not weak evidence, it is no evidence.

## Vocal-dependent labels are gated, not guessed

Verse, chorus and bridge are distinctions from vocal-led songwriting. On an
instrumental they are not "hard to detect" — they are not present, and assigning
them is inventing structure. They are therefore removed from consideration
entirely unless the section carries real vocal energy, and the track's summary
says so ("no vocal content found where chorus would otherwise have been
labelled") rather than quietly producing a label nobody should trust.

Where the top two labels land within a small margin of each other — nearly always
drop vs chorus, which in a vocal house track are the same moment under two names
— the section is flagged ambiguous with the runner-up named, because that is a
choice only the DJ can make.

## Offsets are applied exactly once

A recurring class of bug, worth stating as a rule: a detector emits an *anchor*,
and the engine applies the configured offset. Never both.

> **Observed:** pre-drop cues landed 8 bars before the drop when the user had
> asked for 4, because the detector helpfully applied the offset and then the
> engine applied it again.

The same rule governs mix cues, whose offsets come from the mix panel rather than
the generic per-type offset. And clamping has to happen *after* snapping, not
before — snapping to the nearest phrase boundary will otherwise walk a cue
straight back across the limit it was just moved inside.
