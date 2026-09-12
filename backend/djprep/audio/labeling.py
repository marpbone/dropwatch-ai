"""Turning anonymous segments into DJ-meaningful labels.

Unsupervised segmentation gives you "section A, section B, section C". A DJ
needs "intro, buildup, drop". Bridging that gap is the part people reach for a
neural network to solve; this module argues for a different split of labour:

  * Interpretable rules produce a *score per (section, label)* from statistics a
    DJ would recognise -- energy percentile, bass presence, onset density,
    energy slope, vocal share, position in track.
  * A hidden Markov model over those scores enforces *structural grammar*: a
    drop follows a buildup, an outro only appears at the end, you do not get two
    intros. Viterbi decoding then picks the best globally-consistent labelling
    rather than the best label for each section in isolation.

Why this rather than a classifier:

  1. It is explainable by construction. Every label comes with the evidence that
     produced it, which is exactly what the UI needs to show and what makes a
     wrong answer debuggable instead of mysterious.
  2. There is no labelled dataset of "drop / breakdown / buildup" annotations at
     a useful scale, and building one is a bigger project than this one.
  3. The rules encode genuine domain knowledge that a small model would have to
     rediscover from data you do not have.

The HMM is doing real work, not decoration: it is what stops an isolated
high-energy segment in the middle of a breakdown from being called a drop, and
it is a ~40-line Viterbi rather than a dependency.
"""
from __future__ import annotations

import numpy as np

from djprep.models.analysis import LabelSummary, Section, SectionLabel

L = SectionLabel
LABELS: tuple[SectionLabel, ...] = (
    L.INTRO, L.VERSE, L.BUILDUP, L.DROP, L.CHORUS, L.BREAKDOWN, L.BRIDGE, L.OUTRO,
)

# Structural grammar. Values are unnormalised plausibilities for label i being
# followed by label j; rows are normalised at decode time. These encode how dance
# and pop arrangements are actually put together -- e.g. a buildup resolves into
# a drop or chorus and almost never into another buildup.
_T: dict[SectionLabel, dict[SectionLabel, float]] = {
    L.INTRO:     {L.INTRO: 0.6, L.VERSE: 1.0, L.BUILDUP: 2.5, L.DROP: 1.0,
                  L.CHORUS: 0.6, L.BREAKDOWN: 0.8, L.BRIDGE: 0.4, L.OUTRO: 0.05},
    L.VERSE:     {L.INTRO: 0.02, L.VERSE: 0.6, L.BUILDUP: 2.5, L.DROP: 0.8,
                  L.CHORUS: 2.0, L.BREAKDOWN: 1.0, L.BRIDGE: 0.8, L.OUTRO: 0.4},
    L.BUILDUP:   {L.INTRO: 0.02, L.VERSE: 0.2, L.BUILDUP: 0.5, L.DROP: 4.0,
                  L.CHORUS: 2.0, L.BREAKDOWN: 0.15, L.BRIDGE: 0.2, L.OUTRO: 0.1},
    L.DROP:      {L.INTRO: 0.02, L.VERSE: 0.8, L.BUILDUP: 0.5, L.DROP: 1.2,
                  L.CHORUS: 0.8, L.BREAKDOWN: 2.5, L.BRIDGE: 0.6, L.OUTRO: 1.2},
    L.CHORUS:    {L.INTRO: 0.02, L.VERSE: 1.5, L.BUILDUP: 0.8, L.DROP: 1.0,
                  L.CHORUS: 1.0, L.BREAKDOWN: 1.5, L.BRIDGE: 1.0, L.OUTRO: 1.2},
    L.BREAKDOWN: {L.INTRO: 0.02, L.VERSE: 1.0, L.BUILDUP: 3.5, L.DROP: 1.2,
                  L.CHORUS: 0.8, L.BREAKDOWN: 0.8, L.BRIDGE: 0.6, L.OUTRO: 0.8},
    L.BRIDGE:    {L.INTRO: 0.02, L.VERSE: 1.0, L.BUILDUP: 2.0, L.DROP: 1.0,
                  L.CHORUS: 1.5, L.BREAKDOWN: 1.0, L.BRIDGE: 0.4, L.OUTRO: 0.8},
    L.OUTRO:     {L.INTRO: 0.01, L.VERSE: 0.05, L.BUILDUP: 0.05, L.DROP: 0.05,
                  L.CHORUS: 0.05, L.BREAKDOWN: 0.1, L.BRIDGE: 0.05, L.OUTRO: 2.0},
}


# Labels that are only meaningful when there is a voice to hear. Verse, chorus
# and bridge are distinctions from vocal-led pop songwriting; on an instrumental
# they are not "hard to detect", they are *not present*, and assigning them is
# inventing structure. Gating them on actual vocal evidence removes a whole class
# of confident-but-meaningless output.
VOCAL_DEPENDENT: frozenset[SectionLabel] = frozenset({L.VERSE, L.CHORUS, L.BRIDGE})
MIN_VOCAL_RATIO = 0.22

# Two labels within this margin of each other are not a decision the analyser
# should pretend to have made. Drop vs chorus is the usual case: in a vocal house
# track they are the same moment under two names, and only the DJ knows which
# word they use.
AMBIGUITY_MARGIN = 0.12


def _bell(x: float, mu: float, sigma: float) -> float:
    return float(np.exp(-0.5 * ((x - mu) / max(sigma, 1e-6)) ** 2))


def _hi(x: float, k: float = 8.0, mid: float = 0.6) -> float:
    return float(1.0 / (1.0 + np.exp(-k * (x - mid))))


def _lo(x: float, k: float = 8.0, mid: float = 0.4) -> float:
    return float(1.0 / (1.0 + np.exp(k * (x - mid))))


def emission_scores(sections: list[Section], duration: float,
                    vocals_available: bool = True
                    ) -> tuple[np.ndarray, list[dict[str, float]], set[str]]:
    """Score every (section, label) pair from interpretable statistics.

    Returns (scores (n_sections, n_labels) in [0, 1], per-section detail dicts).
    Each rule below is a claim about how the music works, stated so it can be
    argued with:
    """
    n = len(sections)
    S = np.zeros((n, len(LABELS)))
    detail: list[dict[str, float]] = []
    suppressed: set[str] = set()

    for i, s in enumerate(sections):
        pos = s.start_sec / max(duration, 1e-6)
        nxt = sections[i + 1] if i + 1 < n else None
        prv = sections[i - 1] if i > 0 else None
        is_first, is_last = i == 0, i == n - 1

        # A buildup is defined by what comes *after* it as much as by itself.
        next_jump = (nxt.energy_pct - s.energy_pct) if nxt else 0.0
        prev_drop = (prv.energy_pct - s.energy_pct) if prv else 0.0

        scores = {
            # Early, thin, not yet at full energy. Drums may be present but the
            # bass usually is not.
            L.INTRO: (
                _bell(pos, 0.0, 0.14) * 0.6
                + _lo(s.energy_pct, mid=0.45) * 0.3
                + (0.35 if is_first else 0.0)
            ),
            # Mid energy with vocals present but not peaking.
            L.VERSE: (
                _bell(s.energy_pct, 0.45, 0.22) * 0.4
                + _hi(s.vocal_ratio, k=10, mid=0.35) * 0.5
                + _bell(pos, 0.4, 0.35) * 0.2
            ),
            # Rising energy, rising brightness, thinning low end, and -- the
            # decisive part -- resolving into something louder.
            L.BUILDUP: (
                _hi(s.energy_slope, k=3.0, mid=0.25) * 0.5
                + _hi(s.high_energy_pct, mid=0.5) * 0.25
                + _hi(next_jump, k=6.0, mid=0.15) * 0.6
                + _hi(s.onset_density_pct, mid=0.45) * 0.2
            ),
            # Loud, bass-heavy, dense, and not at the very start of the track.
            L.DROP: (
                _hi(s.energy_pct, mid=0.6) * 0.5
                + _hi(s.low_energy_pct, mid=0.6) * 0.5
                + _hi(s.onset_density_pct, mid=0.5) * 0.3
                + _hi(prev_drop, k=6.0, mid=0.15) * 0.3
                + (-0.5 if pos < 0.08 else 0.0)
            ),
            # Loud like a drop, but carried by voice rather than by low end.
            L.CHORUS: (
                _hi(s.energy_pct, mid=0.55) * 0.4
                + _hi(s.vocal_ratio, k=10, mid=0.45) * 0.7
                + _bell(pos, 0.5, 0.4) * 0.1
            ),
            # The bass leaves. This is the defining feature -- overall loudness
            # can stay moderate while the low end drops out entirely.
            L.BREAKDOWN: (
                _lo(s.low_energy_pct, mid=0.45) * 0.7
                + _lo(s.onset_density_pct, mid=0.4) * 0.35
                + _bell(pos, 0.5, 0.3) * 0.25
                + (-0.6 if is_first or is_last else 0.0)
            ),
            L.BRIDGE: (
                _bell(s.energy_pct, 0.4, 0.25) * 0.3
                + _bell(pos, 0.62, 0.18) * 0.3
                + 0.12
            ),
            # Late and decaying.
            L.OUTRO: (
                _hi(pos, k=14, mid=0.83) * 0.7
                + _lo(s.energy_pct, mid=0.5) * 0.3
                + (0.4 if is_last else -0.35)
                + _lo(s.energy_slope, k=2.0, mid=-0.1) * 0.2
            ),
        }
        # Gate the vocal-dependent labels. Without a voice in this section there
        # is no verse/chorus/bridge to find, so they are removed from
        # consideration rather than allowed to win on position alone.
        if not vocals_available or s.vocal_ratio < MIN_VOCAL_RATIO:
            for lab in VOCAL_DEPENDENT:
                if scores[lab] > 0.25:
                    suppressed.add(lab.value)
                scores[lab] = 0.0

        row = np.array([max(0.0, scores[lab]) for lab in LABELS])
        if row.max() <= 1e-9:
            row = np.full(len(LABELS), 1e-6)
        S[i] = row / (row.max() + 1e-9)
        detail.append({lab.value: round(float(scores[lab]), 4) for lab in LABELS})
    return S, detail, suppressed


# Self-transition bonus. The transition table above describes *what follows
# what*; on its own it also implicitly encodes a duration model, and a wrong one:
# a section that segmentation split into three pieces pays the self-transition
# cost twice, which is enough to make Viterbi relabel the middle piece just to
# avoid it. Boosting the diagonal separates the two concerns -- staying put is
# cheap, and the off-diagonal entries are left to mean only what they say.
SELF_TRANSITION_BONUS = 2.5


def viterbi(S: np.ndarray, durations: np.ndarray | None = None,
            floor: float = 1e-4) -> list[int]:
    """Decode the best globally-consistent label sequence.

    Standard Viterbi in log space. Emissions come from `emission_scores`, the
    transition matrix encodes arrangement grammar, and the initial distribution
    strongly favours starting on an intro. The point of doing this rather than
    taking the per-section argmax is that a label's plausibility depends on its
    neighbours: an energetic segment sandwiched between two breakdowns is much
    more likely a short drop than a chorus, and only a sequence model can use
    that.
    """
    n, k = S.shape
    if n == 0:
        return []
    T = np.zeros((k, k))
    for i, a in enumerate(LABELS):
        for j, b in enumerate(LABELS):
            T[i, j] = _T[a][b] + (SELF_TRANSITION_BONUS if a is b else 0.0)
    T = T / T.sum(axis=1, keepdims=True)
    logT = np.log(np.maximum(T, floor))
    logE = np.log(np.maximum(S, floor))
    # Weight each segment's evidence by how much of the track it accounts for.
    # Segment-level observations are not equal-sized samples: a 32-bar drop is a
    # far stronger observation than an 8-bar fragment, and without this a run of
    # short fragments can outvote the section they belong to.
    if durations is not None and durations.size == n:
        w = durations / max(float(np.median(durations)), 1e-9)
        logE = logE * np.clip(w, 0.4, 3.0)[:, None]

    init = np.array([3.0 if lab is L.INTRO else (0.02 if lab is L.OUTRO else 1.0)
                     for lab in LABELS])
    init = np.log(init / init.sum())
    # Terminal prior, the symmetric counterpart of the initial one: tracks end on
    # an outro far more often than on anything else. Without it the self-
    # transition bonus can carry a final drop state straight through a fading
    # outro, since nothing else in the model knows the track is about to stop.
    final = np.array([0.05 if lab is L.INTRO else (4.0 if lab is L.OUTRO else 1.0)
                      for lab in LABELS])
    final = np.log(final / final.sum())

    dp = init + logE[0]
    back = np.zeros((n, k), dtype=int)
    for t in range(1, n):
        trans = dp[:, None] + logT
        back[t] = np.argmax(trans, axis=0)
        dp = trans.max(axis=0) + logE[t]

    dp = dp + final
    path = [int(np.argmax(dp))]
    for t in range(n - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    return path[::-1]


def label_sections(sections: list[Section], duration: float,
                   vocals_available: bool = True) -> set[str]:
    """Label sections in place. Returns the set of labels that were suppressed.

    Sections already marked `is_manual` keep the label the user gave them: a
    re-analysis may refine everything else, but it must never overwrite a
    decision a human made.
    """
    if not sections:
        return set()
    S, detail, suppressed = emission_scores(sections, duration, vocals_available)
    durations = np.array([s.end_sec - s.start_sec for s in sections])
    path = viterbi(S, durations=durations)

    for i, (sec, idx) in enumerate(zip(sections, path, strict=False)):
        sec.label_scores = detail[i]
        if sec.is_manual:
            continue
        sec.label = LABELS[idx]

        # Confidence from how far the chosen label stands above the runner-up.
        order = np.argsort(S[i])[::-1]
        row = S[i][order]
        margin = float(row[0] - row[1]) if row.size > 1 else 0.5
        sec.confidence = round(float(np.clip(0.35 + 0.65 * margin, 0.05, 0.98)), 3)

        # Flag a genuinely close call instead of hiding it behind the number.
        if row.size > 1 and margin < AMBIGUITY_MARGIN:
            runner_up = LABELS[int(order[1])]
            if runner_up is not sec.label:
                sec.ambiguous_with = runner_up.value
                sec.ambiguity_margin = round(margin, 4)

    number_occurrences(sections)
    return suppressed


def number_occurrences(sections: list[Section]) -> None:
    """Number repeats per label in time order: the second drop is occurrence 2."""
    seen: dict[SectionLabel, int] = {}
    for s in sorted(sections, key=lambda x: x.start_sec):
        seen[s.label] = seen.get(s.label, 0) + 1
        s.occurrence = seen[s.label]


def summarise_labels(sections: list[Section], suppressed: set[str],
                     vocals_available: bool) -> LabelSummary:
    """Report which section types this track actually contains.

    Not every song has every section, and a UI that offers a Chorus checkbox on
    an instrumental is offering something that can never fire. Presence is
    reported explicitly so the frontend can grey those controls out and say why.
    """
    counts: dict[str, int] = {}
    for s in sections:
        counts[s.label.value] = counts.get(s.label.value, 0) + 1

    present = sorted(counts)
    absent = [lab.value for lab in LABELS if lab.value not in counts]
    notes: list[str] = []

    if not vocals_available:
        notes.append("Vocal detection was off, so verse, chorus and bridge were "
                     "not considered.")
    elif suppressed:
        notes.append(f"No vocal content found where {', '.join(sorted(suppressed))} "
                     f"would otherwise have been labelled — treated as instrumental.")

    notes.extend(
        f"{s.label.value.title()} at bar {s.start_bar} is a close call with "
        f"{s.ambiguous_with} — worth checking by ear."
        for s in sections if s.ambiguous_with)

    repeats = {k: v for k, v in counts.items() if v > 1}
    if repeats:
        notes.append("Repeated sections: "
                     + ", ".join(f"{v}x {k}" for k, v in sorted(repeats.items())) + ".")

    return LabelSummary(present=present, absent=absent,
                        suppressed=sorted(suppressed), counts=counts, notes=notes)
