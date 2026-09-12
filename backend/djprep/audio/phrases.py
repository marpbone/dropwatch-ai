"""Phrase-level structure: how bars group into musical sentences.

Dance music is written in power-of-two bar groups -- almost always 8, sometimes
4, 16 or 32. Knowing the phrase length *and phase* is what separates a tool that
places a marker "every 32 bars from zero" from one that places it where the music
actually turns over. Nearly every cue a DJ would set by ear sits on a phrase
boundary, so this grid is the scaffold the recommendation stage hangs off.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu

from djprep.audio.features import Features
from djprep.models.analysis import BeatGrid, PhraseGrid

PLAUSIBLE_PHRASES = (4, 8, 16, 32)

# Evidence thresholds for subdividing the metrical hierarchy. These are the only
# tunable constants in this module and they are interpretable: a level is
# accepted when its unique boundaries beat ordinary bars at least 65% of the time
# and that margin is significant at p < 0.05.
MIN_AUC = 0.65
MAX_P = 0.05


def bar_features(f: Features, grid: BeatGrid) -> tuple[np.ndarray, np.ndarray]:
    """Bar-synchronous feature matrix. Returns (features (D, n_bars), bar_times)."""
    db_times = np.asarray(grid.downbeat_times(), dtype=float)
    duration = float(f.times[-1]) if f.n_frames else 0.0
    db_times = db_times[db_times < duration]
    if db_times.size < 2:
        return np.zeros((0, 0)), db_times

    blocks = []
    for i, t0 in enumerate(db_times):
        t1 = db_times[i + 1] if i + 1 < db_times.size else min(t0 + grid.bar_period, duration)
        a, b = f.frame_at(t0), max(f.frame_at(t1), f.frame_at(t0) + 1)
        blocks.append(np.concatenate([
            f.mfcc[1:14, a:b].mean(axis=1),                 # timbre (MFCC0 = loudness, dropped)
            f.chroma[:, a:b].mean(axis=1) * 10.0,           # harmony
            np.array([f.band_energy[k][a:b].mean() for k in
                      ("sub", "bass", "lowmid", "mid", "high")]),
            np.array([
                f.rms_db[a:b].mean(),
                f.onset_density[a:b].mean(),
                f.percussive_ratio[a:b].mean(),
                f.centroid[a:b].mean() / 1000.0,
            ]),
        ]))

    X = np.stack(blocks, axis=1).astype(float)
    mu, sd = X.mean(axis=1, keepdims=True), X.std(axis=1, keepdims=True)
    return (X - mu) / np.where(sd < 1e-9, 1.0, sd), db_times


def bar_novelty(X: np.ndarray) -> np.ndarray:
    """Distance between consecutive bars, normalised to [0, 1].

    Deliberately *not* smoothed. Convolving novelty with a boxcar turns a step
    into a ramp and shifts its peak by half the kernel width -- on 4/4 material
    with a half-bar kernel that is exactly one beat, which is enough to put every
    downstream boundary on the wrong beat. Consecutive differencing has no group
    delay by construction.
    """
    if X.shape[1] < 2:
        return np.zeros(max(X.shape[1], 0))
    d = np.linalg.norm(np.diff(X, axis=1), axis=0)
    nov = np.concatenate([[0.0], d])
    return nov / (nov.max() + 1e-9)


def estimate_phrase_grid(f: Features, grid: BeatGrid,
                         phrase_hint: int | None = None) -> PhraseGrid:
    X, bar_times = bar_features(f, grid)
    n_bars = X.shape[1]
    if n_bars < 8:
        return PhraseGrid(phrase_bars=phrase_hint or 8, phase_bars=0,
                          boundaries_sec=list(bar_times), confidence=0.1)
    nov = bar_novelty(X)

    def boundary_auc(length: int, phase: int, odd_only: bool = False) -> tuple[float, float]:
        """Rank-based evidence that this level's boundaries are genuinely marked.

        Returns (AUC, p) from a one-sided Mann-Whitney U test comparing novelty at
        boundary bars against novelty at non-boundary bars. AUC is literally
        P(a boundary bar is more novel than a non-boundary bar): 0.5 = no
        evidence, 1.0 = perfect separation.

        Why a rank test rather than a ratio of means. Novelty has no meaningful
        absolute scale and its distribution is badly skewed -- a drop landing is
        an order of magnitude more novel than a chord change -- so any
        ratio-threshold criterion has to be retuned per track. Two earlier
        versions of this function used mean-ratio and then median-ratio
        thresholds; they selected *different* phrase lengths for the same audio
        purely as a function of which normalisation was chosen, which is the
        signature of a criterion measuring the wrong thing. A rank statistic is
        invariant to monotone rescaling and comes with a significance test, which
        is what lets the estimator say "do not subdivide" honestly rather than
        always bottoming out at the finest grid.

        `odd_only` restricts the boundary set to bars unique to this level.
        Phrase levels nest -- every 16-bar boundary is also an 8-bar boundary --
        so without it each finer level inherits the coarser level's evidence.
        """
        idx = np.arange(phase, n_bars, length)
        num_idx = idx
        if odd_only:
            num_idx = np.setdiff1d(idx, np.arange(phase, n_bars, length * 2))
        mask = np.ones(n_bars, dtype=bool)
        mask[idx] = False
        non_idx = np.flatnonzero(mask)
        if num_idx.size < 3 or non_idx.size < 3:
            return 0.5, 1.0
        u = mannwhitneyu(nov[num_idx], nov[non_idx], alternative="greater")
        return float(u.statistic) / (num_idx.size * non_idx.size), float(u.pvalue)

    def best_phase(length: int) -> tuple[int, float]:
        auc, ph = max((boundary_auc(length, p)[0], p) for p in range(length))
        return ph, auc

    if phrase_hint in PLAUSIBLE_PHRASES:
        length = int(phrase_hint)
        phase, auc = best_phase(length)
        alts = {length: round(auc, 4)}
    else:
        cands = [L for L in PLAUSIBLE_PHRASES if n_bars > L * 2]
        if not cands:
            return PhraseGrid(phrase_bars=8, phase_bars=0,
                              boundaries_sec=list(bar_times), confidence=0.1)
        phases = {L: best_phase(L) for L in cands}
        alts = {L: round(phases[L][1], 4) for L in cands}

        # Descend the metrical hierarchy: begin at the coarsest level and
        # subdivide only while the finer level's unique boundaries are
        # significantly more novel than ordinary bars. This mirrors how metre is
        # heard -- the largest period first, subdividing only when the
        # subdivision is audible -- and it degrades gracefully: an unstructured
        # track stops descending early instead of being forced onto a fine grid
        # it does not actually have.
        length = max(cands)
        while length // 2 in phases:
            finer = length // 2
            auc, pval = boundary_auc(finer, phases[finer][0], odd_only=True)
            if auc < MIN_AUC or pval > MAX_P:
                break
            length = finer
        phase, auc = phases[length]

    conf = float(np.clip((auc - 0.5) * 2.0, 0.05, 0.95))
    boundaries = [float(bar_times[i]) for i in range(phase, n_bars, length)]
    return PhraseGrid(phrase_bars=length, phase_bars=int(phase),
                      boundaries_sec=boundaries, confidence=round(conf, 3),
                      alt_lengths={str(k): v for k, v in alts.items()})


def snap_to_phrase(t: float, grid: BeatGrid, phrases: PhraseGrid,
                   max_shift_bars: float = 2.0) -> float:
    """Snap to the nearest phrase boundary, but only if it is close.

    Guard rail: if the nearest phrase boundary is more than `max_shift_bars`
    away, the detection probably found something real that simply is not on a
    phrase line -- a mid-phrase vocal entry, say. Dragging it two bars would be
    worse than leaving it on its bar, so we fall back to downbeat snapping.
    """
    if not phrases.boundaries_sec:
        return grid.snap_to_downbeat(t)
    b = np.asarray(phrases.boundaries_sec)
    i = int(np.argmin(np.abs(b - t)))
    if abs(b[i] - t) <= max_shift_bars * grid.bar_period:
        return float(b[i])
    return grid.snap_to_downbeat(t)
