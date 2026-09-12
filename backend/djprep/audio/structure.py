"""Structural segmentation: where does one section end and the next begin.

Pipeline:
  1. Beat-synchronous feature matrix (timbre + harmony + band energy).
  2. Delay embedding, so each column describes a short *passage* rather than an
     instant. Without this the self-similarity matrix is dominated by which drum
     hit is sounding; with it, the matrix shows arrangement.
  3. Self-similarity matrix and Foote checkerboard novelty.
  4. Peak-pick the novelty into boundary candidates.
  5. **Snap every boundary to the phrase grid.** This is the step that makes the
     output usable by a DJ rather than merely correct-ish. Sections in club music
     start on phrase boundaries; a boundary detected 1.3 bars early is not
     evidence against that, it is measurement error, and rounding it onto the
     grid is strictly better than trusting the raw peak.
  6. Cluster segments so repeated material (two drops, three choruses) is
     recognisably the same thing.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from djprep.audio.features import Features, pct_rank
from djprep.models.analysis import BeatGrid, PhraseGrid, Section, SectionLabel


def beat_sync_features(f: Features, grid: BeatGrid) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(grid.beat_times(), dtype=float)
    duration = float(f.times[-1]) if f.n_frames else 0.0
    times = times[times < duration]
    if times.size < 4:
        return np.zeros((0, 0)), times
    edges = np.clip((times * f.sr / f.hop).astype(int), 0, f.n_frames)
    cols = []
    for i in range(len(edges)):
        a = edges[i]
        b = edges[i + 1] if i + 1 < len(edges) else min(a + 8, f.n_frames)
        b = max(b, a + 1)
        cols.append(np.concatenate([
            f.mfcc[1:14, a:b].mean(axis=1),
            f.chroma[:, a:b].mean(axis=1) * 8.0,
            np.array([f.band_energy[k][a:b].mean() for k in
                      ("sub", "bass", "lowmid", "mid", "high")]) / 10.0,
        ]))
    X = np.stack(cols, axis=1)
    mu, sd = X.mean(axis=1, keepdims=True), X.std(axis=1, keepdims=True)
    return (X - mu) / np.where(sd < 1e-9, 1.0, sd), times


def _delay_embed(X: np.ndarray, k: int = 4, stride: int = 1) -> np.ndarray:
    """Stack k lagged copies so each column describes a passage, not an instant."""
    if X.size == 0:
        return X
    n = X.shape[1]
    parts = [X[:, max(0, i * stride):n - (k - 1 - i) * stride] for i in range(k)]
    m = min(p.shape[1] for p in parts)
    return np.vstack([p[:, :m] for p in parts])


def _checkerboard(size: int) -> np.ndarray:
    """Gaussian-tapered checkerboard kernel (Foote, 2000)."""
    g = np.outer(np.hanning(2 * size), np.hanning(2 * size))
    k = np.ones((2 * size, 2 * size))
    k[:size, size:] = -1
    k[size:, :size] = -1
    return k * g


def novelty_curve(X: np.ndarray, kernel_beats: int = 32) -> np.ndarray:
    """Foote novelty from the self-similarity matrix of `X`."""
    if X.shape[1] < 8:
        return np.zeros(max(X.shape[1], 0))
    Xn = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-9)
    S = Xn.T @ Xn
    n = S.shape[0]
    size = max(4, min(kernel_beats, n // 4))
    K = _checkerboard(size)
    nov = np.zeros(n)
    for i in range(size, n - size):
        nov[i] = float((S[i - size:i + size, i - size:i + size] * K).sum())
    nov = np.maximum(nov, 0.0)
    return nov / (nov.max() + 1e-9)


def detect_boundaries(f: Features, grid: BeatGrid, phrases: PhraseGrid,
                      min_section_bars: int = 8) -> list[float]:
    X, beat_times = beat_sync_features(f, grid)
    if X.shape[1] < 16:
        return [0.0, float(f.times[-1]) if f.n_frames else 0.0]

    E = _delay_embed(X, k=4, stride=grid.beats_per_bar)
    nov = novelty_curve(E, kernel_beats=grid.beats_per_bar * 8)
    nov = uniform_filter1d(nov, size=3, mode="nearest")

    min_dist = max(4, min_section_bars * grid.beats_per_bar // 2)
    peaks, _ = find_peaks(nov, distance=min_dist, prominence=0.06)
    cand = [float(beat_times[min(p, beat_times.size - 1)]) for p in peaks]

    # Snap onto the phrase grid and de-duplicate. Unlike cue snapping we allow
    # an unlimited shift here: a *section* in club music always begins on a
    # phrase boundary, so a boundary detected 3 bars off the grid is measurement
    # error, not a genuine mid-phrase section start. (Cue snapping keeps its
    # 2-bar guard because a vocal really can enter mid-phrase.)
    from djprep.audio.phrases import snap_to_phrase
    max_shift = max(2.0, phrases.phrase_bars / 2.0)
    snapped = sorted({round(snap_to_phrase(t, grid, phrases, max_shift_bars=max_shift), 3)
                      for t in cand})

    duration = float(f.times[-1]) if f.n_frames else 0.0
    bounds = [0.0] + [t for t in snapped if 0.5 < t < duration - 0.5] + [duration]

    # Enforce a minimum section length: merge anything shorter into its neighbour.
    min_len = min_section_bars * grid.bar_period * 0.75
    out = [bounds[0]]
    for t in bounds[1:]:
        if t - out[-1] >= min_len:
            out.append(t)
    if len(out) < 2:
        out = [0.0, duration]
    elif out[-1] < duration - 1e-6:
        out[-1] = duration
    return out


def merge_similar(f: Features, grid: BeatGrid, bounds: list[float],
                  merge_factor: float = 0.5, max_bars: int = 64) -> list[float]:
    """Merge adjacent segments that are not meaningfully different.

    Foote novelty happily splits a 32-bar drop in half when the lead changes,
    which is real but is not a *section* boundary.

    The threshold is expressed relative to the track's own distribution of
    adjacent-segment distances rather than as an absolute number, and that
    matters more than it looks. Feature distances here span roughly 1 to 25
    depending on the track's dynamic range and instrumentation, so any fixed
    threshold is wrong for most music -- an earlier fixed value of 0.55 merged
    nothing at all on the test fixture, where within-section distances were ~1-4
    and true boundaries ~10-24.

    Merging when `d < merge_factor * median(d)` is scale-free and has the right
    degenerate behaviour: if every adjacent pair is equally different (no
    over-segmentation), nothing is below half the median and nothing merges. It
    only fires when the distances are genuinely bimodal, which is exactly the
    situation it exists to fix.
    """
    if len(bounds) < 3:
        return bounds

    def vec(t0: float, t1: float) -> np.ndarray:
        a, b = f.frame_at(t0), max(f.frame_at(t1), f.frame_at(t0) + 1)
        # Note the absence of chroma. Two 8-bar halves of the same drop often
        # sit on different chords, and including harmony here would keep them
        # apart -- but a section is defined by its *arrangement*, not its chord.
        # Timbre and band energy are what actually distinguish a drop from a
        # breakdown, so those are what the merge decision uses.
        return np.concatenate([
            f.mfcc[1:14, a:b].mean(axis=1) / 8.0,
            np.array([f.band_energy[k][a:b].mean() for k in
                      ("sub", "bass", "lowmid", "mid", "high")]) / 12.0,
            np.array([f.onset_density[a:b].mean() / 4.0,
                      f.percussive_ratio[a:b].mean() * 2.0]),
        ])

    vecs = [vec(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    dists = np.array([float(np.linalg.norm(vecs[i] - vecs[i + 1]))
                      for i in range(len(vecs) - 1)])
    if dists.size == 0:
        return bounds
    threshold = merge_factor * float(np.median(dists))

    out = [bounds[0]]
    i = 0
    while i < len(bounds) - 1:
        t0, t1 = bounds[i], bounds[i + 1]
        j = i + 1
        while j < len(bounds) - 1:
            nxt = bounds[j + 1]
            if (nxt - t0) / grid.bar_period > max_bars:
                break
            if float(np.linalg.norm(vec(t0, t1) - vec(bounds[j], nxt))) > threshold:
                break
            t1 = nxt
            j += 1
        out.append(t1)
        i = j
    return out


def segment_stats(f: Features, grid: BeatGrid, bounds: list[float],
                  vocal_like: np.ndarray | None) -> list[Section]:
    """Interpretable per-section statistics, all normalised within the track.

    Everything here is a percentile rank against the track's own distribution
    rather than an absolute level, so the same thresholds work for a quiet,
    dynamic mix and a brickwalled one. This is what makes the labelling rules
    portable across genres and masters.
    """
    rms = f.rms_db
    low = f.band_energy["sub"] + f.band_energy["bass"]
    high = f.band_energy["high"]
    dens = f.onset_density
    cent = f.centroid

    out: list[Section] = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        a, b = f.frame_at(t0), max(f.frame_at(t1), f.frame_at(t0) + 1)
        seg_rms = rms[a:b]
        # Energy slope: dB per bar, the signature of a buildup.
        if seg_rms.size > 4:
            x = np.arange(seg_rms.size) * (f.hop / f.sr) / grid.bar_period
            slope = float(np.polyfit(x, seg_rms, 1)[0])
        else:
            slope = 0.0
        vr = float(vocal_like[a:b].mean()) if vocal_like is not None and vocal_like.size else 0.0
        out.append(Section(
            start_sec=round(t0, 3), end_sec=round(t1, 3),
            start_bar=grid.bar_index(t0),
            length_bars=round((t1 - t0) / grid.bar_period, 2),
            label=SectionLabel.UNKNOWN,
            energy_pct=round(float(pct_rank(rms, seg_rms.mean())), 4),
            low_energy_pct=round(float(pct_rank(low, low[a:b].mean())), 4),
            high_energy_pct=round(float(pct_rank(high, high[a:b].mean())), 4),
            onset_density_pct=round(float(pct_rank(dens, dens[a:b].mean())), 4),
            brightness_pct=round(float(pct_rank(cent, cent[a:b].mean())), 4),
            energy_slope=round(slope, 4),
            vocal_ratio=round(vr, 4),
        ))
    return out


def cluster_sections(f: Features, grid: BeatGrid, sections: list[Section],
                     max_clusters: int = 6) -> None:
    """Assign cluster ids in place so repeated material is recognisable.

    Agglomerative clustering on mean timbre+harmony per section. Two drops in the
    same track land in the same cluster, which lets the labeller apply "the
    highest-energy recurring cluster is the drop" instead of judging each section
    in isolation.
    """
    if len(sections) < 2:
        for s in sections:
            s.cluster_id = 0
        return
    feats = []
    for s in sections:
        a, b = f.frame_at(s.start_sec), max(f.frame_at(s.end_sec), f.frame_at(s.start_sec) + 1)
        feats.append(np.concatenate([
            f.mfcc[1:14, a:b].mean(axis=1),
            f.chroma[:, a:b].mean(axis=1) * 8.0,
        ]))
    F = np.stack(feats)
    F = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-9)

    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        Z = linkage(F, method="ward")
        k = min(max_clusters, max(2, len(sections) // 2))
        labels = fcluster(Z, t=k, criterion="maxclust")
    except Exception:  # pragma: no cover
        labels = np.arange(1, len(sections) + 1)
    for s, c in zip(sections, labels, strict=False):
        s.cluster_id = int(c)
