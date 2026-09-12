"""Loop candidate generation and scoring.

A good DJ loop is not just "8 bars starting somewhere sensible". It has to
*survive being repeated*, which is a measurable property:

  seamlessness      -- how similar the loop's content is to the material that
                       immediately follows it. If bar N+1 sounds like bar 1, the
                       loop is transparent; if the track has moved on, looping
                       creates an audible lurch every cycle. Measured as the
                       cosine similarity between the loop window's features and
                       the equal-length window after it.
  internal stability-- how much the content varies *inside* the loop. A loop
                       containing a transition will always sound wrong.
  phrase alignment  -- loops that start on phrase boundaries preserve the
                       listener's sense of where "one" is.
  vocal integrity   -- a loop whose end cuts a vocal phrase mid-word is unusable
                       regardless of how good its other numbers are.

That first metric is the one that makes this more than a bar-counter, and it is
cheap: we already have beat-synchronous features from the analysis stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from djprep.audio.features import Features
from djprep.models.analysis import SectionLabel, TrackAnalysis
from djprep.models.config import LoopConfig
from djprep.models.prep import LoopKind

# Which section label each loop location maps to. Keeping this table explicit
# rather than inferring it keeps the UI's vocabulary and the analyser's
# vocabulary independent -- they are allowed to diverge.
LOCATION_SECTIONS: dict[LoopKind, tuple[SectionLabel, ...]] = {
    LoopKind.INTRO: (SectionLabel.INTRO,),
    LoopKind.OUTRO: (SectionLabel.OUTRO,),
    LoopKind.BREAKDOWN: (SectionLabel.BREAKDOWN,),
    LoopKind.DROP: (SectionLabel.DROP, SectionLabel.CHORUS),
    LoopKind.PRE_DROP: (SectionLabel.BUILDUP,),
    LoopKind.VOCAL: (SectionLabel.CHORUS, SectionLabel.VERSE),
}

CANDIDATE_LENGTHS = (4.0, 8.0, 16.0)


@dataclass
class LoopCandidate:
    start_sec: float
    end_sec: float
    length_bars: float
    kind: LoopKind
    evidence: dict[str, float] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)


def _window(F: np.ndarray, times: np.ndarray, t0: float, t1: float) -> np.ndarray | None:
    a = int(np.searchsorted(times, t0))
    b = int(np.searchsorted(times, t1))
    if b - a < 2 or b > F.shape[1]:
        return None
    return F[:, a:b]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[1], b.shape[1])
    if n < 2:
        return 0.0
    x, y = a[:, :n].ravel(), b[:, :n].ravel()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 1e-9 else 0.0


def evaluate_loop(f: Features, an: TrackAnalysis, F: np.ndarray, times: np.ndarray,
                  start: float, length_bars: float, kind: LoopKind) -> LoopCandidate | None:
    g = an.beat_grid
    dur = length_bars * g.bar_period
    end = start + dur
    if end > an.meta.duration_sec - 0.05 or start < 0:
        return None

    loop_w = _window(F, times, start, end)
    next_w = _window(F, times, end, end + dur)
    if loop_w is None:
        return None

    seam = _cos(loop_w, next_w) if next_w is not None else 0.5
    # Internal stability: variance across the loop's own frames, inverted.
    var = float(np.mean(np.var(loop_w, axis=1)))
    stability = float(np.exp(-var / 1.5))

    on_phrase = bool(an.phrase_grid.boundaries_sec) and min(
        abs(np.asarray(an.phrase_grid.boundaries_sec) - start)) < 0.05

    # Does the loop boundary land inside a vocal phrase?
    cuts_vocal = 0.0
    for v in an.vocals:
        if v.start_sec + 0.25 < end < v.end_sec - 0.25:
            cuts_vocal = 1.0
        if v.start_sec + 0.25 < start < v.end_sec - 0.25:
            cuts_vocal = 1.0

    a_f, b_f = f.frame_at(start), max(f.frame_at(end), f.frame_at(start) + 1)
    dens = float(f.onset_density[a_f:b_f].mean())
    has_drums = float(np.clip(dens / 4.0, 0, 1))

    ev = {
        "seamlessness": float(np.clip((seam + 1.0) / 2.0, 0, 1)),
        "internal_stability": stability,
        "loop_on_phrase": 1.0 if on_phrase else 0.0,
        "has_drums": has_drums,
        "vocal_cut": cuts_vocal,
        "grid_confidence": g.confidence,
    }
    return LoopCandidate(round(start, 4), round(end, 4), length_bars, kind, ev)


def generate(f: Features, an: TrackAnalysis, cfg: LoopConfig) -> list[LoopCandidate]:
    """Propose loops at every enabled location, at the configured length(s)."""
    from djprep.audio.structure import beat_sync_features

    if not cfg.enabled:
        return []
    F, times = beat_sync_features(f, an.beat_grid)
    if F.size == 0:
        return []

    wanted = cfg.enabled_kinds()
    fixed = cfg.length.bars()
    lengths = [fixed] if fixed else list(CANDIDATE_LENGTHS)

    out: list[LoopCandidate] = []
    for kind in wanted:
        labels = LOCATION_SECTIONS.get(kind, ())
        for s in an.sections:
            if s.label not in labels:
                continue
            for L in lengths:
                if s.length_bars < L:
                    continue
                # Try the section start, and -- for pre-drop loops -- the last L
                # bars of the build-up, which is the part a DJ actually wants to
                # hold before releasing the drop.
                starts = [s.start_sec]
                if kind is LoopKind.PRE_DROP:
                    starts = [s.end_sec - L * an.beat_grid.bar_period]
                elif kind is LoopKind.OUTRO and s.length_bars >= L * 2:
                    starts.append(s.start_sec + L * an.beat_grid.bar_period)
                for st in starts:
                    st = an.beat_grid.snap_to_downbeat(st)
                    c = evaluate_loop(f, an, F, times, st, L, kind)
                    if c is None:
                        continue
                    c.notes["location"] = f"{L:.0f}-bar loop in the {s.label.value}"
                    out.append(c)
    return out
