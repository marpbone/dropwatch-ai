"""Turning evidence into a calibrated confidence, with the reasons attached.

The model is deliberately a **logistic regression over interpretable features**:

    logit = bias + sum_i w_i * x_i
    confidence = sigmoid(logit)

Three reasons this is the right choice here rather than something bigger:

  1. *It is the explanation.* Each term's contribution w_i * x_i is literally that
     reason's weight in the decision, so "Reasons: strong energy increase (+1.4),
     on a downbeat (+0.8), 16-bar build-up before it (+0.6)" is read directly off
     the model rather than reconstructed post-hoc. Nothing about the displayed
     explanation can drift away from what the model actually did.

  2. *It is refittable from the feedback this app naturally produces.* Every
     accept, reject and drag in the editor is a labelled example. A few hundred
     of them are enough to fit a dozen weights properly -- and nowhere near
     enough to train anything larger. Matching model capacity to the data you can
     realistically collect is the whole game.

  3. *It degrades honestly.* Hand-set priors are a working model on day one, and
     the same code path serves refitted weights on day one hundred.

Weights live in `djprep/data/cue_weights.json`, not in code, so refitting is a
data update rather than a deploy.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from djprep.models.common import Reason
from djprep.models.prep import CueKind

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "cue_weights.json"

# Human-readable phrasing for each evidence feature. Sign-aware: a feature can
# read as supporting or undermining depending on which way it points.
PHRASING: dict[str, tuple[str, str]] = {
    "on_downbeat":            ("Lands on a downbeat", "Not on a downbeat"),
    "on_phrase":              ("Lands on a phrase boundary", "Not on a phrase boundary"),
    "energy_jump":            ("Strong energy increase", "Energy drops here"),
    "low_jump":               ("Bass enters", "Bass drops out"),
    "density_jump":           ("Rhythmic density increases", "Rhythmic density falls"),
    "energy_pct":             ("High overall energy", "Low overall energy"),
    "low_energy_pct":         ("Strong low end", "Little low end"),
    "onset_density_pct":      ("Busy rhythmic texture", "Sparse rhythmic texture"),
    "preceded_by_buildup":    ("Preceded by a build-up", ""),
    "buildup_bars":           ("Long build-up beforehand", ""),
    "energy_slope":           ("Energy rising through the section", "Energy falling"),
    "high_energy_pct":        ("Bright, high-frequency heavy", "Little high-frequency content"),
    "resolves_to_drop":       ("Resolves into a drop", "Does not resolve into a drop"),
    "next_energy_jump":       ("Big lift into the next section", ""),
    "bass_drop_out":          ("Bass drops out", ""),
    "density_drop":           ("Drums thin out", ""),
    "vocal_confidence":       ("Clear vocal activity", "Weak vocal evidence"),
    "vocal_ratio":            ("Vocal-led section", "Little vocal content"),
    "vocal_length":           ("Sustained vocal phrase", "Very short vocal phrase"),
    "in_labelled_vocal_section": ("Inside a vocal section", ""),
    "has_drums":              ("Beat-driven, easy to mix over", "Few drums to mix against"),
    "stable_energy":          ("Stable, consistent energy", "Energy is changing rapidly"),
    "no_vocals":              ("No vocals to clash", "Vocals present -- may clash"),
    "long_enough":            ("Long enough to beatmatch over", "Short mixing window"),
    "not_too_loud":           ("Leaves headroom for the other track", ""),
    "transition_size":        ("Sits before a large transition", ""),
    "into_drop":              ("Immediately before a drop", ""),
    "section_confidence":     ("Section detected confidently", "Section labelling uncertain"),
    "grid_confidence":        ("Beat grid is reliable", "Beat grid is uncertain"),
    "downbeat_confidence":    ("Downbeat is unambiguous", "Downbeat placement uncertain"),
    "phrase_confidence":      ("Phrase structure is clear", "Phrase structure unclear"),
    "bars_before_drop":       ("Positioned ahead of the drop", ""),
    "position":               ("Late in the track", "Early in the track"),
    "length_bars":            ("Substantial section", "Short section"),
    "seamlessness":           ("Loops back seamlessly", "Audible seam when it repeats"),
    "internal_stability":     ("Musically stable throughout", "Content changes inside the loop"),
    "vocal_cut":              ("", "Cuts a vocal phrase"),
    "loop_on_phrase":         ("Loop starts on a phrase boundary", "Loop is off the phrase grid"),
}

# Features that are scale-like rather than 0/1, and so should be normalised
# before entering the logit. Value is the divisor.
SCALES: dict[str, float] = {"buildup_bars": 16.0, "length_bars": 16.0, "bar_index": 64.0,
                            "bars_before_drop": 8.0}

# Features that must never be reported as a reason: they are context, not evidence.
SILENT = {"bar_index", "position"}


@dataclass
class Scored:
    confidence: float
    reasons: list[Reason]
    logit: float


# Total absolute weight the logit is normalised to before scoring. With a span of
# 6 and an auto-set bias of -span/2, a candidate with every piece of supporting
# evidence at full strength lands near 0.95 confidence, one with average evidence
# near 0.5, and one with none near 0.05.
LOGIT_SPAN = 6.0
# Nothing is ever certain. Reporting 100% confidence on a heuristic is a lie the
# UI would then have to live with, and it destroys the user's ability to
# distinguish "very good" from "merely good".
MAX_CONFIDENCE = 0.97
MIN_CONFIDENCE = 0.02


@lru_cache(maxsize=1)
def load_weights() -> dict:
    if WEIGHTS_PATH.exists():
        return json.loads(WEIGHTS_PATH.read_text())
    return {}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def score(kind: CueKind | str, evidence: dict[str, float],
          notes: dict[str, str] | None = None,
          min_reason_weight: float = 0.12) -> Scored:
    """Score one candidate and produce its explanation."""
    table = load_weights()
    key = kind.value if isinstance(kind, CueKind) else str(kind)
    model = table.get(key) or table.get("_default", {"bias": -0.5, "weights": {}})
    weights: dict[str, float] = dict(model.get("weights", {}))
    notes = notes or {}

    # Calibration. Hand-set weights express *relative* importance -- "a big energy
    # jump matters about twice as much as landing on a downbeat" -- which is the
    # only thing a human can set honestly. Their absolute scale is arbitrary, and
    # left alone it saturates the sigmoid: an early version of this file summed to
    # a logit of +7.9 on a good drop and reported "100% confident", which is both
    # false and useless for ranking. Normalising the total absolute weight to a
    # fixed span and deriving the bias from it makes the output well-spread
    # without anyone hand-tuning a bias per cue type.
    #
    # A model refitted from real user feedback carries its own calibration and
    # sets "calibrated": true to opt out of this.
    if model.get("calibrated"):
        bias = float(model.get("bias", -0.5))
    else:
        total = sum(abs(w) for w in weights.values())
        if total > 1e-9:
            k = LOGIT_SPAN / total
            weights = {n: w * k for n, w in weights.items()}
        bias = -LOGIT_SPAN / 2.0

    logit = bias
    contributions: list[tuple[float, str, str, float]] = []
    for name, w in weights.items():
        if name not in evidence:
            continue
        x = float(evidence[name])
        if name in SCALES:
            x = x / SCALES[name]
        c = w * x
        logit += c
        if name in SILENT:
            continue
        pos, neg = PHRASING.get(name, (name.replace("_", " ").capitalize(), ""))
        text = pos if c >= 0 else neg
        if not text:
            continue
        if name in notes:
            text = notes[name]
        contributions.append((c, name, text, x))

    reasons = [
        Reason(code=n, text=t, weight=round(c, 3), value=round(v, 4))
        for c, n, t, v in sorted(contributions, key=lambda r: -abs(r[0]))
        if abs(c) >= min_reason_weight
    ]
    # Free-text notes the detector attached that are not tied to a weighted
    # feature (e.g. "16-bar build-up immediately before") still belong in the
    # explanation.
    for k, txt in notes.items():
        if k not in weights and txt:
            reasons.append(Reason(code=f"note_{k}", text=txt, weight=0.0))

    conf = min(MAX_CONFIDENCE, max(MIN_CONFIDENCE, _sigmoid(logit)))
    return Scored(confidence=round(conf, 4), reasons=reasons, logit=round(logit, 4))
