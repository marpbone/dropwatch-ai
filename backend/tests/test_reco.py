"""Recommendations: the properties a DJ would actually check."""
import pytest

from djprep.models.config import (
    DEFAULT_CONFIG, AnalysisConfig, LoopConfig, LoopLength, LoopLocationConfig,
    LoopSelection,
)
from djprep.models.common import RGB
from djprep.models.prep import CueKind, LoopKind
from djprep.reco.engine import recommend
from djprep.reco import scoring


def test_every_cue_lands_on_a_bar_line(prep, analysis):
    g = analysis.beat_grid
    for c in prep.cues:
        assert abs(c.time_sec - g.snap_to_downbeat(c.time_sec)) < 0.05, \
            f"{c.label} at {c.time_sec:.3f}s is not on a bar line"


def test_phrase_snapped_cues_are_on_phrase_boundaries(prep, analysis):
    """...unless they were clamped, which deliberately overrides snapping.

    A mix cue whose requested offset does not fit inside its section is moved to
    the nearest bar that does fit. Staying on the phrase grid matters less than
    staying inside the track, so the clamp wins -- and says so in its reasons.
    """
    import numpy as np
    b = np.asarray(analysis.phrase_grid.boundaries_sec)
    phrase_kinds = {c.kind for c in DEFAULT_CONFIG.cues if c.snap == "phrase"}
    for c in prep.cues:
        if c.kind not in phrase_kinds:
            continue
        if any(r.code == "note_clamped" for r in c.reasons):
            continue
        assert float(np.min(np.abs(b - c.time_sec))) < 0.06, \
            f"{c.label} at {c.time_sec:.2f}s is off the phrase grid"


def test_drop_cue_is_at_the_actual_drop(prep, truth):
    drops = sorted(c.time_sec for c in prep.cues if c.kind is CueKind.DROP)
    assert drops, "no drop cue produced"
    assert drops[0] == pytest.approx(truth["drop_times_sec"][0], abs=truth["bar_sec"] / 2)


def test_pre_drop_precedes_its_drop_by_the_configured_offset(prep, analysis):
    drops = sorted(c.time_sec for c in prep.cues if c.kind is CueKind.DROP)
    pres = sorted(c.time_sec for c in prep.cues if c.kind is CueKind.PRE_DROP)
    assert pres and drops
    bar = analysis.beat_grid.bar_period
    cfg = DEFAULT_CONFIG.cue_config(CueKind.PRE_DROP)
    for p in pres:
        nearest = min(drops, key=lambda d: abs(d - p))
        assert p < nearest, "pre-drop must come before the drop"
        assert (nearest - p) == pytest.approx(abs(cfg.offset_bars) * bar, abs=bar * 0.6)


def test_every_ai_cue_is_explained(prep):
    for c in prep.cues:
        if c.source.value == "ai":
            assert c.confidence is not None and 0 < c.confidence < 1
            assert c.reasons, f"{c.label} has a confidence but no reasons"


def test_reason_weights_are_signed_contributions(prep):
    """Reasons must be the model's own terms, sorted by how much they mattered."""
    for c in prep.cues:
        if len(c.reasons) > 1:
            mags = [abs(r.weight) for r in c.reasons if r.weight != 0.0]
            assert mags == sorted(mags, reverse=True)


def test_minimum_spacing_is_enforced(prep, analysis):
    gap = DEFAULT_CONFIG.min_cue_spacing_bars * analysis.beat_grid.bar_period
    times = sorted(c.time_sec for c in prep.cues)
    for a, b in zip(times, times[1:]):
        assert b - a >= gap - 1e-6


def test_disabling_a_type_removes_it(feats, analysis):
    cfg = DEFAULT_CONFIG.model_copy(deep=True)
    for c in cfg.cues:
        if c.kind is CueKind.DROP:
            c.enabled = False
    out = recommend(feats, analysis, cfg)
    assert not any(c.kind is CueKind.DROP for c in out.cues)


def test_max_count_is_respected(feats, analysis):
    cfg = DEFAULT_CONFIG.model_copy(deep=True)
    for c in cfg.cues:
        c.max_count = 1
    out = recommend(feats, analysis, cfg)
    from collections import Counter
    assert all(v <= 1 for v in Counter(c.kind for c in out.cues).values())


def test_colour_config_reaches_the_marker(feats, analysis):
    cfg = DEFAULT_CONFIG.model_copy(deep=True)
    pink = RGB.from_hex("#FF00AA")
    for c in cfg.cues:
        if c.kind is CueKind.DROP:
            c.color = pink
    out = recommend(feats, analysis, cfg)
    drops = [c for c in out.cues if c.kind is CueKind.DROP]
    assert drops and all(c.color == pink for c in drops)


def test_loops_are_whole_bars_and_start_on_a_bar(prep, analysis):
    g = analysis.beat_grid
    for l in prep.loops:
        assert abs(l.start_sec - g.snap_to_downbeat(l.start_sec)) < 0.05
        assert l.length_bars == pytest.approx(round(l.length_bars), abs=0.02)


def test_loop_length_config_is_honoured(feats, analysis):
    cfg = DEFAULT_CONFIG.model_copy(deep=True)
    cfg.loops.length = LoopLength.BARS_4
    out = recommend(feats, analysis, cfg)
    assert out.loops and all(l.length_bars == 4 for l in out.loops)


def test_disabling_loops_yields_none(feats, analysis):
    cfg = DEFAULT_CONFIG.model_copy(deep=True)
    cfg.loops.enabled = False
    assert recommend(feats, analysis, cfg).loops == []


def test_loops_of_a_kind_never_overlap(prep):
    from collections import defaultdict
    by = defaultdict(list)
    for l in prep.loops:
        by[l.kind].append(l)
    for group in by.values():
        group.sort(key=lambda l: l.start_sec)
        for a, b in zip(group, group[1:]):
            assert a.end_sec <= b.start_sec + 1e-6


def test_confidence_is_calibrated_not_saturated(prep):
    """Regression: unnormalised weights once made every drop '100% confident'."""
    confs = [c.confidence for c in prep.cues if c.confidence is not None]
    assert confs
    assert max(confs) < 0.98, "confidence is saturating"
    assert min(confs) > 0.02


def test_scoring_is_monotone_in_supporting_evidence():
    base = {"energy_jump": 0.1, "low_jump": 0.1, "on_downbeat": 0.0, "on_phrase": 0.0,
            "preceded_by_buildup": 0.0, "section_confidence": 0.5, "grid_confidence": 0.5}
    strong = {**base, "energy_jump": 0.6, "low_jump": 0.6, "on_downbeat": 1.0,
              "on_phrase": 1.0, "preceded_by_buildup": 1.0}
    assert scoring.score(CueKind.DROP, strong).confidence > \
           scoring.score(CueKind.DROP, base).confidence


def test_unknown_cue_kind_falls_back_to_default_model():
    s = scoring.score("not_a_real_kind", {"on_downbeat": 1.0, "grid_confidence": 0.9})
    assert 0.0 < s.confidence < 1.0
