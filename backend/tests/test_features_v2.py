"""Coverage for the second round of features.

Grouped by the behaviour a DJ would describe, not by the module that implements
it -- these are the promises the UI makes.
"""
import numpy as np
import pytest

from djprep.audio import beats, features as F, labeling
from djprep.models.analysis import SectionLabel
from djprep.models.config import DEFAULT_CONFIG, MixOffset
from djprep.models.prep import CueKind
from djprep.reco.engine import recommend

SR = 22050


def _clicks(bpm: float, secs: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(secs * SR)
    y = np.zeros(n)
    period, t = 60.0 / bpm, 0.0
    while t < secs:
        i = int(t * SR)
        L = min(int(0.05 * SR), n - i)
        if L > 0:
            y[i:i + L] += np.sin(2 * np.pi * 70 * np.arange(L) / SR) * np.exp(
                -np.linspace(0, 8, L))
        t += period
    return y + rng.standard_normal(n) * 0.005


# --- every track gets a load point ----------------------------------------- #
def test_initial_cue_always_exists(prep):
    initial = [c for c in prep.cues if c.kind is CueKind.INITIAL]
    assert len(initial) == 1, "every track must get exactly one start cue"
    assert initial[0].time_sec < 5.0


def test_initial_cue_cannot_be_disabled_away(feats, analysis):
    """Even with everything else switched off, the load point survives."""
    cfg = DEFAULT_CONFIG.model_copy(deep=True)
    for c in cfg.cues:
        c.enabled = c.kind is CueKind.INITIAL
    out = recommend(feats, analysis, cfg)
    assert [c.kind for c in out.cues] == [CueKind.INITIAL]


def test_initial_cue_ranks_first_for_slot_allocation(prep):
    """If only one hot cue slot survives export, it should be the load point."""
    first_id = prep.cue_priority[0]
    assert next(c for c in prep.cues if c.id == first_id).kind is CueKind.INITIAL


# --- repeats --------------------------------------------------------------- #
def test_all_repeated_sections_are_marked(prep, analysis):
    n_drops = sum(1 for s in analysis.sections if s.label is SectionLabel.DROP)
    assert n_drops >= 2, "fixture should contain two drops"
    assert sum(1 for c in prep.cues if c.kind is CueKind.DROP) == n_drops


def test_occurrences_are_numbered_in_time_order(analysis):
    drops = [s for s in analysis.sections if s.label is SectionLabel.DROP]
    assert [s.occurrence for s in drops] == list(range(1, len(drops) + 1))


def test_select_all_off_keeps_only_the_best(feats, analysis):
    cfg = DEFAULT_CONFIG.model_copy(deep=True)
    for c in cfg.cues:
        if c.kind is CueKind.DROP:
            c.select_all, c.max_count = False, 1
    out = recommend(feats, analysis, cfg)
    assert sum(1 for c in out.cues if c.kind is CueKind.DROP) == 1


# --- which sections a track actually has ----------------------------------- #
def test_label_summary_reports_presence_and_absence(analysis):
    s = analysis.label_summary
    assert "drop" in s.present and "intro" in s.present
    # The fixture is instrumental, so the vocal-led labels must not appear.
    for lab in ("verse", "chorus", "bridge"):
        assert lab in s.absent
        assert lab not in s.present
    assert s.counts["drop"] >= 2
    assert any("Repeated sections" in n for n in s.notes)


def test_vocal_dependent_labels_need_vocal_evidence(analysis):
    """Assigning 'chorus' to an instrumental is inventing structure, not detecting it."""
    for sec in analysis.sections:
        if sec.label in (SectionLabel.VERSE, SectionLabel.CHORUS, SectionLabel.BRIDGE):
            assert sec.vocal_ratio >= labeling.MIN_VOCAL_RATIO


def test_labels_are_suppressed_not_silently_dropped(analysis):
    assert analysis.label_summary.suppressed
    assert any("instrumental" in n for n in analysis.label_summary.notes)


# --- tempo changes --------------------------------------------------------- #
def test_constant_tempo_track_reports_no_changes(analysis):
    assert analysis.tempo_changes == []


def test_genuine_tempo_change_is_found():
    y = np.concatenate([_clicks(128, 60), _clicks(140, 60, seed=1)]).astype(np.float32)
    changes = beats.detect_tempo_changes(F.extract(y, SR))
    assert len(changes) == 1
    c = changes[0]
    assert c.time_sec == pytest.approx(60.0, abs=8.0)
    assert c.bpm_before == pytest.approx(128, abs=1.5)
    assert c.bpm_after == pytest.approx(140, abs=1.5)
    assert not c.is_octave


def test_drumless_passage_does_not_fake_a_tempo_change():
    """Regression: a quiet passage used to manufacture two changes in a
    rigidly constant-tempo track, because the comb filter will happily lock
    onto pad swells when there are no beats to find."""
    quiet = np.random.default_rng(3).standard_normal(int(40 * SR)) * 0.002
    y = np.concatenate([_clicks(128, 50), quiet, _clicks(128, 50, seed=2)]).astype(np.float32)
    assert beats.detect_tempo_changes(F.extract(y, SR)) == []


def test_octave_change_is_flagged_as_such():
    y = np.concatenate([_clicks(128, 60), _clicks(64, 60, seed=4)]).astype(np.float32)
    changes = beats.detect_tempo_changes(F.extract(y, SR))
    if changes:                     # detection is allowed to miss; mislabelling is not
        assert changes[0].is_octave


# --- mix in / mix out ------------------------------------------------------ #
@pytest.mark.parametrize("bars", list(MixOffset))
def test_mix_cues_exist_at_every_offset(feats, analysis, bars):
    cfg = DEFAULT_CONFIG.model_copy(deep=True)
    cfg.mix.mix_in_bars = bars
    cfg.mix.mix_out_bars = bars
    out = recommend(feats, analysis, cfg)
    kinds = {c.kind for c in out.cues}
    assert CueKind.MIX_IN in kinds, f"no mix-in at {bars.value} bars"
    assert CueKind.MIX_OUT in kinds, f"no mix-out at {bars.value} bars"


def test_mix_in_precedes_the_intro_boundary(feats, analysis):
    cfg = DEFAULT_CONFIG.model_copy(deep=True)
    cfg.mix.mix_in_bars = MixOffset.BARS_4
    out = recommend(feats, analysis, cfg)
    mix_in = next(c for c in out.cues if c.kind is CueKind.MIX_IN)
    boundary = analysis.sections[0].end_sec
    bar = analysis.beat_grid.bar_period
    assert mix_in.time_sec < boundary
    assert boundary - mix_in.time_sec == pytest.approx(4 * bar, abs=bar * 0.6)


def test_mix_out_follows_the_outro_boundary(feats, analysis):
    cfg = DEFAULT_CONFIG.model_copy(deep=True)
    cfg.mix.mix_out_bars = MixOffset.BARS_4
    out = recommend(feats, analysis, cfg)
    mix_out = next(c for c in out.cues if c.kind is CueKind.MIX_OUT)
    outro = [s for s in analysis.sections if s.label is SectionLabel.OUTRO]
    anchor = outro[-1].start_sec if outro else analysis.sections[-1].start_sec
    assert mix_out.time_sec > anchor


def test_mix_cues_stay_inside_the_track(feats, analysis):
    """A 32-bar offset on a short section must not run off either end."""
    cfg = DEFAULT_CONFIG.model_copy(deep=True)
    cfg.mix.mix_in_bars = MixOffset.BARS_32
    cfg.mix.mix_out_bars = MixOffset.BARS_32
    out = recommend(feats, analysis, cfg)
    for c in out.cues:
        if c.kind in (CueKind.MIX_IN, CueKind.MIX_OUT):
            assert 0 <= c.time_sec <= analysis.meta.duration_sec - 1.0


def test_clamping_is_explained(feats, analysis):
    cfg = DEFAULT_CONFIG.model_copy(deep=True)
    cfg.mix.mix_in_bars = MixOffset.BARS_32      # intro is only 16 bars
    out = recommend(feats, analysis, cfg)
    mix_in = next(c for c in out.cues if c.kind is CueKind.MIX_IN)
    texts = " ".join(r.text for r in mix_in.reasons)
    assert "Shortened" in texts, "a clamped cue must say it was clamped"


# --- BPM correction -------------------------------------------------------- #
def test_bpm_lock_is_exact(feats):
    grid = beats.build_beat_grid(feats)
    for factor in (0.5, 2.0):
        locked = beats.build_beat_grid(feats, bpm_lock=grid.bpm * factor)
        assert locked.bpm == pytest.approx(grid.bpm * factor, abs=1e-6)


def test_bpm_lock_does_not_re_search(feats):
    """The point of locking: the DJ's correction must not be overruled by the
    same evidence that produced the wrong answer."""
    locked = beats.build_beat_grid(feats, bpm_lock=97.0)
    assert locked.bpm == pytest.approx(97.0, abs=1e-6)


# --- overlapping sections -------------------------------------------------- #
def test_sections_may_overlap_and_lookup_prefers_the_shorter(analysis):
    an = analysis.model_copy(deep=True)
    wide = an.sections[2]
    narrow = wide.model_copy(deep=True)
    narrow.start_sec = wide.start_sec + 1.0
    narrow.end_sec = wide.start_sec + 5.0
    narrow.label = SectionLabel.BRIDGE
    narrow.is_manual = True
    an.sections.append(narrow)
    probe = wide.start_sec + 2.0
    assert len(an.sections_at(probe)) == 2
    assert an.section_at(probe).label is SectionLabel.BRIDGE


def test_manual_labels_survive_relabelling(analysis):
    an = analysis.model_copy(deep=True)
    an.sections[1].label = SectionLabel.BRIDGE
    an.sections[1].is_manual = True
    labeling.label_sections(an.sections, an.meta.duration_sec, vocals_available=True)
    assert an.sections[1].label is SectionLabel.BRIDGE
