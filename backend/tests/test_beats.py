"""Beat grid: the foundation everything else is measured against.

These assertions are tight on purpose. A tempo error of 0.1 BPM accumulates to a
whole beat of drift over four minutes, and a downbeat off by one makes every
phrase-aligned cue land in the wrong place, so "close enough" is not a useful
standard here.
"""
import numpy as np
import pytest

from djprep.audio import beats, features as F, phrases


def test_bpm_is_exact(feats, truth):
    g = beats.build_beat_grid(feats)
    assert g.bpm == pytest.approx(truth["bpm"], abs=0.05)


def test_anchor_within_ten_milliseconds(feats):
    g = beats.build_beat_grid(feats)
    assert abs(g.anchor_sec) < 0.010, f"grid anchor off by {g.anchor_sec * 1000:.1f} ms"


def test_downbeat_phase_correct_and_confident(feats):
    g = beats.build_beat_grid(feats)
    assert g.downbeat_offset == 0
    assert g.downbeat_confidence > 0.5


def test_bar_32_lands_on_the_drop(feats, truth):
    """The single assertion that matters most: does bar N mean what a DJ thinks."""
    g = beats.build_beat_grid(feats)
    predicted = g.beat_time(32 * g.beats_per_bar + g.downbeat_offset)
    assert predicted == pytest.approx(truth["drop_times_sec"][0], abs=0.02)


def test_grid_survives_a_time_shift(audio, truth):
    """Prepending silence must change the phase, not the tempo or the bar phase.

    This is the regression test for the whole parametric-grid design: a tracker
    that locks onto absolute positions fails here, a properly fitted grid does not.
    """
    shift = 0.937          # deliberately not a whole beat
    sr = audio.sr
    y = np.concatenate([np.zeros(int(shift * sr), dtype=np.float32), audio.y])
    f2 = F.extract(y, sr)
    g2 = beats.build_beat_grid(f2)
    assert g2.bpm == pytest.approx(truth["bpm"], abs=0.05)
    drop = truth["drop_times_sec"][0] + shift
    predicted = g2.beat_time(32 * g2.beats_per_bar + g2.downbeat_offset)
    assert predicted == pytest.approx(drop, abs=0.04)


def test_tempo_octave_is_resolved_not_halved(feats):
    """Regression: mean on-grid energy used to select 85.33 BPM (= 128 x 2/3)."""
    g = beats.build_beat_grid(feats)
    assert 100 < g.bpm < 160, f"metrical level wrong: got {g.bpm}"


def test_bpm_hint_is_honoured(feats):
    g = beats.build_beat_grid(feats, bpm_hint=128.0)
    assert g.bpm == pytest.approx(128.0, abs=0.1)


def test_phrase_length_and_phase(feats, truth):
    g = beats.build_beat_grid(feats)
    pg = phrases.estimate_phrase_grid(feats, g)
    assert pg.phrase_bars == truth["phrase_bars"]
    assert pg.phase_bars == 0
    expected = [i * truth["phrase_bars"] * truth["bar_sec"] for i in range(6)]
    for got, want in zip(pg.boundaries_sec[:6], expected):
        assert got == pytest.approx(want, abs=0.05)


def test_phrase_hierarchy_is_reported(feats):
    """The UI offers 8 vs 16 as a choice, so the evidence for each must survive."""
    g = beats.build_beat_grid(feats)
    pg = phrases.estimate_phrase_grid(feats, g)
    assert {"4", "8", "16"} <= set(pg.alt_lengths)
    assert all(0.0 <= v <= 1.0 for v in pg.alt_lengths.values())


def test_phrase_hint_overrides_detection(feats):
    g = beats.build_beat_grid(feats)
    assert phrases.estimate_phrase_grid(feats, g, phrase_hint=16).phrase_bars == 16
