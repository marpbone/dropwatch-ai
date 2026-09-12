"""Grid arithmetic and colour primitives -- pure, fast, no audio."""
import pytest

from djprep.models.analysis import BeatGrid
from djprep.models.common import RGB
from djprep.models.config import DEFAULT_CONFIG
from djprep.models.prep import CueKind, CuePoint, Loop, LoopKind, Source, TrackPreparation


def grid() -> BeatGrid:
    return BeatGrid(bpm=128.0, anchor_sec=0.0, n_beats=1000, downbeat_offset=0)


def test_beat_and_bar_periods():
    g = grid()
    assert g.beat_period == pytest.approx(0.46875)
    assert g.bar_period == pytest.approx(1.875)


@pytest.mark.parametrize("t,expected_bar", [(0.0, 0), (1.9, 1), (60.0, 32), (59.9, 32)])
def test_bar_index(t, expected_bar):
    assert grid().bar_index(t) == expected_bar


def test_snap_to_downbeat_is_idempotent():
    g = grid()
    for t in (0.3, 7.2, 61.4, 200.9):
        once = g.snap_to_downbeat(t)
        assert g.snap_to_downbeat(once) == pytest.approx(once)
        # and the result really is a bar line
        assert (once / g.bar_period) == pytest.approx(round(once / g.bar_period), abs=1e-6)


def test_downbeat_offset_shifts_bar_lines():
    g = BeatGrid(bpm=128.0, anchor_sec=0.0, n_beats=1000, downbeat_offset=1)
    assert g.downbeat_times()[0] == pytest.approx(g.beat_period)


def test_rgb_hex_roundtrip():
    for h in ("#3B82F6", "#000000", "#FFFFFF", "#A855F7"):
        assert RGB.from_hex(h).to_hex() == h
    with pytest.raises(ValueError):
        RGB.from_hex("nope")


def test_rejected_markers_are_inactive_but_kept():
    """Rejecting is not deleting: the marker stays for the feedback log."""
    c = CuePoint(time_sec=1.0, color=RGB.from_hex("#FFFFFF"), accepted=False)
    assert not c.is_active
    p = TrackPreparation(track_id="t", cues=[c])
    assert p.cues == [c] and p.active_cues() == []


def test_priority_ordering_falls_back_to_time():
    col = RGB.from_hex("#FFFFFF")
    a = CuePoint(id="a", time_sec=10.0, color=col)
    b = CuePoint(id="b", time_sec=5.0, color=col)
    p = TrackPreparation(track_id="t", cues=[a, b], cue_priority=["a"])
    assert [c.id for c in p.prioritised_cues()] == ["a", "b"]


def test_default_config_has_no_duplicate_kinds():
    kinds = [c.kind for c in DEFAULT_CONFIG.cues]
    assert len(kinds) == len(set(kinds))


def test_loop_duration():
    l = Loop(start_sec=0.0, end_sec=15.0, length_bars=8, kind=LoopKind.INTRO,
             color=RGB.from_hex("#14B8A6"))
    assert l.duration == 15.0
    assert l.source is Source.AI


def test_cue_kind_covers_every_configured_type():
    for c in DEFAULT_CONFIG.cues:
        assert isinstance(c.kind, CueKind)
