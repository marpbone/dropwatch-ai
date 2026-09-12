"""Export: does the lowered output actually say what we promised.

The point of these tests is that the internal model is vendor-neutral, so the
only place vendor rules can be checked is here.
"""
import xml.etree.ElementTree as ET

import pytest

import djprep.export as EX
from djprep.export.base import ColorMode, Severity
from djprep.export.color import delta_e, get_palette, rgb_to_lab, snap
from djprep.models.common import RGB
from djprep.models.prep import CueKind, CuePoint, Loop, LoopKind, TrackPreparation

WAV = "/tmp/does-not-need-to-exist.wav"


@pytest.fixture
def xml_root(prep, analysis):
    data, _ = EX.get("rekordbox_xml").export(prep, analysis, WAV)
    return ET.fromstring(data)


def test_registry_lists_both_targets():
    ids = {c.format_id for c in EX.available()}
    assert {"rekordbox_xml", "djprep_json"} <= ids
    with pytest.raises(KeyError):
        EX.get("nope")


def test_document_structure(xml_root):
    assert xml_root.tag == "DJ_PLAYLISTS"
    assert xml_root.find("PRODUCT") is not None
    assert xml_root.find("COLLECTION/TRACK") is not None
    assert xml_root.find("PLAYLISTS/NODE/NODE/TRACK") is not None


def test_tempo_element_matches_the_grid(xml_root, analysis):
    tempo = xml_root.find("COLLECTION/TRACK/TEMPO")
    assert float(tempo.get("Bpm")) == pytest.approx(analysis.beat_grid.bpm, abs=0.01)
    assert tempo.get("Metro") == "4/4"
    assert tempo.get("Battito") == "1"
    # Inizio must be a downbeat, not merely any beat.
    assert float(tempo.get("Inizio")) == pytest.approx(
        analysis.beat_grid.downbeat_times()[0], abs=0.01)


def test_hot_cue_slots_are_chronological(xml_root):
    """Pads A-H run left to right under the DJ's hand; the times must too."""
    hot = [(int(pm.get("Num")), float(pm.get("Start")))
           for pm in xml_root.findall("COLLECTION/TRACK/POSITION_MARK")
           if int(pm.get("Num")) >= 0]
    assert hot, "no hot cues assigned"
    by_slot = [t for _, t in sorted(hot)]
    assert by_slot == sorted(by_slot), f"hot cue slots out of time order: {sorted(hot)}"


def test_hot_cue_slots_are_unique_and_within_range(xml_root):
    nums = [int(pm.get("Num")) for pm in xml_root.findall("COLLECTION/TRACK/POSITION_MARK")]
    hot = [n for n in nums if n >= 0]
    assert len(hot) == len(set(hot))
    assert max(hot) <= 7 and min(hot) >= 0


def test_overflow_becomes_memory_cues_and_is_reported(prep, analysis):
    data, report = EX.get("rekordbox_xml").export(prep, analysis, WAV)
    root = ET.fromstring(data)
    total = len(root.findall("COLLECTION/TRACK/POSITION_MARK"))
    hot = sum(1 for pm in root.findall("COLLECTION/TRACK/POSITION_MARK")
              if int(pm.get("Num")) >= 0)
    overflow = total - hot
    reported = sum(1 for n in report.notes if n.code == "hot_cue_overflow")
    assert overflow == reported, "every dropped-to-memory marker must be reported"


def test_loops_carry_an_end_and_cues_do_not(xml_root, prep):
    marks = xml_root.findall("COLLECTION/TRACK/POSITION_MARK")
    with_end = [m for m in marks if m.get("End")]
    assert len(with_end) == len(prep.active_loops())
    for m in with_end:
        assert float(m.get("End")) > float(m.get("Start"))


def test_colours_are_snapped_to_the_target_palette(xml_root):
    palette = {(c["r"], c["g"], c["b"]) for c in get_palette("rekordbox_hotcue")}
    for pm in xml_root.findall("COLLECTION/TRACK/POSITION_MARK"):
        rgb = (int(pm.get("Red")), int(pm.get("Green")), int(pm.get("Blue")))
        assert rgb in palette, f"{rgb} is not a palette colour"


def test_lowering_report_names_everything_it_changed(prep, analysis):
    _, report = EX.get("rekordbox_xml").export(prep, analysis, WAV)
    assert not report.lossless
    codes = {n.code for n in report.notes}
    assert "color_snapped" in codes
    assert "explanations_not_representable" in codes
    for n in report.notes:
        assert n.message and n.severity in Severity


def test_json_export_is_lossless_and_round_trips(prep, analysis):
    import json
    data, report = EX.get("djprep_json").export(prep, analysis, WAV)
    assert report.lossless
    payload = json.loads(data)
    back = TrackPreparation.model_validate(payload["preparation"])
    assert len(back.cues) == len(prep.cues)
    assert back.cues[0].reasons == prep.cues[0].reasons
    assert back.cues[0].confidence == prep.cues[0].confidence


def test_rejected_markers_are_not_exported(analysis):
    col = RGB.from_hex("#3B82F6")
    keep = CuePoint(time_sec=10.0, kind=CueKind.DROP, label="Keep", color=col)
    drop = CuePoint(time_sec=20.0, kind=CueKind.DROP, label="Rejected", color=col,
                    accepted=False)
    p = TrackPreparation(track_id="t", cues=[keep, drop])
    data, _ = EX.get("rekordbox_xml").export(p, analysis, WAV)
    names = {pm.get("Name") for pm in ET.fromstring(data).findall(
        "COLLECTION/TRACK/POSITION_MARK")}
    assert "Keep" in names and "Rejected" not in names


def test_location_is_a_percent_encoded_file_uri(analysis, prep):
    data, _ = EX.get("rekordbox_xml").export(prep, analysis, "/tmp/a b/track #1.wav")
    loc = ET.fromstring(data).find("COLLECTION/TRACK").get("Location")
    assert loc.startswith("file://localhost/")
    assert " " not in loc and "#" not in loc


def test_capabilities_describe_the_real_constraints():
    caps = {c.format_id: c for c in EX.available()}
    rb = caps["rekordbox_xml"]
    assert rb.max_hot_cues == 8
    assert rb.color_mode is ColorMode.PALETTE
    assert len(rb.palette) == 16
    assert caps["djprep_json"].max_hot_cues is None


# --- colour science -------------------------------------------------------- #
def test_lab_conversion_reference_values():
    assert rgb_to_lab((255, 255, 255))[0] == pytest.approx(100.0, abs=0.1)
    assert rgb_to_lab((0, 0, 0))[0] == pytest.approx(0.0, abs=0.1)
    L, a, b = rgb_to_lab((255, 0, 0))
    assert (L, a, b) == pytest.approx((53.24, 80.09, 67.20), abs=0.5)


def test_snapping_picks_the_same_hue_family():
    palette = get_palette("rekordbox_hotcue")
    for hexc, expected in [("#3B82F6", "blue"), ("#EAB308", "yellow"),
                           ("#EC4899", "pink"), ("#22C55E", "green")]:
        out, entry, d = snap(RGB.from_hex(hexc), palette)
        assert expected in entry["name"].lower() or d < 20, \
            f"{hexc} snapped to {entry['name']} (dE {d:.1f})"


def test_delta_e_is_symmetric_and_zero_on_identity():
    a, b = (12, 200, 90), (200, 20, 60)
    assert delta_e(a, a) == pytest.approx(0.0)
    assert delta_e(a, b) == pytest.approx(delta_e(b, a))


def test_empty_palette_is_a_no_op():
    c = RGB.from_hex("#123456")
    out, entry, d = snap(c, [])
    assert out == c and d == 0.0
