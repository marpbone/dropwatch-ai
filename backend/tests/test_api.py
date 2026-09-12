"""End-to-end through the HTTP layer."""
import os
import tempfile
import time
from pathlib import Path

import pytest

os.environ["DJPREP_DATA"] = tempfile.mkdtemp(prefix="djprep-test-")

from fastapi.testclient import TestClient  # noqa: E402

from djprep.api.main import app  # noqa: E402

WAV = Path(__file__).parent / "fixtures" / "synth_128.wav"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def track(client):
    with WAV.open("rb") as fh:
        r = client.post("/api/tracks", files={"file": ("synth_128.wav", fh, "audio/wav")})
    assert r.status_code == 201
    tid = r.json()["track_id"]

    job = client.post(f"/api/tracks/{tid}/analyze").json()
    deadline = time.time() + 180
    while time.time() < deadline:
        s = client.get(f"/api/jobs/{job['id']}").json()
        if s["status"] == "done":
            break
        assert s["status"] != "error", s.get("error")
        time.sleep(0.4)
    else:
        pytest.fail("analysis did not finish in time")
    return tid


def test_health_and_capabilities(client):
    assert client.get("/api/health").json()["status"] == "ok"
    fmts = client.get("/api/export/formats").json()
    assert any(f["format_id"] == "rekordbox_xml" and f["max_hot_cues"] == 8 for f in fmts)


def test_default_config_is_valid_and_complete(client):
    cfg = client.get("/api/config/default").json()
    assert len(cfg["cues"]) >= 10
    assert {"kind", "enabled", "color", "snap", "max_count"} <= set(cfg["cues"][0])
    assert cfg["loops"]["locations"]


def test_rejects_unsupported_file_type(client):
    r = client.post("/api/tracks", files={"file": ("x.txt", b"nope", "text/plain")})
    assert r.status_code == 400


def test_analysis_and_preparation_available(client, track, truth):
    an = client.get(f"/api/tracks/{track}/analysis").json()
    assert an["beat_grid"]["bpm"] == pytest.approx(truth["bpm"], abs=0.05)
    assert an["phrase_grid"]["phrase_bars"] == truth["phrase_bars"]
    assert len(an["sections"]) >= 5

    prep = client.get(f"/api/tracks/{track}/preparation").json()
    assert prep["cues"] and prep["loops"]
    assert all(c["reasons"] for c in prep["cues"] if c["source"] == "ai")


def test_waveform_peaks_are_normalised(client, track):
    wf = client.get(f"/api/tracks/{track}/waveform?points=800").json()
    assert wf["points"] == 800
    assert wf["duration_sec"] > 0
    assert 0.0 <= min(wf["peaks"]) and max(wf["peaks"]) <= 1.0


def test_reconfiguring_is_far_cheaper_than_analysing(client, track):
    """The whole reason /recommend exists as a separate endpoint."""
    cfg = client.get("/api/config/default").json()
    for c in cfg["cues"]:
        if c["kind"] == "drop":
            c["enabled"] = False
    t0 = time.time()
    prep = client.post(f"/api/tracks/{track}/recommend", json=cfg).json()
    elapsed = time.time() - t0
    assert not any(c["kind"] == "drop" for c in prep["cues"])
    assert elapsed < 2.0, f"recommendation took {elapsed:.2f}s; it must stay interactive"


def test_colour_config_round_trips_through_the_api(client, track):
    cfg = client.get("/api/config/default").json()
    for c in cfg["cues"]:
        if c["kind"] == "breakdown":
            c["enabled"] = True
            c["color"] = {"r": 255, "g": 0, "b": 170}
    prep = client.post(f"/api/tracks/{track}/recommend", json=cfg).json()
    bd = [c for c in prep["cues"] if c["kind"] == "breakdown"]
    assert bd and all(c["color"] == {"r": 255, "g": 0, "b": 170} for c in bd)


def test_user_edits_are_persisted(client, track):
    prep = client.get(f"/api/tracks/{track}/preparation").json()
    prep["cues"][0]["time_sec"] = 99.5
    prep["cues"][0]["label"] = "Hand placed"
    prep["cues"][0]["source"] = "user"
    assert client.put(f"/api/tracks/{track}/preparation", json=prep).status_code == 200
    back = client.get(f"/api/tracks/{track}/preparation").json()
    hand = [c for c in back["cues"] if c["label"] == "Hand placed"]
    assert hand and hand[0]["time_sec"] == 99.5


def test_export_preview_reports_what_will_be_lost(client, track):
    r = client.post(f"/api/tracks/{track}/export/preview?format_id=rekordbox_xml").json()
    assert r["bytes"] > 500
    assert r["preview"].lstrip().startswith("<?xml")
    assert not r["report"]["lossless"]
    assert r["report"]["counts"]["adjusted"] > 0


def test_export_download_has_attachment_headers(client, track):
    r = client.post(f"/api/tracks/{track}/export?format_id=rekordbox_xml")
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    assert r.headers["x-lowering-lossless"] == "false"
    assert r.content.lstrip().startswith(b"<?xml")


def test_unknown_export_format_is_a_400(client, track):
    assert client.post(f"/api/tracks/{track}/export?format_id=bogus").status_code == 400


def test_feedback_delta_is_recorded_in_bars(client, track, truth):
    """Bars, not seconds: a 4-bar bias means the same thing at any tempo."""
    prep = client.get(f"/api/tracks/{track}/preparation").json()
    cue = prep["cues"][0]
    moved = cue["time_sec"] + truth["bar_sec"] * 4
    r = client.post(f"/api/tracks/{track}/feedback", json=[{
        "marker_id": cue["id"], "marker_class": "cue", "kind": cue["kind"],
        "action": "moved", "original_time": cue["time_sec"], "final_time": moved,
        "confidence": cue["confidence"],
    }])
    assert r.json()["logged"] == 1
    stats = client.get("/api/feedback/stats").json()
    moved_rows = [s for s in stats if s["action"] == "moved"]
    assert moved_rows and moved_rows[0]["avg_abs_delta_bars"] == pytest.approx(4.0, abs=0.01)


def test_missing_track_is_a_404(client):
    assert client.get("/api/tracks/deadbeef/analysis").status_code == 404


def test_analysis_before_upload_is_a_409(client):
    with WAV.open("rb") as fh:
        tid = client.post("/api/tracks",
                          files={"file": ("b.wav", fh, "audio/wav")}).json()["track_id"]
    assert client.get(f"/api/tracks/{tid}/analysis").status_code == 409


# --------------------------------------------------------------------------- #
# Second-round features
# --------------------------------------------------------------------------- #
def test_analysis_reports_sections_and_tempo(client, track):
    an = client.get(f"/api/tracks/{track}/analysis").json()
    assert "label_summary" in an and "tempo_changes" in an
    s = an["label_summary"]
    assert "drop" in s["present"]
    assert set(s["absent"]) >= {"verse", "chorus", "bridge"}
    assert all("occurrence" in sec for sec in an["sections"])


def test_relabelling_a_section_sticks_and_is_marked_manual(client, track):
    an = client.get(f"/api/tracks/{track}/analysis").json()
    idx = next(i for i, s in enumerate(an["sections"]) if s["label"] == "drop")
    out = client.put(f"/api/tracks/{track}/sections",
                     json=[{"index": idx, "label": "chorus"}]).json()
    assert out["sections"][idx]["label"] == "chorus"
    assert out["sections"][idx]["is_manual"] is True
    # and it survives a round trip
    again = client.get(f"/api/tracks/{track}/analysis").json()
    assert again["sections"][idx]["label"] == "chorus"
    client.put(f"/api/tracks/{track}/sections", json=[{"index": idx, "label": "drop"}])


def test_sections_may_be_edited_into_overlap(client, track):
    """Detection produces a partition; only a human can create an overlap."""
    an = client.get(f"/api/tracks/{track}/analysis").json()
    target = an["sections"][2]
    out = client.put(f"/api/tracks/{track}/sections", json=[
        {"index": 2, "start_sec": target["start_sec"] - 4.0},
    ]).json()
    edited = out["sections"][2]
    assert edited["start_sec"] < an["sections"][1]["end_sec"]
    client.put(f"/api/tracks/{track}/sections",
               json=[{"index": 2, "start_sec": target["start_sec"]}])


def test_unknown_section_label_is_rejected(client, track):
    r = client.put(f"/api/tracks/{track}/sections",
                   json=[{"index": 0, "label": "banger"}])
    assert r.status_code == 400


def test_bpm_halving_and_doubling_round_trip(client, track):
    """The common real correction: the detector picked the wrong octave."""
    before = client.get(f"/api/tracks/{track}/analysis").json()["beat_grid"]["bpm"]
    halved = client.post(f"/api/tracks/{track}/grid/scale?factor=0.5").json()
    assert halved["beat_grid"]["bpm"] == pytest.approx(before / 2, abs=0.01)
    restored = client.post(f"/api/tracks/{track}/grid/scale?factor=2").json()
    assert restored["beat_grid"]["bpm"] == pytest.approx(before, abs=0.01)


def test_bpm_scale_rebuilds_the_cues_too(client, track):
    """A corrected grid with stale cues would be worse than no correction."""
    before = client.get(f"/api/tracks/{track}/preparation").json()
    client.post(f"/api/tracks/{track}/grid/scale?factor=0.5")
    after = client.get(f"/api/tracks/{track}/preparation").json()
    assert after["cues"], "cues must survive a grid rescale"
    assert [c["time_sec"] for c in after["cues"]] != [c["time_sec"] for c in before["cues"]] \
        or len(after["cues"]) != len(before["cues"])
    client.post(f"/api/tracks/{track}/grid/scale?factor=2")


def test_bpm_scale_rejects_a_nonsense_factor(client, track):
    assert client.post(f"/api/tracks/{track}/grid/scale?factor=3").status_code == 400


def test_bpm_scale_refuses_to_leave_the_plausible_range(client, track):
    """Repeated presses must not walk the grid off into nonsense."""
    for _ in range(3):
        r = client.post(f"/api/tracks/{track}/grid/scale?factor=2")
        if r.status_code == 400:
            break
    else:
        pytest.fail("doubling was never refused")
    # put it back
    for _ in range(3):
        if client.post(f"/api/tracks/{track}/grid/scale?factor=0.5").status_code == 400:
            break


def test_mix_offsets_change_where_the_cues_land(client, track):
    cfg = client.get("/api/config/default").json()
    cfg["mix"]["mix_in_bars"] = "4"
    near = client.post(f"/api/tracks/{track}/recommend", json=cfg).json()
    cfg["mix"]["mix_in_bars"] = "16"
    far = client.post(f"/api/tracks/{track}/recommend", json=cfg).json()
    t_near = next(c["time_sec"] for c in near["cues"] if c["kind"] == "mix_in")
    t_far = next(c["time_sec"] for c in far["cues"] if c["kind"] == "mix_in")
    assert t_far < t_near, "a larger offset must place the mix-in earlier"


def test_frontend_is_served_from_the_api(client):
    """One command, one port: the built app is served by the same process."""
    r = client.get("/")
    if r.status_code == 404:
        pytest.skip("frontend not built in this environment")
    assert r.status_code == 200
    assert b"<div id=\"root\">" in r.content
