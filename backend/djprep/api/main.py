"""HTTP API.

Design notes worth stating explicitly:

* **Analysis and recommendation are separate endpoints.** Analysis is expensive
  (seconds) and depends only on the audio. Recommendation is cheap
  (milliseconds) and depends on user configuration. Toggling a checkbox in the UI
  therefore re-runs `/recommend`, not `/analyze` -- which is what makes the
  configuration panel feel live instead of like a batch job.

* **The config model is the contract.** `AnalysisConfig` in `djprep.models.config`
  is the request body, is what FastAPI validates against, and is what generates
  the OpenAPI schema the frontend's TypeScript types are derived from. There is
  exactly one definition of what the user can ask for.

* **Waveform peaks are computed server-side.** Shipping ~2000 floats lets the
  editor render instantly instead of downloading and decoding the whole file in
  the browser.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import djprep.export as EX
from djprep.api.jobs import JobManager
from djprep.api.store import Store
from djprep.audio import features as F
from djprep.audio import io, labeling, pipeline
from djprep.models.analysis import Section, SectionLabel, TrackAnalysis
from djprep.models.config import DEFAULT_CONFIG, AnalysisConfig
from djprep.models.prep import TrackPreparation
from djprep.reco.engine import recommend

DATA_DIR = Path(os.environ.get("DJPREP_DATA", Path.home() / ".djprep"))
UPLOAD_DIR = DATA_DIR / "audio"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED = {".mp3", ".wav", ".flac", ".aiff", ".aif", ".m4a", ".ogg"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

app = FastAPI(title="djprep", version="0.1.0",
              description="AI-assisted DJ track preparation")
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)

store = Store(DATA_DIR / "djprep.db")
jobs = JobManager()

# Feature objects are expensive to compute and needed again by /recommend.
# A tiny LRU keyed by track id keeps the config panel responsive without
# recomputing a spectrogram on every checkbox.
_feature_cache: dict[str, F.Features] = {}
_feature_order: list[str] = []
FEATURE_CACHE_SIZE = 3


def _cache_features(track_id: str, feats: F.Features) -> None:
    _feature_cache[track_id] = feats
    if track_id in _feature_order:
        _feature_order.remove(track_id)
    _feature_order.append(track_id)
    while len(_feature_order) > FEATURE_CACHE_SIZE:
        _feature_cache.pop(_feature_order.pop(0), None)


def _features_for(track_id: str, path: str) -> F.Features:
    if track_id not in _feature_cache:
        audio = io.load(path)
        _cache_features(track_id, F.extract(audio.y, audio.sr))
    return _feature_cache[track_id]


def _track_or_404(track_id: str) -> dict:
    t = store.get_track(track_id)
    if not t:
        raise HTTPException(404, f"track {track_id} not found")
    return t


def _analysis_or_404(track_id: str) -> TrackAnalysis:
    raw = store.get_analysis(track_id)
    if not raw:
        raise HTTPException(409, "track has not been analysed yet")
    return TrackAnalysis.model_validate(raw)


# --------------------------------------------------------------------------- #
# config + capabilities
# --------------------------------------------------------------------------- #
@app.get("/api/config/default", response_model=AnalysisConfig)
def default_config() -> AnalysisConfig:
    """The default cue/loop configuration the UI renders itself from."""
    return DEFAULT_CONFIG


@app.get("/api/export/formats")
def export_formats() -> list[dict]:
    """Declared capabilities of every export target.

    The UI uses this to warn *before* export -- "Rekordbox supports 8 hot cues,
    you have 11 cues enabled" -- rather than surprising the user afterwards.
    """
    return [{
        "format_id": c.format_id, "display_name": c.display_name,
        "file_extension": c.file_extension, "max_hot_cues": c.max_hot_cues,
        "supports_loops": c.supports_loops, "supports_cue_names": c.supports_cue_names,
        "color_mode": c.color_mode.value, "palette_size": len(c.palette),
        "palette": [{"r": r, "g": g, "b": b} for r, g, b in c.palette],
        "notes": c.notes,
    } for c in EX.available()]


# --------------------------------------------------------------------------- #
# tracks
# --------------------------------------------------------------------------- #
@app.get("/api/tracks")
def list_tracks() -> list[dict]:
    return store.list_tracks()


@app.post("/api/tracks", status_code=201)
async def upload_track(file: UploadFile = File(...)) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"unsupported file type {ext!r}; "
                                 f"allowed: {sorted(ALLOWED)}")
    tid_path = UPLOAD_DIR / f"{os.urandom(6).hex()}{ext}"
    size = 0
    with tid_path.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                fh.close()
                tid_path.unlink(missing_ok=True)
                raise HTTPException(413, "file too large (limit 200 MB)")
            fh.write(chunk)
    try:
        sr, ch, dur = io.probe(tid_path)
    except Exception as exc:
        tid_path.unlink(missing_ok=True)
        raise HTTPException(400, "could not decode audio file") from exc
    track_id = store.create_track(file.filename or tid_path.name, str(tid_path), dur)
    return {"track_id": track_id, "filename": file.filename,
            "duration_sec": round(dur, 3), "sample_rate": sr, "channels": ch}


@app.delete("/api/tracks/{track_id}", status_code=204)
def delete_track(track_id: str) -> Response:
    t = _track_or_404(track_id)
    Path(t["path"]).unlink(missing_ok=True)
    store.delete_track(track_id)
    _feature_cache.pop(track_id, None)
    return Response(status_code=204)


@app.get("/api/tracks/{track_id}/audio")
def get_audio(track_id: str) -> FileResponse:
    t = _track_or_404(track_id)
    return FileResponse(t["path"], filename=t["filename"])


@app.get("/api/tracks/{track_id}/waveform")
def get_waveform(track_id: str, points: int = 2000) -> dict:
    t = _track_or_404(track_id)
    peaks = io.peaks_for_waveform(t["path"], n_points=max(200, min(points, 8000)))
    duration = t["duration_sec"]
    if not duration:
        duration = io.probe(t["path"])[2]
    return {"track_id": track_id, "points": len(peaks), "peaks": peaks,
            "duration_sec": round(float(duration), 3)}


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
@app.post("/api/tracks/{track_id}/analyze", status_code=202)
def analyze_track(track_id: str, config: AnalysisConfig | None = None) -> dict:
    t = _track_or_404(track_id)
    cfg = config or DEFAULT_CONFIG

    def work(progress):
        an = pipeline.analyze(t["path"], cfg, track_id=track_id, progress=progress)
        store.save_analysis(track_id, an.analysis_version, an.model_dump(mode="json"))
        audio = io.load(t["path"])
        feats = F.extract(audio.y, audio.sr)
        _cache_features(track_id, feats)
        prep = recommend(feats, an, cfg)
        store.save_preparation(track_id, prep.model_dump(mode="json"),
                               cfg.model_dump(mode="json"))
        return {"track_id": track_id, "cues": len(prep.cues), "loops": len(prep.loops)}

    return jobs.submit("analyze", work).as_dict()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    d = job.as_dict()
    if job.status == "done":
        d["result"] = job.result
    return d


@app.get("/api/tracks/{track_id}/analysis", response_model=TrackAnalysis)
def get_analysis(track_id: str) -> TrackAnalysis:
    _track_or_404(track_id)
    return _analysis_or_404(track_id)


# --------------------------------------------------------------------------- #
# recommendation + editing
# --------------------------------------------------------------------------- #
@app.post("/api/tracks/{track_id}/recommend", response_model=TrackPreparation)
def rerun_recommend(track_id: str, config: AnalysisConfig) -> TrackPreparation:
    """Re-run recommendation with new configuration. Cheap -- no DSP re-run.

    This is the endpoint the configuration panel calls on every change.
    """
    t = _track_or_404(track_id)
    an = _analysis_or_404(track_id)
    feats = _features_for(track_id, t["path"])
    prep = recommend(feats, an, config)
    store.save_preparation(track_id, prep.model_dump(mode="json"),
                           config.model_dump(mode="json"))
    return prep


@app.post("/api/tracks/{track_id}/grid/scale", response_model=TrackAnalysis)
def scale_grid(track_id: str, factor: float = 2.0) -> TrackAnalysis:
    """Halve or double the detected tempo and re-derive everything that depends on it.

    Octave errors are the single most common tempo failure -- 87 and 174 BPM
    explain the same onsets -- and when one happens the DJ knows immediately and
    the analyser does not. So this is a correction, not a re-analysis: the
    expensive spectral work is reused from cache, the tempo is *locked* to the
    corrected value rather than re-searched (searching again would only reproduce
    the original mistake), and the grid, phrase structure and cues are rebuilt on
    top. Roughly a second instead of fifteen.

    Section boundaries keep their labels and are re-snapped onto the new grid.
    """
    if factor not in (0.5, 2.0):
        raise HTTPException(400, "factor must be 0.5 or 2.0")
    t = _track_or_404(track_id)
    an = _analysis_or_404(track_id)
    feats = _features_for(track_id, t["path"])

    new_bpm = an.beat_grid.bpm * factor
    # Wide enough for genres that genuinely live out there (hardcore, footwork,
    # half-time hip-hop) while still stopping a runaway from repeated presses.
    lo, hi = 40.0, 300.0
    if not (lo <= new_bpm <= hi):
        raise HTTPException(400, f"{new_bpm:.1f} BPM is outside the plausible range")

    from djprep.audio import beats as _beats
    from djprep.audio import phrases as _phrases

    grid = _beats.build_beat_grid(feats, beats_per_bar=an.beat_grid.beats_per_bar,
                                  bpm_lock=new_bpm)
    an.beat_grid = grid
    an.phrase_grid = _phrases.estimate_phrase_grid(feats, grid)

    # Re-snap boundaries onto the new grid; labels and manual edits are kept.
    for sec in an.sections:
        sec.start_sec = round(grid.snap_to_downbeat(sec.start_sec), 3)
        sec.end_sec = round(grid.snap_to_downbeat(sec.end_sec), 3)
        sec.start_bar = grid.bar_index(sec.start_sec)
        sec.length_bars = round((sec.end_sec - sec.start_sec) / grid.bar_period, 2)
    an.sections = [s for s in an.sections if s.end_sec > s.start_sec]
    labeling.number_occurrences(an.sections)

    store.save_analysis(track_id, an.analysis_version, an.model_dump(mode="json"))

    cfg_raw = store.get_config(track_id)
    cfg = AnalysisConfig.model_validate(cfg_raw) if cfg_raw else DEFAULT_CONFIG
    prep = recommend(feats, an, cfg)
    store.save_preparation(track_id, prep.model_dump(mode="json"),
                           cfg.model_dump(mode="json"))
    return an


class SectionEdit(BaseModel):
    """One hand-made change to a section."""

    index: int | None = Field(None, description="Existing section to modify.")
    start_sec: float | None = None
    end_sec: float | None = None
    label: str | None = None
    # Whether this edit should mark the section as the user's. Sent explicitly
    # rather than assumed, so a client that re-sends untouched sections does not
    # accidentally brand the whole track as hand-edited.
    is_manual: bool | None = None
    delete: bool = False


@app.put("/api/tracks/{track_id}/sections", response_model=TrackAnalysis)
def edit_sections(track_id: str, edits: list[SectionEdit]) -> TrackAnalysis:
    """Apply hand edits to the detected sections.

    Sections are stored as independent intervals rather than a strict partition,
    so an edit may legitimately produce overlap -- a vocal-led passage that
    starts before the drop lands is a real thing and the model should be able to
    say so. Detection still emits a partition; only the user can create overlap.

    Every touched section is marked `is_manual`, which makes it immune to
    relabelling if the recommender is re-run. A decision a person made outranks
    anything the analyser would infer.
    """
    _track_or_404(track_id)
    an = _analysis_or_404(track_id)
    sections = list(an.sections)

    for e in edits:
        if e.index is None:
            if e.start_sec is None or e.end_sec is None:
                raise HTTPException(400, "new sections need start_sec and end_sec")
            sections.append(Section(
                start_sec=e.start_sec, end_sec=e.end_sec,
                start_bar=an.beat_grid.bar_index(e.start_sec),
                length_bars=round((e.end_sec - e.start_sec) / an.beat_grid.bar_period, 2),
                label=SectionLabel(e.label) if e.label else SectionLabel.UNKNOWN,
                confidence=1.0, is_manual=True))
            continue
        if not (0 <= e.index < len(sections)):
            raise HTTPException(400, f"no section at index {e.index}")
        sec = sections[e.index]
        if e.delete:
            sec = None
        else:
            if e.start_sec is not None:
                sec.start_sec = e.start_sec
                sec.start_bar = an.beat_grid.bar_index(e.start_sec)
            if e.end_sec is not None:
                sec.end_sec = e.end_sec
            if e.label is not None:
                try:
                    sec.label = SectionLabel(e.label)
                except ValueError as exc:
                    raise HTTPException(400, f"unknown label {e.label!r}") from exc
                sec.ambiguous_with = None      # the user has settled it
                sec.ambiguity_margin = None
            sec.length_bars = round(
                (sec.end_sec - sec.start_sec) / an.beat_grid.bar_period, 2)
            if e.is_manual is not False:
                sec.confidence = 1.0
                sec.is_manual = True
        sections[e.index] = sec

    an.sections = sorted([s for s in sections if s is not None],
                         key=lambda s: s.start_sec)
    labeling.number_occurrences(an.sections)
    an.label_summary = labeling.summarise_labels(
        an.sections, set(an.label_summary.suppressed),
        vocals_available=bool(an.vocals) or an.label_summary.present != [])
    store.save_analysis(track_id, an.analysis_version, an.model_dump(mode="json"))
    return an


@app.get("/api/tracks/{track_id}/preparation", response_model=TrackPreparation)
def get_preparation(track_id: str) -> TrackPreparation:
    _track_or_404(track_id)
    raw = store.get_preparation(track_id)
    if not raw:
        raise HTTPException(409, "no preparation yet; run /analyze first")
    return TrackPreparation.model_validate(raw)


@app.put("/api/tracks/{track_id}/preparation", response_model=TrackPreparation)
def put_preparation(track_id: str, prep: TrackPreparation) -> TrackPreparation:
    """Save the user's edited preparation. The DJ's version always wins."""
    _track_or_404(track_id)
    store.save_preparation(track_id, prep.model_dump(mode="json"),
                           store.get_config(track_id))
    return prep


# --------------------------------------------------------------------------- #
# feedback (human-in-the-loop)
# --------------------------------------------------------------------------- #
class FeedbackItem(BaseModel):
    marker_id: str
    marker_class: str = Field("cue", pattern="^(cue|loop)$")
    kind: str = "custom"
    action: str = Field(..., pattern="^(accepted|rejected|moved|deleted|recoloured)$")
    original_time: float | None = None
    final_time: float | None = None
    confidence: float | None = None
    evidence: dict | None = None


@app.post("/api/tracks/{track_id}/feedback")
def post_feedback(track_id: str, items: list[FeedbackItem]) -> dict:
    """Record what the user did with each recommendation.

    This is the training set. `delta_bars` -- how far the DJ dragged an AI cue,
    expressed in bars rather than seconds so it is tempo-independent -- is the
    single most informative signal the application produces: a systematic bias of
    -4 bars on pre-drop cues is a weight that needs refitting, not a user error.
    """
    _track_or_404(track_id)
    an = store.get_analysis(track_id)
    bar = None
    if an:
        g = an["beat_grid"]
        bar = (60.0 / g["bpm"]) * g["beats_per_bar"]

    rows = []
    for it in items:
        delta = None
        if it.original_time is not None and it.final_time is not None and bar:
            delta = round((it.final_time - it.original_time) / bar, 4)
        rows.append({**it.model_dump(), "track_id": track_id, "delta_bars": delta})
    return {"logged": store.log_feedback(rows)}


@app.get("/api/feedback/stats")
def feedback_stats() -> list[dict]:
    return store.feedback_stats()


@app.get("/api/feedback/export")
def feedback_export() -> list[dict]:
    """Raw feedback, for scripts/refit_weights.py."""
    return store.export_feedback()


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
@app.post("/api/tracks/{track_id}/export")
def export_track(track_id: str, format_id: str = "rekordbox_xml",
                 prep: TrackPreparation | None = None) -> Response:
    t = _track_or_404(track_id)
    an = _analysis_or_404(track_id)
    if prep is None:
        raw = store.get_preparation(track_id)
        if not raw:
            raise HTTPException(409, "no preparation to export")
        prep = TrackPreparation.model_validate(raw)
    try:
        exporter = EX.get(format_id)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc

    data, report = exporter.export(prep, an, t["path"])
    name = Path(t["filename"]).stem + exporter.caps.file_extension
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
            # The lowering report rides along in a header so a plain download
            # still carries it; /export/preview returns it as JSON for the UI.
            "X-Lowering-Lossless": str(report.lossless).lower(),
            "X-Lowering-Notes": str(len(report.notes)),
        },
    )


@app.post("/api/tracks/{track_id}/export/preview")
def export_preview(track_id: str, format_id: str = "rekordbox_xml",
                   prep: TrackPreparation | None = None) -> dict:
    """What would be exported, and everything that would be lost doing it."""
    t = _track_or_404(track_id)
    an = _analysis_or_404(track_id)
    if prep is None:
        raw = store.get_preparation(track_id)
        if not raw:
            raise HTTPException(409, "no preparation to export")
        prep = TrackPreparation.model_validate(raw)
    exporter = EX.get(format_id)
    data, report = exporter.export(prep, an, t["path"])
    return {"format_id": format_id, "bytes": len(data),
            "preview": data.decode("utf-8", errors="replace")[:20000],
            "report": report.summary()}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": "0.2.0",
            "formats": [c.format_id for c in EX.available()]}


# --------------------------------------------------------------------------- #
# Static frontend
# --------------------------------------------------------------------------- #
# Serving the built React app from the same process turns two dev servers and two
# ports into one command and one URL, and removes Node from the runtime entirely
# -- it becomes a build-time dependency only. Mounted last so every /api route
# above wins; the catch-all returns index.html for unknown paths so client-side
# routing keeps working.
FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"


def _mount_frontend() -> None:
    if not FRONTEND_DIST.is_dir():
        return

    assets = FRONTEND_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/", include_in_schema=False)
    def _index() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    def _spa(path: str) -> FileResponse:
        candidate = (FRONTEND_DIST / path).resolve()
        if (candidate.is_file()
                and FRONTEND_DIST.resolve() in candidate.parents):
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")


_mount_frontend()
