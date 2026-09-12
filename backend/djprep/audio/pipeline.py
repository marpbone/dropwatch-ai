"""Orchestration: audio file in, TrackAnalysis out.

Deliberately separate from recommendation. Analysis is expensive and depends
only on the audio; recommendation is cheap and depends on user configuration.
Splitting them means changing a checkbox in the UI re-runs milliseconds of work
instead of seconds, and the analysis can be cached and versioned independently.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from djprep.audio import beats, io, labeling, phrases, structure, vocals
from djprep.audio import features as feat
from djprep.models.analysis import EnergyCurve, TrackAnalysis, TrackMeta
from djprep.models.config import AnalysisConfig

ANALYSIS_VERSION = "1.2"
ProgressFn = Callable[[str, float], None]


def _decimate(x: np.ndarray, n: int) -> list[float]:
    if x.size == 0:
        return []
    if x.size <= n:
        return [round(float(v), 3) for v in x]
    idx = np.linspace(0, x.size - 1, n).astype(int)
    return [round(float(v), 3) for v in x[idx]]


def analyze(path: str | Path, config: AnalysisConfig | None = None,
            track_id: str = "", progress: ProgressFn | None = None,
            curve_points: int = 1500) -> TrackAnalysis:
    cfg = config or AnalysisConfig()
    timings: dict[str, float] = {}

    def step(name: str, frac: float) -> None:
        if progress:
            progress(name, frac)

    def timed(name: str, fn):
        t0 = time.perf_counter()
        out = fn()
        timings[name] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    step("loading", 0.02)
    audio = timed("load", lambda: io.load(path))

    step("features", 0.10)
    f = timed("features", lambda: feat.extract(audio.y, audio.sr))

    step("beat grid", 0.35)
    grid = timed("beats", lambda: beats.build_beat_grid(
        f, bpm_range=cfg.bpm_range, bpm_hint=cfg.bpm_hint,
        beats_per_bar=cfg.beats_per_bar))

    step("phrases", 0.55)
    pg = timed("phrases", lambda: phrases.estimate_phrase_grid(
        f, grid, phrase_hint=cfg.phrase_bars_hint))

    step("vocals", 0.65)
    detector = vocals.get_detector(cfg.vocal_mode.value)
    vocal_like = np.zeros(f.n_frames)
    vocal_segs = []
    if detector is not None:
        def _run_vocals():
            try:
                return detector.likelihood(f)
            except RuntimeError:
                # Separation requested but unavailable: degrade to the heuristic
                # rather than failing the whole analysis.
                return vocals.HeuristicVocalDetector().likelihood(f)
        vocal_like = timed("vocals", _run_vocals)
        vocal_segs = vocals.segments_from_likelihood(vocal_like, f)

    step("tempo changes", 0.72)
    tempo_changes = timed("tempo_changes", lambda: beats.detect_tempo_changes(
        f, bpm_range=cfg.bpm_range))

    step("structure", 0.80)
    bounds = timed("boundaries", lambda: structure.detect_boundaries(f, grid, pg))
    bounds = timed("merge", lambda: structure.merge_similar(f, grid, bounds))
    sections = structure.segment_stats(f, grid, bounds,
                                       vocal_like if detector else None)
    structure.cluster_sections(f, grid, sections)
    vocals_available = detector is not None
    suppressed = timed("labeling", lambda: labeling.label_sections(
        sections, audio.duration, vocals_available=vocals_available))
    label_summary = labeling.summarise_labels(sections, suppressed, vocals_available)
    if tempo_changes:
        label_summary.notes.append(
            f"{len(tempo_changes)} tempo change(s) detected — the constant-tempo "
            f"grid is only valid up to the first one.")

    step("packaging", 0.95)
    energy = EnergyCurve(
        times_sec=_decimate(f.times, curve_points),
        rms_db=_decimate(f.rms_db, curve_points),
        low=_decimate(f.band_energy["sub"] + f.band_energy["bass"], curve_points),
        mid=_decimate(f.band_energy["mid"], curve_points),
        high=_decimate(f.band_energy["high"], curve_points),
        onset_density=_decimate(f.onset_density, curve_points),
        vocal_likelihood=_decimate(vocal_like, curve_points) if detector else [],
    )

    step("done", 1.0)
    return TrackAnalysis(
        meta=TrackMeta(
            track_id=track_id or Path(path).stem,
            filename=Path(path).name,
            duration_sec=round(audio.duration, 3),
            sample_rate=audio.source_sr,
            channels=audio.channels,
        ),
        beat_grid=grid, phrase_grid=pg, sections=sections, vocals=vocal_segs,
        energy=energy, analysis_version=ANALYSIS_VERSION, timings_ms=timings,
        tempo_changes=tempo_changes, label_summary=label_summary,
    )
