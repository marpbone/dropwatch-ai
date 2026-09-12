"""Audio loading.

We decode with librosa/soundfile and fall back to ffmpeg for anything libsndfile
cannot open (notably MP3 variants and M4A). Analysis runs on a mono, fixed-rate
signal: 22050 Hz is plenty for everything we measure (the highest band we care
about tops out around 11 kHz) and halves the cost of every FFT downstream.
"""
from __future__ import annotations

import contextlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

ANALYSIS_SR = 22050


@dataclass
class LoadedAudio:
    y: np.ndarray          # mono float32, ANALYSIS_SR
    sr: int
    duration: float
    source_sr: int
    channels: int


def _ffmpeg_decode(path: Path, sr: int) -> tuple[np.ndarray, int]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found and libsndfile could not decode this file")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = Path(tmp.name)
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(path),
             "-ac", "1", "-ar", str(sr), "-f", "wav", str(out)],
            check=True, capture_output=True,
        )
        y, got_sr = sf.read(str(out), dtype="float32", always_2d=False)
        return np.asarray(y, dtype=np.float32), got_sr
    finally:
        out.unlink(missing_ok=True)


def probe(path: str | Path) -> tuple[int, int, float]:
    """Return (sample_rate, channels, duration) without fully decoding."""
    p = Path(path)
    try:
        info = sf.info(str(p))
        return info.samplerate, info.channels, info.duration
    except Exception:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=sample_rate,channels:format=duration", "-of", "csv=p=0", str(p)],
            capture_output=True, text=True, check=True,
        )
        parts = [x for x in r.stdout.replace("\n", ",").split(",") if x.strip()]
        return int(parts[0]), int(parts[1]), float(parts[2])


def load(path: str | Path, sr: int = ANALYSIS_SR) -> LoadedAudio:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    source_sr, channels = sr, 1
    try:
        info = sf.info(str(p))
        source_sr, channels = info.samplerate, info.channels
        data, native_sr = sf.read(str(p), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if native_sr != sr:
            import librosa
            data = librosa.resample(np.ascontiguousarray(data), orig_sr=native_sr, target_sr=sr)
    except Exception:
        data, got = _ffmpeg_decode(p, sr)
        if got != sr:
            import librosa
            data = librosa.resample(np.ascontiguousarray(data), orig_sr=got, target_sr=sr)
        with contextlib.suppress(Exception):
            source_sr, channels, _ = probe(p)

    y = np.ascontiguousarray(np.nan_to_num(data, copy=False), dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak  # peak-normalise: makes all energy thresholds mastering-independent
    return LoadedAudio(y=y, sr=sr, duration=len(y) / sr, source_sr=source_sr, channels=channels)


def peaks_for_waveform(path: str | Path, n_points: int = 2000) -> list[float]:
    """Min/max envelope for the frontend waveform.

    Computing this server-side and shipping ~2000 floats means WaveSurfer renders
    instantly instead of downloading and decoding the whole file in the browser.
    """
    a = load(path)
    y = a.y
    if y.size == 0:
        return [0.0] * n_points
    block = max(1, len(y) // n_points)
    usable = (len(y) // block) * block
    reshaped = np.abs(y[:usable]).reshape(-1, block)
    env = reshaped.max(axis=1)
    if env.size < n_points:
        env = np.pad(env, (0, n_points - env.size))
    m = float(env.max()) or 1.0
    return (env[:n_points] / m).astype(float).round(4).tolist()
