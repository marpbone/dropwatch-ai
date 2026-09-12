"""Frame-level feature extraction.

One STFT, everything derived from it. Every downstream stage reads a single
`Features` object so no spectrogram is ever computed twice.

Frame rate: hop 512 @ 22050 Hz = 23.2 ms/frame (~43 fps). That is ample for
structure and energy analysis; sub-frame beat timing comes from parabolic
interpolation of the onset envelope in `beats.py`, not from the frame rate.

Performance note (measured, not assumed): a naive implementation spent ~85% of
its time in full-resolution HPSS. Running the harmonic/percussive median filters
on the 128-band mel spectrogram instead of the 1025-bin linear one is ~20x
faster and loses nothing we actually use -- mel resolution in the 200 Hz-4 kHz
vocal region is finer than the decision we make from it. Budget for a 6-minute
track is now ~3 s instead of ~40 s. See docs/PERFORMANCE.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

HOP = 512
N_FFT = 2048
N_MELS = 128

# Bands chosen for what they reveal about dance-music arrangement:
#   sub    -- kick fundamental / 808s. Disappears in breakdowns.
#   bass   -- bassline body. The single best "has the drop landed?" indicator.
#   lowmid -- chords and pads; the mud region filter sweeps travel through.
#   mid    -- vocals, snares, leads.
#   high   -- hats, cymbals, air. Climbs steadily through buildups.
BANDS: dict[str, tuple[float, float]] = {
    "sub": (20, 60),
    "bass": (60, 250),
    "lowmid": (250, 800),
    "mid": (800, 3000),
    "high": (3000, 10000),
}

VOCAL_BAND = (180.0, 4000.0)


@dataclass
class Features:
    sr: int
    hop: int
    n_frames: int
    times: np.ndarray                 # (T,) frame centre times, seconds
    S: np.ndarray                     # (F, T) magnitude STFT
    mel_db: np.ndarray                # (N_MELS, T) log-mel
    mel_freqs: np.ndarray             # (N_MELS,) mel band centre frequencies
    mfcc: np.ndarray                  # (20, T) timbre
    chroma: np.ndarray                # (12, T) pitch class
    rms_db: np.ndarray                # (T,)
    band_energy: dict[str, np.ndarray]
    onset_env: np.ndarray
    onset_times: np.ndarray
    onset_density: np.ndarray
    centroid: np.ndarray
    flatness: np.ndarray
    percussive_ratio: np.ndarray      # (T,) 0=tonal, 1=percussive
    harmonic_mask: np.ndarray         # (N_MELS, T) soft harmonic mask, mel domain
    mel_power: np.ndarray             # (N_MELS, T) linear mel power
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def frame_at(self, t: float) -> int:
        return int(np.clip(round(t * self.sr / self.hop), 0, self.n_frames - 1))

    def slice(self, arr: np.ndarray, t0: float, t1: float) -> np.ndarray:
        a, b = self.frame_at(t0), self.frame_at(t1)
        if b <= a:
            b = min(a + 1, self.n_frames)
        return arr[..., a:b]


def _db(x: np.ndarray, floor: float = -80.0) -> np.ndarray:
    return np.maximum(20.0 * np.log10(np.maximum(x, 1e-10)), floor)


def _soft_hpss(mel_power: np.ndarray, kernel_t: int = 17, kernel_f: int = 9,
               power: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Wiener-style soft HPSS masks on the mel spectrogram.

    Harmonic content is smooth along time (median-filter across time), percussive
    content is smooth along frequency (median-filter across frequency). The two
    filtered versions compete via a soft mask rather than a hard assignment,
    which keeps the ratio continuous and therefore usable as a feature.
    """
    H = median_filter(mel_power, size=(1, kernel_t), mode="nearest")
    P = median_filter(mel_power, size=(kernel_f, 1), mode="nearest")
    Hp, Pp = H ** power, P ** power
    tot = Hp + Pp + 1e-12
    return Hp / tot, Pp / tot


def extract(y: np.ndarray, sr: int, hop: int = HOP, n_fft: int = N_FFT) -> Features:
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    n_frames = S.shape[1]
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    mel_power = librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=N_MELS, fmax=sr // 2)
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmax=sr // 2)
    mel_db = librosa.power_to_db(mel_power, ref=np.max)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(S=S**2, sr=sr)

    rms_db = _db(librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop)[0])

    band_energy: dict[str, np.ndarray] = {}
    for name, (lo, hi) in BANDS.items():
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        band_energy[name] = (
            np.full(n_frames, -80.0) if idx.size == 0
            else _db(np.sqrt((S[idx] ** 2).mean(axis=0)))
        )

    # Onsets: median-aggregated spectral flux (superflux-style). Median aggregation
    # suppresses the false positives that vibrato and slow filter sweeps produce.
    onset_env = librosa.onset.onset_strength(
        S=librosa.power_to_db(S**2, ref=np.max), sr=sr, hop_length=hop, aggregate=np.median
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop, backtrack=False, units="frames"
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)

    # Onset density: onsets/sec over a 2 s sliding window. Climbs sharply through
    # buildups (snare rolls) and stays high through drops.
    spikes = np.zeros(n_frames)
    if onset_frames.size:
        spikes[np.clip(onset_frames, 0, n_frames - 1)] = 1.0
    win = max(1, round(2.0 * sr / hop))
    onset_density = uniform_filter1d(spikes, size=win, mode="nearest") * (sr / hop)

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]

    harmonic_mask, perc_mask = _soft_hpss(mel_power)
    w = mel_power / (mel_power.sum(axis=0, keepdims=True) + 1e-12)
    percussive_ratio = (perc_mask * w).sum(axis=0)

    return Features(
        sr=sr, hop=hop, n_frames=n_frames, times=times, S=S, mel_db=mel_db,
        mel_freqs=mel_freqs, mfcc=mfcc, chroma=chroma, rms_db=rms_db,
        band_energy=band_energy, onset_env=onset_env, onset_times=onset_times,
        onset_density=onset_density, centroid=centroid, flatness=flatness,
        percussive_ratio=percussive_ratio, harmonic_mask=harmonic_mask,
        mel_power=mel_power, y=y,
    )


def smooth(x: np.ndarray, seconds: float, sr: int, hop: int) -> np.ndarray:
    n = max(1, round(seconds * sr / hop))
    return uniform_filter1d(x.astype(float), size=n, mode="nearest")


def pct_rank(reference: np.ndarray, values: np.ndarray | float) -> np.ndarray | float:
    """Where `values` sit inside `reference`'s own distribution, in [0,1].

    Used instead of absolute thresholds throughout, so a quiet dynamic mix and a
    brickwalled master are described on the same scale.
    """
    ref = np.sort(np.asarray(reference).ravel())
    v = np.asarray(values)
    out = np.searchsorted(ref, v, side="right") / max(1, ref.size)
    return float(out) if np.isscalar(values) or v.ndim == 0 else out


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-9 else 1.0)
