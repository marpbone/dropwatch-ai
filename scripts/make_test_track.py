#!/usr/bin/env python3
"""Generate a synthetic EDM track with exact ground-truth annotations.

Why this exists: you cannot commit copyrighted music to a public repo, and you
cannot evaluate a structure detector without labels. This renders a track whose
tempo, downbeat phase, phrase length, section boundaries and vocal regions are
known *by construction*, so CI can assert on them to the bar.

It is not a substitute for real music -- see docs/EVALUATION.md for the real
evaluation protocol -- but it catches every regression that matters: octave
errors, phase-off-by-one, boundaries drifting off the grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def _env(n: int, attack: float, decay: float, sr: int = SR) -> np.ndarray:
    a = max(1, int(attack * sr))
    d = max(1, int(decay * sr))
    e = np.zeros(n)
    a = min(a, n)
    e[:a] = np.linspace(0, 1, a)
    rest = min(d, n - a)
    if rest > 0:
        e[a:a + rest] = np.exp(-np.linspace(0, 5, rest))
    return e


def kick(sr: int = SR, dur: float = 0.28) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = 120 * np.exp(-t * 42) + 45          # pitch sweep: the "thump"
    sig = np.sin(2 * np.pi * np.cumsum(f) / sr)
    click = np.random.default_rng(1).standard_normal(n) * np.exp(-t * 400) * 0.25
    return (sig * _env(n, 0.001, dur) + click) * 0.95


def clap(sr: int = SR, dur: float = 0.22) -> np.ndarray:
    n = int(dur * sr)
    rng = np.random.default_rng(2)
    noise = rng.standard_normal(n)
    from scipy.signal import butter, lfilter
    b, a = butter(4, [900 / (sr / 2), 5200 / (sr / 2)], btype="band")
    return lfilter(b, a, noise) * _env(n, 0.002, dur) * 0.55


def hat(sr: int = SR, dur: float = 0.06, seed: int = 3) -> np.ndarray:
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    from scipy.signal import butter, lfilter
    b, a = butter(4, 7000 / (sr / 2), btype="high")
    return lfilter(b, a, rng.standard_normal(n)) * _env(n, 0.001, dur) * 0.28


def saw(freq: float, n: int, sr: int = SR, detune: float = 0.0) -> np.ndarray:
    t = np.arange(n) / sr
    out = np.zeros(n)
    for d in ([-detune, 0.0, detune] if detune else [0.0]):
        f = freq * (2 ** (d / 1200))
        for h in range(1, 12):
            out += np.sin(2 * np.pi * f * h * t) / h
    return out / (12 * (3 if detune else 1))


def vocal_like(freq: float, n: int, sr: int = SR) -> np.ndarray:
    """A voice-ish source: harmonic, with vibrato and formant peaks.

    Deliberately *not* a pure tone -- the heuristic vocal detector keys on
    continuous pitch modulation and mid-band harmonic energy, so the fixture has
    to exhibit both or the test proves nothing.
    """
    t = np.arange(n) / sr
    vib = 1.0 + 0.022 * np.sin(2 * np.pi * 5.2 * t)         # 5 Hz vibrato
    glide = 1.0 + 0.05 * np.sin(2 * np.pi * 0.35 * t)       # slow pitch glide
    f0 = freq * vib * glide
    ph = 2 * np.pi * np.cumsum(f0) / sr
    formants = [(700, 1.0), (1220, 0.55), (2600, 0.3), (3400, 0.15)]
    sig = np.zeros(n)
    for h in range(1, 22):
        hf = freq * h
        amp = sum(g * np.exp(-((hf - fc) ** 2) / (2 * 260 ** 2)) for fc, g in formants)
        sig += amp * np.sin(ph * h) / h
    tremolo = 0.85 + 0.15 * np.sin(2 * np.pi * 1.1 * t)
    return sig * tremolo * 0.5


def crash(sr: int = SR, dur: float = 1.6) -> np.ndarray:
    """Crash cymbal: long, bright, broadband. Marks section downbeats in
    essentially every dance record, which is what makes downbeat phase
    recoverable in the first place."""
    n = int(dur * sr)
    rng = np.random.default_rng(11)
    from scipy.signal import butter, lfilter
    b, a = butter(2, 3500 / (sr / 2), btype="high")
    noise = lfilter(b, a, rng.standard_normal(n))
    t = np.arange(n) / sr
    return noise * np.exp(-t * 2.4) * 0.5


def riser(n: int, sr: int = SR) -> np.ndarray:
    from scipy.signal import butter, lfilter
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(n)
    t = np.linspace(0, 1, n)
    out = np.zeros(n)
    block = max(1, n // 64)
    for i in range(0, n, block):
        seg = noise[i:i + block]
        if seg.size < 8:
            continue
        cut = 300 + 7000 * (i / max(1, n)) ** 1.6
        b, a = butter(2, min(cut / (sr / 2), 0.98), btype="high")
        out[i:i + block] = lfilter(b, a, seg)
    sweep_f = 200 * np.exp(t * 2.6)
    sweep = np.sin(2 * np.pi * np.cumsum(sweep_f) / sr) * 0.3
    return (out * 0.5 + sweep) * (t ** 1.8)


SECTIONS = [
    # (label, start_bar, end_bar)
    ("intro",     0,   16),
    ("buildup",   16,  32),
    ("drop",      32,  64),
    ("breakdown", 64,  96),
    ("buildup",   96,  112),
    ("drop",      112, 144),
    ("outro",     144, 160),
]
VOCAL_BARS = [(68, 92), (116, 132)]     # breakdown vocal + drop vocal
ROOTS = [55.0, 55.0, 65.41, 61.74]      # A1, A1, C2, B1 per 8-bar group


def render(bpm: float = 128.0, sr: int = SR, seed: int = 42) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar = beat * 4
    total_bars = SECTIONS[-1][2]
    n = int(total_bars * bar * sr) + sr
    out = np.zeros(n)

    K, C, H, CR = kick(sr), clap(sr), hat(sr), crash(sr)

    def place(buf: np.ndarray, sample: np.ndarray, t: float, gain: float = 1.0) -> None:
        i = int(t * sr)
        j = min(len(buf), i + len(sample))
        if i < len(buf):
            buf[i:j] += sample[:j - i] * gain

    def label_at(b: int) -> str:
        for lab, s, e in SECTIONS:
            if s <= b < e:
                return lab
        return "outro"

    for b in range(total_bars):
        t0 = b * bar
        lab = label_at(b)
        pos_in = next((b - s) / (e - s) for _lab, s, e in SECTIONS if s <= b < e)

        drums = lab in ("intro", "buildup", "drop", "outro")
        full = lab == "drop"

        # Crash on the downbeat of every section change: the single most reliable
        # downbeat marker in real dance music.
        if any(b == s for _, s, _ in SECTIONS) and b > 0:
            place(out, CR, t0, 0.9)
        # Crash on the downbeat of every 8-bar phrase inside a drop, quieter.
        elif full and b % 8 == 0:
            place(out, CR, t0, 0.35)

        # Fill on the last bar of every 8-bar phrase: tom/clap run that makes the
        # 8-bar phrase level audibly marked, as it is in real arrangements.
        if drums and b % 8 == 7:
            for k in range(8):
                place(out, C, t0 + k * beat / 2, 0.20 + 0.30 * (k / 8))

        if drums:
            kg = {"intro": 0.55, "buildup": 0.8, "drop": 1.0, "outro": 0.5}[lab]
            if lab == "outro":
                kg *= max(0.15, 1.0 - pos_in)
            # Classic buildup: kick drops out for the last bar before the drop.
            if lab == "buildup" and pos_in > 0.93:
                kg = 0.0
            for beat_i in range(4):
                if kg > 0:
                    place(out, K, t0 + beat_i * beat, kg)
                backbeat = lab in ("drop", "outro") or (lab == "buildup" and pos_in > 0.4)
                if backbeat and beat_i in (1, 3):
                    place(out, C, t0 + beat_i * beat, 0.75 if full else 0.5)
                for off in (0.5,) if not full else (0.25, 0.5, 0.75):
                    place(out, H, t0 + (beat_i + off) * beat, 0.5 if full else 0.32)

        # Snare roll accelerating through the buildup -- the onset-density spike
        # the buildup detector is supposed to find.
        if lab == "buildup" and pos_in > 0.5:
            div = 2 if pos_in < 0.7 else (4 if pos_in < 0.88 else 8)
            for k in range(4 * div):
                place(out, C, t0 + k * beat / div, 0.28 + 0.35 * pos_in)

        seg_n = int(bar * sr)
        idx = slice(int(t0 * sr), int(t0 * sr) + seg_n)
        root = ROOTS[(b // 8) % len(ROOTS)]

        if full:
            # Bass with sidechain ducking on every beat -- the bass-band energy
            # signature that marks a drop.
            bassline = saw(root, seg_n, sr) * 0.85
            duck = np.ones(seg_n)
            for beat_i in range(4):
                s = int(beat_i * beat * sr)
                L = int(0.16 * sr)
                duck[s:s + L] *= np.linspace(0.12, 1.0, min(L, seg_n - s))
            out[idx][:seg_n] += (bassline * duck)[:seg_n]
            lead = sum(saw(root * m, seg_n, sr, detune=12) for m in (4, 5, 6)) * 0.16
            out[idx][:seg_n] += lead[:seg_n]

        if lab == "breakdown":
            pad = sum(saw(root * m, seg_n, sr, detune=18) for m in (2, 2.5, 3, 4)) * 0.14
            out[idx][:seg_n] += pad[:seg_n]

        if lab == "intro":
            # Filtered chord stab, no sub -- intro is thin on purpose.
            from scipy.signal import butter, lfilter
            stab = saw(root * 4, int(beat * sr), sr, detune=8) * 0.2
            bq, aq = butter(2, 2400 / (sr / 2), btype="low")
            stab = lfilter(bq, aq, stab)
            place(out, stab, t0, 0.6)
            place(out, stab, t0 + 2 * beat, 0.6)

        if lab == "buildup":
            r = riser(seg_n, sr) * (0.10 + 0.30 * pos_in)
            out[idx][:seg_n] += r[:seg_n]

        for vs, ve in VOCAL_BARS:
            if vs <= b < ve:
                note = root * 4 * (2 ** ((b % 4) / 12))
                v = vocal_like(note, seg_n, sr) * 0.42
                out[idx][:seg_n] += v[:seg_n]

    out += rng.standard_normal(n) * 0.0015
    out = out / (np.max(np.abs(out)) + 1e-9) * 0.89

    truth = {
        "bpm": bpm,
        "beats_per_bar": 4,
        "anchor_sec": 0.0,
        "phrase_bars": 8,
        "bar_sec": bar,
        "duration_sec": len(out) / sr,
        "sections": [
            {"label": lab, "start_bar": s, "end_bar": e,
             "start_sec": round(s * bar, 4), "end_sec": round(e * bar, 4)}
            for lab, s, e in SECTIONS
        ],
        "vocal_regions": [
            {"start_bar": s, "end_bar": e,
             "start_sec": round(s * bar, 4), "end_sec": round(e * bar, 4)}
            for s, e in VOCAL_BARS
        ],
        "drop_times_sec": [round(s * bar, 4) for lab, s, _ in SECTIONS if lab == "drop"],
        "breakdown_times_sec": [round(s * bar, 4) for lab, s, _ in SECTIONS if lab == "breakdown"],
        "buildup_times_sec": [round(s * bar, 4) for lab, s, _ in SECTIONS if lab == "buildup"],
    }
    return out.astype(np.float32), truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="backend/tests/fixtures/synth_128.wav")
    ap.add_argument("--bpm", type=float, default=128.0)
    args = ap.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio, truth = render(bpm=args.bpm)
    sf.write(str(path), audio, SR)
    truth_path = path.with_suffix(".truth.json")
    truth_path.write_text(json.dumps(truth, indent=2))
    print(f"wrote {path} ({truth['duration_sec']:.1f}s) and {truth_path}")


if __name__ == "__main__":
    main()
