#!/usr/bin/env python3
"""Evaluate the analyser against annotated tracks.

Usage:
    python scripts/evaluate.py annotations/*.json [--csv results.csv]

Each annotation file describes one track:

    {
      "audio": "/path/to/track.mp3",
      "bpm": 128.0,
      "downbeat_sec": 0.031,          # time of any known bar-1 downbeat
      "phrase_bars": 8,
      "sections": [{"label": "intro", "start_sec": 0.0, "end_sec": 30.0}, ...],
      "vocal_regions": [{"start_sec": 68.0, "end_sec": 92.0}],
      "cues": [{"kind": "drop", "time_sec": 60.0}]     # optional, your own cues
    }

Metrics, and why each one:

  BPM accuracy          -- exact and octave-tolerant. Reporting both is the point:
                           an octave error is a completely different failure from
                           a 2 BPM error and needs a different fix.
  Downbeat phase        -- fraction of tracks where bar 1 is on the right beat.
                           This is binary and unforgiving because it is: being off
                           by one beat ruins every phrase-aligned cue downstream.
  Boundary P/R/F1       -- MIREX-style, at 0.5-bar and 1-bar tolerance, expressed
                           in bars rather than the conventional 0.5 s/3 s because
                           musical tolerance scales with tempo.
  Frame label accuracy  -- fraction of track duration labelled correctly.
  Vocal F1              -- frame-level.
  Cue distance          -- median |predicted - annotated| in bars, per cue type,
                           plus the fraction landing within half a bar.

Building the annotation set is the real work and there is no way around it. Forty
tracks you know well, annotated in about two hours with any DAW or in Rekordbox
itself, is enough to tell a working detector from a broken one and enough to
catch a regression. Skewing the set toward the genres you actually play is a
feature, not a bias -- so is including a handful of tracks with live drums, tempo
changes or unusual phrasing, because those are where you find out what the
constant-tempo assumption costs.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np
from djprep.audio import features as F
from djprep.audio import io, pipeline
from djprep.models.config import DEFAULT_CONFIG
from djprep.reco.engine import recommend


def boundary_prf(pred: list[float], ref: list[float], tol: float):
    matched_r, matched_p = set(), set()
    for i, p in enumerate(pred):
        for j, r in enumerate(ref):
            if j not in matched_r and abs(p - r) <= tol:
                matched_r.add(j)
                matched_p.add(i)
                break
    prec = len(matched_p) / max(len(pred), 1)
    rec = len(matched_r) / max(len(ref), 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


def octave_equal(a: float, b: float, tol: float = 1.0) -> bool:
    return any(abs(a * m - b) <= tol for m in (0.25, 1 / 3, 0.5, 2 / 3, 1, 1.5, 2, 3, 4))


def evaluate_one(ann: dict) -> dict:
    audio_path = ann["audio"]
    an = pipeline.analyze(audio_path, DEFAULT_CONFIG, track_id=Path(audio_path).stem)
    g = an.beat_grid
    bar = g.bar_period
    row: dict = {"track": Path(audio_path).name, "bpm_pred": round(g.bpm, 2)}

    if "bpm" in ann:
        row["bpm_ref"] = ann["bpm"]
        row["bpm_exact"] = int(abs(g.bpm - ann["bpm"]) <= 0.5)
        row["bpm_octave_ok"] = int(octave_equal(g.bpm, ann["bpm"]))

    if "downbeat_sec" in ann:
        # Is the annotated downbeat also a downbeat on our grid?
        rel = (ann["downbeat_sec"] - g.anchor_sec) / g.beat_period - g.downbeat_offset
        row["downbeat_ok"] = int(abs(rel - round(rel / g.beats_per_bar) * g.beats_per_bar)
                                 < 0.25)

    if "phrase_bars" in ann:
        row["phrase_ok"] = int(an.phrase_grid.phrase_bars == ann["phrase_bars"])

    if ann.get("sections"):
        ref_b = [s["start_sec"] for s in ann["sections"]][1:]
        pred_b = [s.start_sec for s in an.sections][1:]
        for name, tol in (("half_bar", bar * 0.5), ("one_bar", bar * 1.0)):
            p, r, f1 = boundary_prf(pred_b, ref_b, tol)
            row[f"bnd_P_{name}"] = round(p, 3)
            row[f"bnd_R_{name}"] = round(r, 3)
            row[f"bnd_F1_{name}"] = round(f1, 3)

        step = 0.25
        ts = np.arange(0, an.meta.duration_sec - 1, step)
        def ref_at(x):
            for s in ann["sections"]:
                if s["start_sec"] <= x < s["end_sec"]:
                    return s["label"]
            return None
        hits = tot = 0
        for x in ts:
            r_lab = ref_at(float(x))
            if r_lab is None:
                continue
            tot += 1
            sec = an.section_at(float(x))
            hits += int(bool(sec) and sec.label.value == r_lab)
        row["label_acc"] = round(hits / max(tot, 1), 3)

    if ann.get("vocal_regions"):
        audio = io.load(audio_path)
        f = F.extract(audio.y, audio.sr)
        ref = np.zeros(f.n_frames, bool)
        for r in ann["vocal_regions"]:
            ref[f.frame_at(r["start_sec"]):f.frame_at(r["end_sec"])] = True
        pred = np.zeros(f.n_frames, bool)
        for v in an.vocals:
            pred[f.frame_at(v.start_sec):f.frame_at(v.end_sec)] = True
        tp = int((ref & pred).sum())
        fp = int((~ref & pred).sum())
        fn = int((ref & ~pred).sum())
        row["vocal_F1"] = round(2 * tp / max(2 * tp + fp + fn, 1), 3)

    if ann.get("cues"):
        audio = io.load(audio_path)
        prep = recommend(F.extract(audio.y, audio.sr), an, DEFAULT_CONFIG)
        deltas: dict[str, list[float]] = {}
        for ref_cue in ann["cues"]:
            same = [c for c in prep.cues if c.kind.value == ref_cue["kind"]]
            if not same:
                deltas.setdefault(ref_cue["kind"], []).append(float("inf"))
                continue
            best = min(abs(c.time_sec - ref_cue["time_sec"]) for c in same)
            deltas.setdefault(ref_cue["kind"], []).append(best / bar)
        for kind, ds in deltas.items():
            finite = [d for d in ds if np.isfinite(d)]
            row[f"cue_{kind}_median_bars"] = round(statistics.median(finite), 3) if finite else None
            row[f"cue_{kind}_within_half_bar"] = round(
                sum(1 for d in ds if d <= 0.5) / len(ds), 3)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("annotations", nargs="+")
    ap.add_argument("--csv")
    args = ap.parse_args()

    rows = []
    for path in args.annotations:
        ann = json.loads(Path(path).read_text())
        try:
            rows.append(evaluate_one(ann))
            print(f"  ok   {rows[-1]['track']}")
        except Exception as exc:
            print(f"  FAIL {path}: {exc}")

    if not rows:
        raise SystemExit("nothing evaluated")

    print("\n=== aggregate ===")
    keys = sorted({k for r in rows for k in r if isinstance(r.get(k), (int, float))})
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if not vals:
            continue
        print(f"  {k:<28} mean {statistics.mean(vals):6.3f}   "
              f"median {statistics.median(vals):6.3f}   n={len(vals)}")

    if args.csv:
        cols = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
