#!/usr/bin/env python3
"""Refit the cue-scoring model from real user decisions.

    python scripts/refit_weights.py --db ~/.djprep/djprep.db --out backend/djprep/data/cue_weights.json

This is the human-in-the-loop loop closing. Every accept, reject, move and delete
the editor records lands in the `feedback` table together with the evidence that
produced the original recommendation. That is a labelled dataset:

    x = the evidence features       y = 1 if the DJ kept the cue, 0 if not

and the scoring model is a logistic regression over exactly those features, so
refitting is ordinary maximum-likelihood estimation, done here with plain
gradient descent and L2 regularisation (a dozen weights and a few hundred rows
does not need a solver dependency).

Two design points worth defending:

  *Why logistic regression and not something bigger.* The realistic data volume
  is hundreds of examples, not millions. A model with a dozen parameters can be
  fit meaningfully from that; anything larger would memorise. Capacity should
  match the data you can actually collect, and here that is a small, honest model
  whose coefficients remain the user-facing explanation.

  *Why a `moved` action is not simply a rejection.* If the DJ dragged a drop cue
  by less than a bar they agreed with it and were nudging; if they dragged it
  sixteen bars they disagreed. `--move-tolerance-bars` draws that line, and the
  systematic component of the residual is separately reported, because a
  consistent -4 bar bias on one cue type is not a weighting problem at all -- it
  means the detector's anchor is wrong and should be moved.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_EXAMPLES = 30


def load_rows(db: Path) -> list[dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM feedback")]
    for r in rows:
        r["evidence"] = json.loads(r["evidence"]) if r["evidence"] else {}
    return rows


def label_for(row: dict, move_tol: float) -> int | None:
    action = row["action"]
    if action == "accepted":
        return 1
    if action in ("rejected", "deleted"):
        return 0
    if action == "moved":
        d = row.get("delta_bars")
        if d is None:
            return None
        return 1 if abs(d) <= move_tol else 0
    return None


def fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 4000,
        lr: float = 0.15) -> tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        gw = X.T @ (p - y) / n + l2 * w / n
        gb = float((p - y).mean())
        w -= lr * gw
        b -= lr * gb
    return w, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(Path.home() / ".djprep" / "djprep.db"))
    ap.add_argument("--out", default="backend/djprep/data/cue_weights.json")
    ap.add_argument("--move-tolerance-bars", type=float, default=1.0)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"no feedback database at {db}")
    rows = load_rows(db)

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if label_for(r, args.move_tolerance_bars) is not None and r["evidence"]:
            by_kind[r["kind"]].append(r)

    existing = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    out = dict(existing)
    report = []

    for kind, group in sorted(by_kind.items()):
        if len(group) < MIN_EXAMPLES:
            report.append(f"  {kind:<12} {len(group):>4} examples — skipped "
                          f"(need {MIN_EXAMPLES})")
            continue
        feats = sorted({k for r in group for k in r["evidence"]})
        X = np.array([[float(r["evidence"].get(f, 0.0)) for f in feats] for r in group])
        y = np.array([label_for(r, args.move_tolerance_bars) for r in group], dtype=float)

        if len(set(y.tolist())) < 2:
            report.append(f"  {kind:<12} {len(group):>4} examples — skipped "
                          f"(all labels identical)")
            continue

        # Standardise so the L2 penalty is even-handed across differently scaled
        # features; fold the transform back into the weights afterwards so the
        # exported model consumes raw evidence exactly as the scorer produces it.
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd < 1e-9] = 1.0
        w, b = fit((X - mu) / sd, y, l2=args.l2)
        raw_w = w / sd
        raw_b = float(b - float(raw_w @ mu))

        p = 1 / (1 + np.exp(-(X @ raw_w + raw_b)))
        acc = float(((p >= 0.5) == (y >= 0.5)).mean())

        out[kind] = {
            "bias": round(raw_b, 4),
            "calibrated": True,       # fitted models bring their own scale
            "weights": {f: round(float(v), 4) for f, v in zip(feats, raw_w, strict=True)},
            "fit": {"n": len(group), "accuracy": round(acc, 3),
                    "kept_fraction": round(float(y.mean()), 3)},
        }
        report.append(f"  {kind:<12} {len(group):>4} examples  train acc {acc:.3f}  "
                      f"kept {y.mean():.0%}")

    # Systematic drag bias: a weighting problem's signature is scattered
    # residuals, an anchor problem's is a consistent offset.
    print("refit summary")
    print("\n".join(report) or "  no usable feedback yet")
    print("\nsystematic drag offsets (median bars moved, AI cues only):")
    drags: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["action"] == "moved" and r.get("delta_bars") is not None:
            drags[r["kind"]].append(r["delta_bars"])
    for kind, ds in sorted(drags.items()):
        med = float(np.median(ds))
        flag = "  <-- anchor looks systematically wrong" if abs(med) >= 1.0 and len(ds) >= 8 else ""
        print(f"  {kind:<12} n={len(ds):>4}  median {med:+.2f} bars{flag}")

    if args.dry_run:
        print("\n(dry run — nothing written)")
        return
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
