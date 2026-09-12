#!/usr/bin/env python3
"""Seed the database with the synthetic fixture, analysed and prepared.

Gives a fresh checkout something to look at immediately, and gives CI and the
screenshot test a deterministic track to run against.
"""
import os
import shutil
import sys
from pathlib import Path

DATA = Path(os.environ.get("DJPREP_DATA", Path.home() / ".djprep"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from djprep.api.store import Store
from djprep.audio import features as F
from djprep.audio import io, pipeline
from djprep.models.config import DEFAULT_CONFIG
from djprep.reco.engine import recommend


def main() -> None:
    src = Path(__file__).resolve().parents[1] / "backend/tests/fixtures/synth_128.wav"
    if not src.exists():
        raise SystemExit("fixture missing — run scripts/make_test_track.py first")
    (DATA / "audio").mkdir(parents=True, exist_ok=True)
    dst = DATA / "audio" / src.name
    shutil.copy(src, dst)

    store = Store(DATA / "djprep.db")
    for t in store.list_tracks():
        store.delete_track(t["id"])
    _sr, _ch, dur = io.probe(dst)
    tid = store.create_track(src.name, str(dst), duration=dur,
                             title="Synthetic Demo", artist="djprep")

    an = pipeline.analyze(dst, DEFAULT_CONFIG, track_id=tid)
    store.save_analysis(tid, an.analysis_version, an.model_dump(mode="json"))
    audio = io.load(dst)
    prep = recommend(F.extract(audio.y, audio.sr), an, DEFAULT_CONFIG)
    store.save_preparation(tid, prep.model_dump(mode="json"),
                           DEFAULT_CONFIG.model_dump(mode="json"))
    print(f"seeded track {tid}: {an.beat_grid.bpm} BPM, {len(an.sections)} sections, "
          f"{len(prep.cues)} cues, {len(prep.loops)} loops")

if __name__ == "__main__":
    main()
