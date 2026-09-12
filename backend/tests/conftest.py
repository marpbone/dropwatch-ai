"""Shared fixtures.

The analysis is computed once per session and reused: it takes ~10 s and every
test in the structure/reco/export suites needs it. Tests must not mutate the
shared objects -- those that need to edit a preparation build their own.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
WAV = FIXTURES / "synth_128.wav"
TRUTH_JSON = FIXTURES / "synth_128.truth.json"


def pytest_configure(config):
    if not WAV.exists():
        pytest.exit("Test fixture missing. Run: python scripts/make_test_track.py", 1)


@pytest.fixture(scope="session")
def truth() -> dict:
    return json.loads(TRUTH_JSON.read_text())


@pytest.fixture(scope="session")
def audio():
    from djprep.audio import io
    return io.load(WAV)


@pytest.fixture(scope="session")
def feats(audio):
    from djprep.audio import features as F
    return F.extract(audio.y, audio.sr)


@pytest.fixture(scope="session")
def analysis():
    from djprep.audio import pipeline
    from djprep.models.config import DEFAULT_CONFIG
    return pipeline.analyze(WAV, DEFAULT_CONFIG, track_id="test")


@pytest.fixture(scope="session")
def prep(feats, analysis):
    from djprep.models.config import DEFAULT_CONFIG
    from djprep.reco.engine import recommend
    return recommend(feats, analysis, DEFAULT_CONFIG)
