"""Domain models for djprep.

Deliberately free of any DJ-software vendor concepts. There is no `hot_cue_slot`,
no `Num`, no palette index, and no Rekordbox colour in this package. Those are
introduced only in `djprep.export`, by a *lowering* pass that translates this
neutral model into a vendor's constrained representation.
"""

from djprep.models.analysis import (
    BeatGrid,
    EnergyCurve,
    PhraseGrid,
    Section,
    SectionLabel,
    TrackAnalysis,
    TrackMeta,
    VocalSegment,
)
from djprep.models.common import RGB, Reason, Source
from djprep.models.config import (
    DEFAULT_CONFIG,
    AnalysisConfig,
    CueTypeConfig,
    LoopConfig,
    LoopLength,
    LoopSelection,
    VocalMode,
)
from djprep.models.prep import CueKind, CuePoint, Loop, TrackPreparation

__all__ = [
    "DEFAULT_CONFIG",
    "RGB",
    "AnalysisConfig",
    "BeatGrid",
    "CueKind",
    "CuePoint",
    "CueTypeConfig",
    "EnergyCurve",
    "Loop",
    "LoopConfig",
    "LoopLength",
    "LoopSelection",
    "PhraseGrid",
    "Reason",
    "Section",
    "SectionLabel",
    "Source",
    "TrackAnalysis",
    "TrackMeta",
    "TrackPreparation",
    "VocalMode",
    "VocalSegment",
]
