"""The export boundary.

The whole point of this package is that *nothing outside it* knows what
Rekordbox is. `djprep.models` describes cues and loops the way a DJ thinks about
them: a musical meaning, a position in time, a free-choice colour, and the
evidence behind it. Real DJ software imposes constraints that have nothing to do
with music -- eight hot cue slots, a fixed 16-colour palette, one integer per
marker type -- and those constraints belong here.

The translation is modelled explicitly as a **lowering pass**, borrowing the
compiler term deliberately: a rich representation is lowered onto a constrained
target, and every concession made along the way is recorded in a
`LoweringReport` rather than silently applied. The user gets told "your Vocal In
cue did not fit in the eight hot cue slots, so it was written as a memory cue"
and "#3B82F6 was snapped to the nearest Rekordbox palette blue".

That report is what proves the decoupling is real. If the internal model were
secretly Rekordbox-shaped, there would be nothing to report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from djprep.models.analysis import TrackAnalysis
from djprep.models.prep import TrackPreparation


class ColorMode(str, Enum):
    FREE_RGB = "free_rgb"     # any 24-bit colour survives
    PALETTE = "palette"       # colours snap to a fixed set
    NONE = "none"             # target has no concept of marker colour


class Severity(str, Enum):
    INFO = "info"             # lossless or cosmetic
    ADJUSTED = "adjusted"     # value changed to fit the target
    DROPPED = "dropped"       # information did not survive


@dataclass
class Note:
    severity: Severity
    code: str
    message: str
    marker_id: str | None = None


@dataclass
class LoweringReport:
    target: str
    notes: list[Note] = field(default_factory=list)

    def add(self, severity: Severity, code: str, message: str,
            marker_id: str | None = None) -> None:
        self.notes.append(Note(severity, code, message, marker_id))

    @property
    def lossless(self) -> bool:
        return all(n.severity is Severity.INFO for n in self.notes)

    def summary(self) -> dict:
        counts = {s.value: 0 for s in Severity}
        for n in self.notes:
            counts[n.severity.value] += 1
        return {"target": self.target, "lossless": self.lossless, "counts": counts,
                "notes": [{"severity": n.severity.value, "code": n.code,
                           "message": n.message, "marker_id": n.marker_id}
                          for n in self.notes]}


@dataclass
class ExporterCapabilities:
    """What a target can and cannot represent. Drives the lowering pass.

    Declaring capabilities as data rather than burying them in exporter code
    means the UI can warn the user *before* they export -- "this target supports
    8 hot cues and you have 11 enabled" -- and means a new target is a small
    declaration plus a serialiser, not a new set of special cases everywhere.
    """

    format_id: str
    display_name: str
    file_extension: str
    max_hot_cues: int | None = None          # None = unlimited
    supports_memory_cues: bool = True
    supports_loops: bool = True
    supports_loop_colors: bool = True
    supports_cue_names: bool = True
    color_mode: ColorMode = ColorMode.FREE_RGB
    palette: list[tuple[int, int, int]] = field(default_factory=list)
    notes: str = ""


@runtime_checkable
class Exporter(Protocol):
    caps: ExporterCapabilities

    def export(self, prep: TrackPreparation, analysis: TrackAnalysis,
               audio_path: str, **options) -> tuple[bytes, LoweringReport]:
        ...
