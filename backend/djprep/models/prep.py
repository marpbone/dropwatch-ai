"""The editable preparation: cues and loops the DJ will actually use.

Vendor-neutral by construction. A `CuePoint` knows *what it means musically*
(kind), *where it is* (time + grid coordinates), *what it looks like* (free RGB)
and *why it exists* (reasons). It knows nothing about hot cue slots, memory cues,
palette indices or XML.
"""
from __future__ import annotations

import uuid
from enum import Enum

from pydantic import BaseModel, Field

from djprep.models.common import RGB, Reason, Source


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class CueKind(str, Enum):
    """Musical/functional meaning of a marker.

    MIX_IN / MIX_OUT are separate from INTRO / OUTRO on purpose: the intro is a
    section of the arrangement, whereas the mix-in point is the specific bar you
    would start the track from in a blend. They often differ.
    """

    # Always emitted, always first: the point a DJ loads the track to. Kept
    # distinct from INTRO because the intro is a *section* and this is a single
    # load position -- on a track with two bars of noise before the music, they
    # are not the same place.
    INITIAL = "initial"
    INTRO = "intro"
    MIX_IN = "mix_in"
    BUILDUP = "buildup"
    PRE_DROP = "pre_drop"
    DROP = "drop"
    BREAKDOWN = "breakdown"
    CHORUS = "chorus"
    VOCAL_IN = "vocal_in"
    VOCAL_OUT = "vocal_out"
    MIX_OUT = "mix_out"
    OUTRO = "outro"
    TEMPO_CHANGE = "tempo_change"
    FX = "fx"
    CUSTOM = "custom"


class LoopKind(str, Enum):
    INTRO = "intro"
    OUTRO = "outro"
    BREAKDOWN = "breakdown"
    PRE_DROP = "pre_drop"
    DROP = "drop"
    VOCAL = "vocal"
    CUSTOM = "custom"


class _Marker(BaseModel):
    id: str = Field(default_factory=_new_id)
    label: str = ""
    color: RGB
    source: Source = Source.AI
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    reasons: list[Reason] = Field(default_factory=list)
    accepted: bool | None = Field(
        None, description="None = pending review, True = kept, False = rejected."
    )
    # Set when the user drags an AI marker; the delta is the single most valuable
    # training signal this application produces (see docs/EVALUATION.md).
    original_time_sec: float | None = None

    @property
    def is_active(self) -> bool:
        """Exported markers: user-made, or AI-made and not rejected."""
        return self.accepted is not False


class CuePoint(_Marker):
    time_sec: float
    kind: CueKind = CueKind.CUSTOM
    beat_index: int | None = None
    bar_index: int | None = None
    on_downbeat: bool = False
    on_phrase: bool = False

    def moved_by(self) -> float | None:
        if self.original_time_sec is None:
            return None
        return self.time_sec - self.original_time_sec


class Loop(_Marker):
    start_sec: float
    end_sec: float
    length_bars: float
    kind: LoopKind = LoopKind.CUSTOM
    start_beat_index: int | None = None
    start_bar_index: int | None = None

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


class TrackPreparation(BaseModel):
    """Everything the DJ has decided about a track. The unit of export."""

    track_id: str
    cues: list[CuePoint] = Field(default_factory=list)
    loops: list[Loop] = Field(default_factory=list)
    # Ordering hint for exporters with limited slots: earlier = higher priority.
    # The user can reorder this in the UI to control which cues win hot cue slots.
    cue_priority: list[str] = Field(default_factory=list)
    notes: str = ""

    def active_cues(self) -> list[CuePoint]:
        return sorted([c for c in self.cues if c.is_active], key=lambda c: c.time_sec)

    def active_loops(self) -> list[Loop]:
        return sorted([lp for lp in self.loops if lp.is_active], key=lambda lp: lp.start_sec)

    def prioritised_cues(self) -> list[CuePoint]:
        """Active cues ordered by explicit priority, then by time."""
        active = self.active_cues()
        rank = {cid: i for i, cid in enumerate(self.cue_priority)}
        return sorted(active, key=lambda c: (rank.get(c.id, 10_000), c.time_sec))

    def by_id(self, marker_id: str) -> CuePoint | Loop | None:
        for c in self.cues:
            if c.id == marker_id:
                return c
        for lp in self.loops:
            if lp.id == marker_id:
                return lp
        return None
