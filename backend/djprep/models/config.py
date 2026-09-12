"""User configuration -- the single contract between frontend and backend.

This module is the *only* place where "what the user asked for" is defined. The
frontend renders itself from these models (TypeScript types are generated from
the OpenAPI schema, see scripts/gen_types.sh), the API validates against them,
and the recommendation engine consumes them. Adding a new cue type means editing
exactly one file.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from djprep.models.common import RGB
from djprep.models.prep import CueKind, LoopKind


class LoopLength(str, Enum):
    BARS_1 = "1"
    BARS_2 = "2"
    BARS_4 = "4"
    BARS_8 = "8"
    BARS_16 = "16"
    BARS_32 = "32"
    AUTO = "auto"

    def bars(self) -> float | None:
        return None if self is LoopLength.AUTO else float(self.value)


class LoopSelection(str, Enum):
    """How many loops to keep per enabled location."""

    FIRST = "first"
    ALL = "all"
    TOP_N = "top_n"


class VocalMode(str, Enum):
    """Vocal detection strategy -- a speed/accuracy dial exposed to the user.

    HEURISTIC is ~0.5s for a 6-minute track and confuses sustained synth leads
    with voices. SEPARATION runs a source-separation model to isolate the vocal
    stem and is far more accurate but costs roughly 0.5-2x realtime on CPU.
    """

    OFF = "off"
    HEURISTIC = "heuristic"
    SEPARATION = "separation"


class MixOffset(str, Enum):
    """How far from the section boundary to place a mix cue, in bars."""

    BARS_4 = "4"
    BARS_8 = "8"
    BARS_16 = "16"
    BARS_32 = "32"

    def bars(self) -> float:
        return float(self.value)


class CueTypeConfig(BaseModel):
    """One row of the cue checklist in the UI."""

    kind: CueKind
    enabled: bool = True
    color: RGB
    max_count: int = Field(1, ge=1, le=16)
    # Mark every occurrence rather than only the best `max_count`. Structural
    # markers default to this: a track with three drops should get three drop
    # cues, because the DJ needs all of them, not the most confident one.
    select_all: bool = False
    min_confidence: float = Field(0.35, ge=0.0, le=1.0)
    # Grid policy: snap to bar line, or all the way to a phrase boundary.
    snap: str = Field("downbeat", pattern="^(beat|downbeat|phrase)$")
    # Offset in bars applied after snapping. Negative = earlier. Lets a DJ say
    # "put my mix-in cue 8 bars before the intro actually starts".
    offset_bars: float = 0.0
    label_template: str = ""


class LoopLocationConfig(BaseModel):
    kind: LoopKind
    enabled: bool = True


class LoopConfig(BaseModel):
    enabled: bool = True
    locations: list[LoopLocationConfig] = Field(default_factory=list)
    length: LoopLength = LoopLength.BARS_8
    color: RGB = RGB(r=0, g=200, b=180)
    selection: LoopSelection = LoopSelection.FIRST
    top_n: int = Field(2, ge=1, le=16)
    min_confidence: float = Field(0.35, ge=0.0, le=1.0)
    # A loop that cuts a vocal phrase in half sounds wrong; on by default.
    avoid_vocal_cuts: bool = True

    def enabled_kinds(self) -> set[LoopKind]:
        return {loc.kind for loc in self.locations if loc.enabled}


class MixConfig(BaseModel):
    """Mix-in / mix-out cue placement.

    Anchored to section boundaries rather than measured independently: the
    mix-in point is N bars *before* the intro gives way to the body of the track
    (so you have N bars of stable intro to blend over), and the mix-out point is
    N bars *after* the outro begins.


    Enabling and colour live on the MIX_IN / MIX_OUT rows of the cue checklist
    like every other cue type; only the offsets are configured here, because
    only the offsets are specific to mixing.
    """

    mix_in_bars: MixOffset = MixOffset.BARS_16
    mix_out_bars: MixOffset = MixOffset.BARS_16


class AnalysisConfig(BaseModel):
    """The complete request body for POST /tracks/{id}/analyze."""

    cues: list[CueTypeConfig] = Field(default_factory=list)
    loops: LoopConfig = Field(default_factory=LoopConfig)
    mix: MixConfig = Field(default_factory=MixConfig)
    vocal_mode: VocalMode = VocalMode.HEURISTIC

    # Grid hints. Most tracks need none of these, but a DJ who knows the tempo
    # should be able to say so rather than fight a halftime/doubletime error.
    bpm_hint: float | None = Field(None, gt=40, lt=250)
    bpm_range: tuple[float, float] = (70.0, 190.0)
    beats_per_bar: int = Field(4, ge=2, le=12)
    phrase_bars_hint: int | None = Field(None, description="Force 4/8/16/32-bar phrasing.")

    # Global spacing constraint: never place two cues closer than this many bars.
    min_cue_spacing_bars: float = Field(2.0, ge=0.0)
    # Sections whose label margin falls below this are still shown, but flagged
    # ambiguous rather than presented as settled.
    min_label_confidence: float = Field(0.35, ge=0.0, le=1.0)
    # Total cue budget across all types (Rekordbox surfaces 8 hot cues; extras
    # become memory cues, so the sane default is generous, not 8).
    max_total_cues: int = Field(32, ge=1, le=64)

    def enabled_cues(self) -> list[CueTypeConfig]:
        return [c for c in self.cues if c.enabled]

    def cue_config(self, kind: CueKind) -> CueTypeConfig | None:
        for c in self.cues:
            if c.kind == kind:
                return c
        return None

    @model_validator(mode="after")
    def _dedupe(self) -> AnalysisConfig:
        seen = set()
        for c in self.cues:
            if c.kind in seen:
                raise ValueError(f"duplicate cue config for {c.kind}")
            seen.add(c.kind)
        return self


def _c(kind: CueKind, hex_color: str, enabled: bool = True, **kw) -> CueTypeConfig:
    return CueTypeConfig(kind=kind, color=RGB.from_hex(hex_color), enabled=enabled, **kw)


DEFAULT_CONFIG = AnalysisConfig(
    cues=[
        _c(CueKind.INITIAL,   "#FFFFFF", snap="downbeat", max_count=1),
        _c(CueKind.MIX_IN,    "#22C55E", snap="phrase", max_count=1),
        _c(CueKind.INTRO,     "#16A34A", snap="phrase", max_count=2, select_all=True),
        _c(CueKind.BUILDUP,   "#EAB308", snap="phrase", max_count=8, select_all=True),
        _c(CueKind.PRE_DROP,  "#F97316", snap="downbeat", max_count=8,
           select_all=True, offset_bars=-4),
        _c(CueKind.DROP,      "#3B82F6", snap="downbeat", max_count=8, select_all=True),
        _c(CueKind.BREAKDOWN, "#A855F7", snap="phrase", max_count=8, select_all=True),
        _c(CueKind.VOCAL_IN,  "#EC4899", enabled=False, snap="downbeat", max_count=4),
        _c(CueKind.VOCAL_OUT, "#DB2777", enabled=False, snap="downbeat", max_count=2),
        _c(CueKind.MIX_OUT,   "#F59E0B", snap="phrase", max_count=1),
        _c(CueKind.OUTRO,     "#EA580C", snap="phrase", max_count=2, select_all=True),
        _c(CueKind.CHORUS,    "#06B6D4", enabled=False, snap="phrase", max_count=8,
           select_all=True),
        _c(CueKind.TEMPO_CHANGE, "#F43F5E", snap="downbeat", max_count=8,
           select_all=True),
        _c(CueKind.FX,        "#94A3B8", enabled=False, snap="downbeat", max_count=6),
    ],
    loops=LoopConfig(
        enabled=True,
        locations=[
            LoopLocationConfig(kind=LoopKind.INTRO, enabled=True),
            LoopLocationConfig(kind=LoopKind.OUTRO, enabled=True),
            LoopLocationConfig(kind=LoopKind.BREAKDOWN, enabled=False),
            LoopLocationConfig(kind=LoopKind.PRE_DROP, enabled=True),
            LoopLocationConfig(kind=LoopKind.VOCAL, enabled=False),
        ],
        length=LoopLength.BARS_8,
        color=RGB.from_hex("#14B8A6"),
        selection=LoopSelection.FIRST,
    ),
)
