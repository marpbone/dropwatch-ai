"""What the DSP pipeline learns about a track.

This is the *observation* layer: purely descriptive, no recommendations. The
recommendation engine (`djprep.reco`) reads a TrackAnalysis and writes a
TrackPreparation. Keeping them separate means you can re-run recommendations
with different user config without re-running the (expensive) DSP.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TrackMeta(BaseModel):
    track_id: str
    filename: str
    duration_sec: float
    sample_rate: int
    channels: int
    title: str | None = None
    artist: str | None = None


class BeatGrid(BaseModel):
    """A constant-tempo beat grid.

    Design note: we deliberately model the grid as (bpm, anchor, downbeat_offset)
    rather than as a list of arbitrary beat times. Nearly all club-oriented music
    is machine-quantised to a fixed tempo, DJ software represents grids this way
    (Rekordbox's TEMPO element is exactly `Inizio` + `Bpm`), and a parametric grid
    is far more robust than per-beat tracking: a single misdetected beat cannot
    corrupt the rest of the track. `beat_times()` regenerates positions on demand.

    For genuinely variable-tempo material `is_constant` is False and `segments`
    holds a piecewise-constant approximation.
    """

    bpm: float
    anchor_sec: float = Field(description="Time of beat 0; the grid phase.")
    beats_per_bar: int = 4
    downbeat_offset: int = Field(
        0, ge=0, description="Which beat index (mod beats_per_bar) is a downbeat."
    )
    n_beats: int
    is_constant: bool = True
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    downbeat_confidence: float = Field(0.0, ge=0.0, le=1.0)

    @property
    def beat_period(self) -> float:
        return 60.0 / self.bpm

    @property
    def bar_period(self) -> float:
        return self.beat_period * self.beats_per_bar

    def beat_time(self, i: int) -> float:
        return self.anchor_sec + i * self.beat_period

    def beat_times(self) -> list[float]:
        return [self.beat_time(i) for i in range(self.n_beats)]

    def downbeat_indices(self) -> list[int]:
        return list(range(self.downbeat_offset, self.n_beats, self.beats_per_bar))

    def downbeat_times(self) -> list[float]:
        return [self.beat_time(i) for i in self.downbeat_indices()]

    def nearest_beat_index(self, t: float) -> int:
        i = round((t - self.anchor_sec) / self.beat_period)
        return max(0, min(self.n_beats - 1, i))

    def snap_to_beat(self, t: float) -> float:
        return self.beat_time(self.nearest_beat_index(t))

    def snap_to_downbeat(self, t: float) -> float:
        """Snap to the nearest bar line."""
        rel = (t - self.anchor_sec) / self.beat_period - self.downbeat_offset
        bar = round(rel / self.beats_per_bar)
        idx = self.downbeat_offset + int(bar) * self.beats_per_bar
        idx = max(self.downbeat_offset, min(self.n_beats - 1, idx))
        return self.beat_time(idx)

    def bar_index(self, t: float) -> int:
        rel = (t - self.anchor_sec) / self.beat_period - self.downbeat_offset
        return round(rel / self.beats_per_bar)

    def bars_to_seconds(self, bars: float) -> float:
        return bars * self.bar_period


class PhraseGrid(BaseModel):
    """Phrase-level (multi-bar) structure.

    Dance music is built in power-of-two bar groups. Knowing the phrase length and
    phase is what separates "a marker every 32 bars" from "a marker where the
    music actually turns over".
    """

    phrase_bars: int = 8
    phase_bars: int = Field(0, description="Bar index of the first phrase boundary.")
    boundaries_sec: list[float] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    alt_lengths: dict[str, float] = Field(
        default_factory=dict,
        description="Contrast score for each candidate phrase length; exposes the "
                    "metrical hierarchy so the UI can offer 8 vs 16 as a choice.")

    def is_phrase_boundary(self, bar_index: int) -> bool:
        return (bar_index - self.phase_bars) % self.phrase_bars == 0


class SectionLabel(str, Enum):
    """DJ-meaningful section taxonomy.

    Chosen to match how DJs actually talk about arrangement, not musicology.
    `UNKNOWN` is a legitimate output -- forcing a label on an ambiguous segment is
    worse than admitting uncertainty.
    """

    INTRO = "intro"
    VERSE = "verse"
    BUILDUP = "buildup"
    DROP = "drop"
    CHORUS = "chorus"
    BREAKDOWN = "breakdown"
    BRIDGE = "bridge"
    OUTRO = "outro"
    UNKNOWN = "unknown"


class TempoChange(BaseModel):
    """A point where the track's tempo demonstrably changes.

    Detected independently of the main grid fit, which assumes a constant tempo.
    Reporting these separately is more honest than silently averaging them away:
    a track with a tempo change has a grid that is correct on one side of the
    change and drifting on the other, and the DJ needs to know where.
    """

    time_sec: float
    bpm_before: float
    bpm_after: float
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    @property
    def ratio(self) -> float:
        return self.bpm_after / max(self.bpm_before, 1e-6)

    @property
    def is_octave(self) -> bool:
        """Half/double-time feel rather than a genuine tempo change."""
        r = self.ratio
        return any(abs(r - m) < 0.04 for m in (0.5, 2.0))


class Section(BaseModel):
    start_sec: float
    end_sec: float
    start_bar: int
    length_bars: float
    label: SectionLabel
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    cluster_id: int = -1
    # Interpretable per-section statistics, all normalised within the track so
    # they are comparable across tracks of different loudness/mastering.
    energy_pct: float = 0.0
    low_energy_pct: float = 0.0
    high_energy_pct: float = 0.0
    onset_density_pct: float = 0.0
    brightness_pct: float = 0.0
    energy_slope: float = 0.0
    vocal_ratio: float = 0.0
    label_scores: dict[str, float] = Field(default_factory=dict)

    # Set when the top two labels scored close enough that the choice between
    # them is not trustworthy -- most often drop vs chorus, which in a vocal
    # house track are genuinely the same moment under two names. Surfaced in the
    # UI as a prompt to decide by ear rather than hidden behind a number.
    ambiguous_with: str | None = None
    ambiguity_margin: float | None = None
    # True once the user has relabelled or moved this section by hand. Manual
    # sections are never overwritten by a re-analysis.
    is_manual: bool = False
    # Repeat index among sections sharing this label: the second drop is 2.
    occurrence: int = 1

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


class VocalSegment(BaseModel):
    start_sec: float
    end_sec: float
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class EnergyCurve(BaseModel):
    """Frame-level curves, decimated for transport to the frontend."""

    times_sec: list[float]
    rms_db: list[float]
    low: list[float]
    mid: list[float]
    high: list[float]
    onset_density: list[float]
    vocal_likelihood: list[float] = Field(default_factory=list)


class LabelSummary(BaseModel):
    """Which section types this track actually contains.

    Not every song has every section. A track with no vocals has no chorus; a
    techno tool has no verse. Reporting presence explicitly lets the UI hide
    controls that cannot apply to the loaded track instead of offering the user
    a checkbox that can never produce anything.
    """

    present: list[str] = Field(default_factory=list)
    absent: list[str] = Field(default_factory=list)
    suppressed: list[str] = Field(
        default_factory=list,
        description="Labels detected but withheld because confidence was below "
                    "the floor -- typically verse/chorus/bridge on instrumental "
                    "tracks, where the distinction is guesswork.")
    counts: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class TrackAnalysis(BaseModel):
    meta: TrackMeta
    beat_grid: BeatGrid
    phrase_grid: PhraseGrid
    sections: list[Section]
    tempo_changes: list[TempoChange] = Field(default_factory=list)
    label_summary: LabelSummary = Field(default_factory=LabelSummary)
    vocals: list[VocalSegment]
    energy: EnergyCurve
    key: str | None = None
    analysis_version: str = "1"
    timings_ms: dict[str, float] = Field(default_factory=dict)

    def section_at(self, t: float) -> Section | None:
        """The section covering `t`.

        Sections are stored as independent intervals rather than a strict
        partition, so that a hand-edited section may overlap its neighbour. When
        several cover the same instant the shortest wins, which matches what a
        user means by dragging a small section on top of a large one.
        """
        hits = [s for s in self.sections if s.start_sec <= t < s.end_sec]
        if not hits:
            return None
        return min(hits, key=lambda s: s.end_sec - s.start_sec)

    def sections_at(self, t: float) -> list[Section]:
        return [s for s in self.sections if s.start_sec <= t < s.end_sec]

    def sections_with(self, label: SectionLabel) -> list[Section]:
        return [s for s in self.sections if s.label == label]
