"""Cue candidate generation.

Each detector answers one question -- "where would a DJ put a drop cue?" -- and
returns candidates carrying the *evidence* that produced them, not a score. Turning
evidence into a confidence number happens in `scoring.py`, so that the scoring
model can be refit from user feedback without touching detection logic.

The governing principle throughout: a recommendation must be derived from musical
structure, never from arithmetic on the timeline. Nothing here says "32 bars in".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from djprep.models.analysis import SectionLabel, TrackAnalysis
from djprep.models.prep import CueKind


@dataclass
class Candidate:
    time_sec: float
    kind: CueKind
    evidence: dict[str, float] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    section_index: int | None = None


def _rel(a: float, b: float) -> float:
    """Signed difference, clipped to [-1, 1]. Used for percentile jumps."""
    return float(np.clip(a - b, -1.0, 1.0))


def _grid_evidence(t: float, an: TrackAnalysis) -> dict[str, float]:
    g, p = an.beat_grid, an.phrase_grid
    bar = g.bar_index(t)
    on_down = abs(t - g.snap_to_downbeat(t)) < 0.045
    on_phrase = bool(p.boundaries_sec) and min(
        abs(np.asarray(p.boundaries_sec) - t)) < 0.045
    return {
        "on_downbeat": 1.0 if on_down else 0.0,
        "on_phrase": 1.0 if on_phrase else 0.0,
        "grid_confidence": g.confidence,
        "downbeat_confidence": g.downbeat_confidence,
        "phrase_confidence": p.confidence,
        "bar_index": float(bar),
    }


def first_content_time(an: TrackAnalysis) -> float:
    """Where the music actually starts, skipping leading silence or noise.

    First frame whose loudness rises meaningfully above the track's own quiet
    floor. Used to anchor the initial cue: loading a track at 0:00 when there is
    a second of room tone before the first kick wastes a bar every time.
    """
    times = an.energy.times_sec
    rms = an.energy.rms_db
    if not times or not rms:
        return 0.0
    arr = np.asarray(rms, dtype=float)
    floor = float(np.percentile(arr, 5))
    ceil = float(np.percentile(arr, 95))
    if ceil - floor < 3.0:
        return 0.0
    thresh = floor + 0.25 * (ceil - floor)
    above = np.flatnonzero(arr >= thresh)
    return float(times[int(above[0])]) if above.size else 0.0


def detect_initial(an: TrackAnalysis) -> list[Candidate]:
    """The load point. Emitted for every track, unconditionally.

    Every other detector can legitimately find nothing -- a track may have no
    breakdown, no vocals, no tempo change. This one always fires, because every
    track has a place you load it from, and a DJ opening a prepared track expects
    a cue waiting on the first beat rather than an empty deck.
    """
    t0 = first_content_time(an)
    g = an.beat_grid
    # Snap forward to a downbeat so the load point is musically usable.
    t = g.snap_to_downbeat(t0)
    if t < t0 - 0.02:
        t = min(t + g.bar_period, an.meta.duration_sec)
    t = max(0.0, t)

    ev = _grid_evidence(t, an)
    ev.update({"position": 0.0, "is_track_start": 1.0})
    lead_in = t0
    notes = {}
    if lead_in > 0.35:
        notes["lead_in"] = f"{lead_in:.1f}s of silence before the music starts"
    return [Candidate(t, CueKind.INITIAL, ev, notes)]


def detect_tempo_changes(an: TrackAnalysis) -> list[Candidate]:
    """A cue at each detected tempo change.

    Worth marking even when you do not intend to mix there: it is the point past
    which the constant-tempo beat grid stops being trustworthy.
    """
    out = []
    for tc in an.tempo_changes:
        ev = _grid_evidence(tc.time_sec, an)
        ev.update({
            "tempo_change_confidence": tc.confidence,
            "tempo_delta": float(np.clip(abs(tc.ratio - 1.0) * 10, 0, 1)),
        })
        kind_note = ("half/double-time feel change" if tc.is_octave
                     else f"tempo {tc.bpm_before:.1f} → {tc.bpm_after:.1f} BPM")
        out.append(Candidate(tc.time_sec, CueKind.TEMPO_CHANGE, ev,
                             {"tempo": kind_note}))
    return out


def detect_drops(an: TrackAnalysis) -> list[Candidate]:
    """A drop is where the track's energy resolves, not merely where it is loud."""
    out = []
    for i, s in enumerate(an.sections):
        if s.label is not SectionLabel.DROP:
            continue
        prev = an.sections[i - 1] if i > 0 else None
        ev = _grid_evidence(s.start_sec, an)
        ev.update({
            "energy_pct": s.energy_pct,
            "low_energy_pct": s.low_energy_pct,
            "onset_density_pct": s.onset_density_pct,
            "energy_jump": _rel(s.energy_pct, prev.energy_pct) if prev else 0.0,
            "low_jump": _rel(s.low_energy_pct, prev.low_energy_pct) if prev else 0.0,
            "density_jump": _rel(s.onset_density_pct, prev.onset_density_pct) if prev else 0.0,
            "preceded_by_buildup": 1.0 if prev and prev.label is SectionLabel.BUILDUP else 0.0,
            "buildup_bars": prev.length_bars if prev and prev.label is SectionLabel.BUILDUP else 0.0,
            "section_confidence": s.confidence,
        })
        notes = {}
        if prev and prev.label is SectionLabel.BUILDUP:
            notes["buildup_bars"] = f"{prev.length_bars:.0f}-bar build-up immediately before"
        out.append(Candidate(s.start_sec, CueKind.DROP, ev, notes, i))
    return out


def detect_pre_drop(an: TrackAnalysis) -> list[Candidate]:
    """The moment before the drop lands -- where you would cut, filter or throw FX.

    Emitted *at* the drop. Moving it earlier is not this function's job: the
    engine applies each cue type's configured `offset_bars` to every candidate
    uniformly, so doing it here as well shifts the cue twice. (That is not
    hypothetical -- an earlier version did exactly this and produced pre-drop
    cues 8 bars early when the user had asked for 4.) One mechanism, applied in
    one place.
    """
    out = []
    for c in detect_drops(an):
        ev = _grid_evidence(c.time_sec, an)
        ev.update({k: v for k, v in c.evidence.items()
                   if k in ("energy_jump", "low_jump", "preceded_by_buildup",
                            "buildup_bars", "section_confidence")})
        out.append(Candidate(c.time_sec, CueKind.PRE_DROP, ev, {}, c.section_index))
    return out


def detect_buildups(an: TrackAnalysis) -> list[Candidate]:
    out = []
    for i, s in enumerate(an.sections):
        if s.label is not SectionLabel.BUILDUP:
            continue
        nxt = an.sections[i + 1] if i + 1 < len(an.sections) else None
        ev = _grid_evidence(s.start_sec, an)
        ev.update({
            "energy_slope": float(np.clip(s.energy_slope / 2.0, -1, 1)),
            "high_energy_pct": s.high_energy_pct,
            "onset_density_pct": s.onset_density_pct,
            "resolves_to_drop": 1.0 if nxt and nxt.label in
                                (SectionLabel.DROP, SectionLabel.CHORUS) else 0.0,
            "next_energy_jump": _rel(nxt.energy_pct, s.energy_pct) if nxt else 0.0,
            "section_confidence": s.confidence,
            "length_bars": s.length_bars,
        })
        out.append(Candidate(s.start_sec, CueKind.BUILDUP, ev, {}, i))
    return out


def detect_breakdowns(an: TrackAnalysis) -> list[Candidate]:
    out = []
    for i, s in enumerate(an.sections):
        if s.label is not SectionLabel.BREAKDOWN:
            continue
        prev = an.sections[i - 1] if i > 0 else None
        ev = _grid_evidence(s.start_sec, an)
        ev.update({
            "low_energy_pct": s.low_energy_pct,
            "energy_pct": s.energy_pct,
            "bass_drop_out": _rel(prev.low_energy_pct, s.low_energy_pct) if prev else 0.0,
            "density_drop": _rel(prev.onset_density_pct, s.onset_density_pct) if prev else 0.0,
            "section_confidence": s.confidence,
            "length_bars": s.length_bars,
        })
        out.append(Candidate(s.start_sec, CueKind.BREAKDOWN, ev, {}, i))
    return out


def detect_chorus(an: TrackAnalysis) -> list[Candidate]:
    out = []
    for i, s in enumerate(an.sections):
        if s.label is not SectionLabel.CHORUS:
            continue
        ev = _grid_evidence(s.start_sec, an)
        ev.update({"energy_pct": s.energy_pct, "vocal_ratio": s.vocal_ratio,
                   "section_confidence": s.confidence})
        out.append(Candidate(s.start_sec, CueKind.CHORUS, ev, {}, i))
    return out


def detect_intro_outro(an: TrackAnalysis) -> list[Candidate]:
    out = []
    for i, s in enumerate(an.sections):
        if s.label is SectionLabel.INTRO:
            ev = _grid_evidence(s.start_sec, an)
            ev.update({"energy_pct": s.energy_pct, "position": 0.0,
                       "section_confidence": s.confidence, "length_bars": s.length_bars})
            out.append(Candidate(s.start_sec, CueKind.INTRO, ev, {}, i))
        elif s.label is SectionLabel.OUTRO:
            ev = _grid_evidence(s.start_sec, an)
            ev.update({"energy_pct": s.energy_pct,
                       "position": s.start_sec / max(an.meta.duration_sec, 1e-6),
                       "section_confidence": s.confidence, "length_bars": s.length_bars})
            out.append(Candidate(s.start_sec, CueKind.OUTRO, ev, {}, i))
    return out


def _mix_window_score(an: TrackAnalysis, s) -> dict[str, float]:
    """How good is this section as material to blend over?"""
    return {
        "has_drums": float(np.clip(s.onset_density_pct * 1.4, 0, 1)),
        "stable_energy": float(np.clip(1.0 - abs(s.energy_slope) / 1.5, 0, 1)),
        "no_vocals": float(np.clip(1.0 - s.vocal_ratio * 2.0, 0, 1)),
        "length_bars": s.length_bars,
        "long_enough": float(np.clip(s.length_bars / 16.0, 0, 1)),
        "section_confidence": s.confidence,
    }


def detect_mix_points(an: TrackAnalysis) -> list[Candidate]:
    """Mix-in and mix-out, anchored to section boundaries.

    The anchors are the two boundaries a DJ actually blends across:

      mix-in  -> the moment the intro gives way to the body of the track. The
                 cue is placed N bars *before* it, so those N bars of stable
                 intro are the window you beatmatch over.
      mix-out -> the moment the outro begins. The cue is placed N bars *after*
                 it, so the outro has settled before you start pulling away.

    N is the user's choice (4/8/16/32) and is applied by the engine, not here --
    this function emits the anchor only. Emitting the offset position as well
    would apply it twice.

    Falling back sensibly matters: a track with no detected intro still has a
    first section, and its end is still the point the arrangement opens up.
    """
    out: list[Candidate] = []
    if not an.sections:
        return out

    intro = next((s for s in an.sections if s.label is SectionLabel.INTRO), None)
    anchor_in = intro or an.sections[0]
    # The boundary is where this section *ends* -- that is what we mix across.
    t_in = anchor_in.end_sec
    if t_in > 0.5:
        ev = _grid_evidence(t_in, an)
        ev.update(_mix_window_score(an, anchor_in))
        ev["position"] = t_in / max(an.meta.duration_sec, 1e-6)
        out.append(Candidate(t_in, CueKind.MIX_IN, ev, {
            "anchor": f"end of the {anchor_in.length_bars:.0f}-bar "
                      f"{anchor_in.label.value}",
        }))

    outro = next((s for s in reversed(an.sections)
                  if s.label is SectionLabel.OUTRO), None)
    anchor_out = outro or an.sections[-1]
    t_out = anchor_out.start_sec
    if t_out < an.meta.duration_sec - 0.5:
        ev = _grid_evidence(t_out, an)
        ev.update(_mix_window_score(an, anchor_out))
        ev["position"] = t_out / max(an.meta.duration_sec, 1e-6)
        out.append(Candidate(t_out, CueKind.MIX_OUT, ev, {
            "anchor": f"start of the {anchor_out.length_bars:.0f}-bar "
                      f"{anchor_out.label.value}",
        }))
    return out


def detect_vocal_cues(an: TrackAnalysis) -> list[Candidate]:
    out = []
    dur = max(an.meta.duration_sec, 1e-6)
    for v in an.vocals:
        sec = an.section_at(v.start_sec)
        ev = _grid_evidence(v.start_sec, an)
        ev.update({
            "vocal_confidence": v.confidence,
            "vocal_length": min((v.end_sec - v.start_sec) / 30.0, 1.0),
            "position": v.start_sec / dur,
            "in_labelled_vocal_section": 1.0 if sec and sec.label in
                (SectionLabel.CHORUS, SectionLabel.VERSE) else 0.0,
        })
        out.append(Candidate(v.start_sec, CueKind.VOCAL_IN, ev, {}))

        ev2 = _grid_evidence(v.end_sec, an)
        ev2.update({"vocal_confidence": v.confidence,
                    "vocal_length": min((v.end_sec - v.start_sec) / 30.0, 1.0),
                    "position": v.end_sec / dur})
        out.append(Candidate(v.end_sec, CueKind.VOCAL_OUT, ev2, {}))
    return out


def detect_fx_points(an: TrackAnalysis) -> list[Candidate]:
    """Phrase endings worth reaching for an effect on.

    Restricted to the last bar of a phrase that immediately precedes a real
    arrangement change -- those are the moments where a delay throw or filter
    sweep does something. Every phrase ending would be noise.
    """
    out: list[Candidate] = []
    g = an.beat_grid
    for i in range(1, len(an.sections)):
        prev, cur = an.sections[i - 1], an.sections[i]
        t = cur.start_sec - g.bar_period
        if t <= 0.5:
            continue
        ev = _grid_evidence(t, an)
        ev.update({
            "transition_size": abs(_rel(cur.energy_pct, prev.energy_pct)),
            "into_drop": 1.0 if cur.label is SectionLabel.DROP else 0.0,
            "section_confidence": min(prev.confidence, cur.confidence),
        })
        out.append(Candidate(t, CueKind.FX, ev,
                             {"target": f"last bar before the {cur.label.value}"}, i))
    return out


DETECTORS = {
    CueKind.DROP: detect_drops,
    CueKind.BUILDUP: detect_buildups,
    CueKind.BREAKDOWN: detect_breakdowns,
    CueKind.CHORUS: detect_chorus,
    CueKind.FX: detect_fx_points,
}
