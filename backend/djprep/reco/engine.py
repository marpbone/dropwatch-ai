"""The recommendation engine: analysis + user config -> a reviewable preparation.

Pipeline per cue type:
    detect candidates -> snap to the grid -> score -> filter -> suppress
    neighbours -> apply per-type and global budgets

The ordering matters. Snapping happens *before* scoring so that "landed on a
downbeat" is measured on the position we will actually use, not the raw
detection. Non-maximum suppression happens *after* scoring so that when two cues
compete for the same bar the more confident one survives.
"""
from __future__ import annotations

from djprep.audio.features import Features
from djprep.audio.phrases import snap_to_phrase
from djprep.models.analysis import TrackAnalysis
from djprep.models.config import AnalysisConfig, LoopSelection
from djprep.models.prep import CuePoint, Loop, Source, TrackPreparation
from djprep.reco import candidates as C
from djprep.reco import loops as LP
from djprep.reco import scoring

LABEL_TEXT = {
    "initial": "Start", "tempo_change": "Tempo",
    "intro": "Intro", "mix_in": "Mix In", "buildup": "Build-up",
    "pre_drop": "Pre-Drop", "drop": "Drop", "breakdown": "Breakdown",
    "chorus": "Chorus", "vocal_in": "Vocal In", "vocal_out": "Vocal Out",
    "mix_out": "Mix Out", "outro": "Outro", "fx": "FX", "custom": "Cue",
}


def _snap(t: float, an: TrackAnalysis, mode: str) -> float:
    g = an.beat_grid
    if mode == "beat":
        return g.snap_to_beat(t)
    if mode == "phrase":
        return snap_to_phrase(t, g, an.phrase_grid)
    return g.snap_to_downbeat(t)


def _collect(an: TrackAnalysis, cfg: AnalysisConfig) -> list[C.Candidate]:
    out: list[C.Candidate] = []
    kinds = {c.kind for c in cfg.enabled_cues()}
    from djprep.models.prep import CueKind as K

    if K.INITIAL in kinds:
        out += C.detect_initial(an)
    if K.TEMPO_CHANGE in kinds:
        out += C.detect_tempo_changes(an)
    if K.DROP in kinds:
        out += C.detect_drops(an)
    if K.PRE_DROP in kinds:
        out += C.detect_pre_drop(an)
    if K.BUILDUP in kinds:
        out += C.detect_buildups(an)
    if K.BREAKDOWN in kinds:
        out += C.detect_breakdowns(an)
    if K.CHORUS in kinds:
        out += C.detect_chorus(an)
    if kinds & {K.INTRO, K.OUTRO}:
        out += [c for c in C.detect_intro_outro(an) if c.kind in kinds]
    if kinds & {K.MIX_IN, K.MIX_OUT}:
        out += [c for c in C.detect_mix_points(an) if c.kind in kinds]
    if kinds & {K.VOCAL_IN, K.VOCAL_OUT}:
        out += [c for c in C.detect_vocal_cues(an) if c.kind in kinds]
    if K.FX in kinds:
        out += C.detect_fx_points(an)
    return out


def recommend(f: Features, an: TrackAnalysis, cfg: AnalysisConfig) -> TrackPreparation:
    prep = TrackPreparation(track_id=an.meta.track_id)
    g = an.beat_grid

    from djprep.models.prep import CueKind as K

    # Mix cues take their offset from the mix panel rather than the generic
    # per-type offset: mix-in sits N bars *before* its anchor, mix-out N bars
    # *after* its own. Everything else uses its cue-type offset. As always the
    # offset is applied exactly once, here, never in the detector.
    offsets: dict[object, float] = {
        K.MIX_IN: -cfg.mix.mix_in_bars.bars(),
        K.MIX_OUT: +cfg.mix.mix_out_bars.bars(),
    }
    start_cue_t = g.snap_to_downbeat(C.first_content_time(an))

    scored: list[tuple[float, CuePoint]] = []
    for cand in _collect(an, cfg):
        tc = cfg.cue_config(cand.kind)
        if tc is None or not tc.enabled:
            continue
        offset_bars = offsets.get(cand.kind, tc.offset_bars)
        t = cand.time_sec + offset_bars * g.bar_period

        # Clamp mix cues into usable territory. Asking for a 16-bar mix-in on a
        # 16-bar intro is a perfectly reasonable request that lands exactly on
        # the track start, and asking for a 16-bar mix-out on a short outro puts
        # the cue past the end of the file. Neither should silently produce a
        # useless marker, and neither should be refused -- so the offset is
        # reduced to what the track can actually give and the compromise is
        # recorded in the cue's reasons.
        # The clamp has to be applied *after* snapping, not before: snapping to
        # the nearest phrase boundary can walk the cue straight back across a
        # limit it was just moved inside. So snap first, then step forward (or
        # back) a bar at a time until the constraint holds and the cue is still
        # on the grid.
        requested = t
        t = _snap(t, an, tc.snap)
        clamped_from: float | None = None

        if cand.kind is K.MIX_IN:
            floor_t = start_cue_t + cfg.min_cue_spacing_bars * g.bar_period
            if t < floor_t - 1e-6:
                clamped_from = requested
                while t < floor_t - 1e-6 and t < an.meta.duration_sec:
                    t += g.bar_period
        elif cand.kind is K.MIX_OUT:
            ceil_t = an.meta.duration_sec - 4.0 * g.bar_period
            if t > ceil_t + 1e-6:
                clamped_from = requested
                while t > ceil_t + 1e-6 and t > 0:
                    t -= g.bar_period
        if t < 0 or t > an.meta.duration_sec:
            continue

        # Re-measure grid alignment at the final position: a cue's claim to be
        # "on a phrase boundary" must describe where it actually ended up.
        ev = dict(cand.evidence)
        ev.update(C._grid_evidence(t, an))
        notes = dict(cand.notes)
        if clamped_from is not None:
            actual = abs(t - cand.time_sec) / g.bar_period
            notes["clamped"] = (f"Shortened to {actual:.0f} bars — the section is "
                                f"not long enough for {abs(offset_bars):.0f}")
        elif offset_bars:
            ev["bars_before_drop"] = abs(offset_bars)
            where = "before" if offset_bars < 0 else "after"
            anchor = notes.get("anchor", f"detected {cand.kind.value.replace('_', ' ')}")
            notes["offset"] = f"Placed {abs(offset_bars):.0f} bars {where} the {anchor}"
        s = scoring.score(cand.kind, ev, notes)
        if s.confidence < tc.min_confidence:
            continue

        bar = g.bar_index(t)
        cue = CuePoint(
            time_sec=round(t, 4), kind=cand.kind,
            label=tc.label_template or LABEL_TEXT.get(cand.kind.value, cand.kind.value),
            color=tc.color, source=Source.AI, confidence=s.confidence,
            reasons=s.reasons, beat_index=g.nearest_beat_index(t), bar_index=bar,
            on_downbeat=bool(ev.get("on_downbeat")), on_phrase=bool(ev.get("on_phrase")),
            original_time_sec=round(t, 4),
        )
        scored.append((s.confidence, cue))

    # Per-type budget. `select_all` keeps every occurrence rather than only the
    # most confident: a track with three drops needs three drop cues, because the
    # DJ needs all of them. `max_count` then acts purely as a safety ceiling.
    kept: list[CuePoint] = []
    by_kind: dict[str, list[CuePoint]] = {}
    for _conf, cue in sorted(scored, key=lambda x: -x[0]):
        by_kind.setdefault(cue.kind.value, []).append(cue)
    for group in by_kind.values():
        tc = cfg.cue_config(group[0].kind)
        limit = (tc.max_count if tc else 1)
        if tc and tc.select_all:
            limit = max(limit, tc.max_count)
        kept += group[:limit]

    # Non-maximum suppression across types: two markers half a bar apart are
    # clutter on a CDJ screen, whatever their labels. The initial cue is exempt
    # and placed first -- it is the load point, it is guaranteed for every track,
    # and it must never lose a tie-break to something that happens to be nearby.
    min_gap = cfg.min_cue_spacing_bars * g.bar_period
    initial = [c for c in kept if c.kind is K.INITIAL]
    rest = sorted((c for c in kept if c.kind is not K.INITIAL),
                  key=lambda c: -(c.confidence or 0.0))
    final: list[CuePoint] = list(initial[:1])
    for c in rest:
        if all(abs(c.time_sec - o.time_sec) >= min_gap for o in final):
            final.append(c)
        if len(final) >= cfg.max_total_cues:
            break

    final.sort(key=lambda c: c.time_sec)
    prep.cues = final
    # Priority for slot-limited exporters: confidence first. The user can
    # reorder this in the UI, which is the only thing that decides who gets a
    # hot cue when there are more cues than slots.
    # Priority for slot-limited exporters. The initial cue always ranks first --
    # if only one hot cue slot survives the lowering pass, it should be the load
    # point. Everything else ranks by confidence, and the user can reorder.
    prep.cue_priority = [c.id for c in sorted(
        final, key=lambda c: (c.kind is not K.INITIAL, -(c.confidence or 0)))]

    prep.loops = _build_loops(f, an, cfg)
    return prep


def _build_loops(f: Features, an: TrackAnalysis, cfg: AnalysisConfig) -> list[Loop]:
    lc = cfg.loops
    if not lc.enabled:
        return []
    cands = LP.generate(f, an, lc)
    out: list[Loop] = []
    by_kind: dict[str, list[tuple[float, Loop]]] = {}

    for c in cands:
        ev = dict(c.evidence)
        if lc.avoid_vocal_cuts and ev.get("vocal_cut", 0.0) >= 1.0:
            continue
        s = scoring.score("loop", ev, c.notes)
        if s.confidence < lc.min_confidence:
            continue
        loop = Loop(
            start_sec=c.start_sec, end_sec=c.end_sec, length_bars=c.length_bars,
            kind=c.kind, label=f"{c.kind.value.replace('_', ' ').title()} Loop",
            color=lc.color, source=Source.AI, confidence=s.confidence,
            reasons=s.reasons,
            start_beat_index=an.beat_grid.nearest_beat_index(c.start_sec),
            start_bar_index=an.beat_grid.bar_index(c.start_sec),
            original_time_sec=c.start_sec,
        )
        by_kind.setdefault(c.kind.value, []).append((s.confidence, loop))

    for group in by_kind.values():
        group.sort(key=lambda x: -x[0])
        if lc.selection is LoopSelection.FIRST:
            chosen = [min(group, key=lambda x: x[1].start_sec)[1]]
        elif lc.selection is LoopSelection.TOP_N:
            chosen = [lp for _, lp in group[: lc.top_n]]
        else:
            chosen = [lp for _, lp in group]
        # Never return overlapping loops of the same kind.
        chosen.sort(key=lambda lp: lp.start_sec)
        acc: list[Loop] = []
        for lp in chosen:
            if all(lp.start_sec >= o.end_sec or lp.end_sec <= o.start_sec for o in acc):
                acc.append(lp)
        out += acc

    out.sort(key=lambda lp: lp.start_sec)
    return out
