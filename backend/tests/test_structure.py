"""Segmentation and labelling, scored the way the literature scores them."""
import numpy as np
import pytest

from djprep.models.analysis import SectionLabel


def boundary_f1(pred: list[float], ref: list[float], tol: float) -> tuple[float, float, float]:
    """Standard MIREX-style boundary retrieval score with a tolerance window."""
    matched_r, matched_p = set(), set()
    for i, p in enumerate(pred):
        for j, r in enumerate(ref):
            if j not in matched_r and abs(p - r) <= tol:
                matched_r.add(j); matched_p.add(i); break
    prec = len(matched_p) / max(len(pred), 1)
    rec = len(matched_r) / max(len(ref), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1


def test_boundaries_match_ground_truth(analysis, truth):
    ref = [s["start_sec"] for s in truth["sections"]][1:]      # skip t=0
    pred = [s.start_sec for s in analysis.sections][1:]
    tol = truth["bar_sec"] * 1.0                                # within one bar
    prec, rec, f1 = boundary_f1(pred, ref, tol)
    assert f1 >= 0.9, f"boundary F1 {f1:.2f} (P {prec:.2f} R {rec:.2f})"


def test_every_boundary_sits_on_the_phrase_grid(analysis):
    """A section that starts mid-phrase is a detection error, not a discovery."""
    b = np.asarray(analysis.phrase_grid.boundaries_sec)
    for s in analysis.sections[1:]:
        assert float(np.min(np.abs(b - s.start_sec))) < 0.06, \
            f"section at {s.start_sec:.2f}s is off the phrase grid"


def test_frame_level_label_accuracy(analysis, truth):
    """Fraction of the track's duration given the right label."""
    step = 0.25
    t = np.arange(0, truth["duration_sec"] - 1, step)
    def ref_at(x):
        for s in truth["sections"]:
            if s["start_sec"] <= x < s["end_sec"]:
                return s["label"]
        return truth["sections"][-1]["label"]
    correct = 0
    for x in t:
        sec = analysis.section_at(float(x))
        if sec and sec.label.value == ref_at(float(x)):
            correct += 1
    acc = correct / len(t)
    assert acc >= 0.85, f"frame label accuracy {acc:.2%}"


def test_expected_labels_are_all_present(analysis):
    labels = {s.label for s in analysis.sections}
    for required in (SectionLabel.INTRO, SectionLabel.BUILDUP,
                     SectionLabel.DROP, SectionLabel.BREAKDOWN, SectionLabel.OUTRO):
        assert required in labels, f"never found a {required.value}"


def test_structural_grammar_is_respected(analysis):
    """The HMM exists to prevent nonsense orderings; check it did its job."""
    seq = [s.label for s in analysis.sections]
    assert seq[0] is SectionLabel.INTRO
    assert seq[-1] is SectionLabel.OUTRO
    assert seq.count(SectionLabel.INTRO) == 1
    # Every drop should be reachable from a buildup or breakdown, never straight
    # out of an intro with nothing in between.
    for i, lab in enumerate(seq):
        if lab is SectionLabel.DROP and i > 0:
            assert seq[i - 1] in (SectionLabel.BUILDUP, SectionLabel.BREAKDOWN,
                                  SectionLabel.DROP, SectionLabel.CHORUS)


def test_section_statistics_are_percentiles(analysis):
    for s in analysis.sections:
        for field in ("energy_pct", "low_energy_pct", "onset_density_pct", "brightness_pct"):
            assert 0.0 <= getattr(s, field) <= 1.0


def test_breakdown_has_less_low_end_than_drops(analysis):
    drops = [s for s in analysis.sections if s.label is SectionLabel.DROP]
    breaks = [s for s in analysis.sections if s.label is SectionLabel.BREAKDOWN]
    assert drops and breaks
    assert min(d.low_energy_pct for d in drops) > max(b.low_energy_pct for b in breaks)


def test_vocal_detection_f1(analysis, truth, feats):
    ref = np.zeros(feats.n_frames, dtype=bool)
    for r in truth["vocal_regions"]:
        ref[feats.frame_at(r["start_sec"]):feats.frame_at(r["end_sec"])] = True
    pred = np.zeros(feats.n_frames, dtype=bool)
    for v in analysis.vocals:
        pred[feats.frame_at(v.start_sec):feats.frame_at(v.end_sec)] = True
    tp = int((ref & pred).sum()); fp = int((~ref & pred).sum()); fn = int((ref & ~pred).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    assert f1 >= 0.75, f"vocal F1 {f1:.2f}"


def test_analysis_reports_its_own_timings(analysis):
    assert {"features", "beats", "phrases"} <= set(analysis.timings_ms)
    assert all(v >= 0 for v in analysis.timings_ms.values())
