"""Lossless JSON export of the internal model.

The reference target: everything the application knows, in the application's own
vocabulary, including the confidence scores and explanations that no DJ software
format has anywhere to put. Useful for round-tripping a preparation between
machines, for diffing two analyses, and as the ground truth a lowering report is
implicitly measured against.
"""
from __future__ import annotations

import json

from djprep.export.base import ColorMode, ExporterCapabilities, LoweringReport, Severity
from djprep.models.analysis import TrackAnalysis
from djprep.models.prep import TrackPreparation

CAPS = ExporterCapabilities(
    format_id="djprep_json",
    display_name="djprep JSON (lossless)",
    file_extension=".json",
    max_hot_cues=None,
    color_mode=ColorMode.FREE_RGB,
    notes="Complete internal representation. Nothing is lost.",
)


class JsonExporter:
    caps = CAPS

    def export(self, prep: TrackPreparation, analysis: TrackAnalysis,
               audio_path: str, include_analysis: bool = True, **options
               ) -> tuple[bytes, LoweringReport]:
        report = LoweringReport(target=self.caps.format_id)
        report.add(Severity.INFO, "lossless", "Full internal model exported unchanged.")
        payload = {
            "format": "djprep/1",
            "audio_path": audio_path,
            "preparation": prep.model_dump(mode="json"),
        }
        if include_analysis:
            payload["analysis"] = analysis.model_dump(mode="json")
        return json.dumps(payload, indent=2).encode("utf-8"), report
