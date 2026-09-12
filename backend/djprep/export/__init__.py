"""Export targets. Registry-based so adding a format touches one line."""
from __future__ import annotations

from djprep.export.base import (
    ColorMode,
    Exporter,
    ExporterCapabilities,
    LoweringReport,
    Note,
    Severity,
)
from djprep.export.json_export import JsonExporter
from djprep.export.rekordbox_xml import RekordboxXmlExporter

_REGISTRY: dict[str, Exporter] = {}


def register(exporter: Exporter) -> None:
    _REGISTRY[exporter.caps.format_id] = exporter


def get(format_id: str) -> Exporter:
    if format_id not in _REGISTRY:
        raise KeyError(f"unknown export format {format_id!r}; "
                       f"available: {sorted(_REGISTRY)}")
    return _REGISTRY[format_id]


def available() -> list[ExporterCapabilities]:
    return [e.caps for e in _REGISTRY.values()]


register(RekordboxXmlExporter())
register(JsonExporter())

__all__ = [
    "ColorMode",
    "Exporter",
    "ExporterCapabilities",
    "JsonExporter",
    "LoweringReport",
    "Note",
    "RekordboxXmlExporter",
    "Severity",
    "available",
    "get",
    "register",
]
