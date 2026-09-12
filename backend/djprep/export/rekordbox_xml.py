"""Rekordbox XML export.

Rekordbox's documented interchange format is `DJ_PLAYLISTS` XML: a COLLECTION of
TRACK elements, each carrying TEMPO elements (the beat grid) and POSITION_MARK
elements (cues and loops), plus a PLAYLISTS tree. Rekordbox imports it as a
separate library tree that the user drags into their collection.

What this exporter has to reconcile:

  * **Eight hot cue slots.** `Num` 0-7 are hot cues A-H; `Num` -1 is a memory
    cue, of which there may be many. More cues than slots is the normal case, not
    an error, so we allocate slots by the user's priority order and lower the
    remainder to memory cues -- reporting each one.

  * **A fixed colour palette.** Rekordbox snaps imported colours to its own
    palette anyway; doing it ourselves, perceptually, means the user sees the
    same colour we predicted rather than whatever nearest-in-RGB produced.

  * **A parametric grid.** TEMPO is (`Inizio`, `Bpm`, `Metro`, `Battito`) -- a
    start time and a tempo, exactly the constant-tempo model the analyser fits.
    `Battito` is the beat's position within the bar, which is where our downbeat
    phase estimate lands.

  * **Loops as marked-up cues.** A loop is a POSITION_MARK with both `Start` and
    `End`. See LOOP_TYPE below for the one genuinely uncertain detail.

Everything here reads the neutral model and writes XML. No other module imports
this one; the API reaches it only through the registry.
"""
from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

from djprep.export.base import (
    ColorMode,
    Exporter,
    ExporterCapabilities,
    LoweringReport,
    Severity,
)
from djprep.export.color import get_palette, snap
from djprep.models.analysis import TrackAnalysis
from djprep.models.prep import TrackPreparation

# POSITION_MARK Type values, per Pioneer's published XML format list:
#   0 = Cue, 1 = Fade-In, 2 = Fade-Out, 3 = Load, 4 = Loop
#
# In practice Rekordbox's own exports write Type="0" for loops and distinguish
# them purely by the presence of an End attribute; the documented Type=4 is
# accepted on import but is not what round-tripping produces. We default to the
# behaviour that matches real exports and leave the documented value available,
# because this is exactly the kind of vendor detail that changes between
# versions and should be a switch rather than a constant buried in a serialiser.
LOOP_TYPE_COMPAT = "0"
LOOP_TYPE_SPEC = "4"

# A perceptual difference below this is not worth telling the user about.
DELTA_E_NOTABLE = 8.0

CAPS = ExporterCapabilities(
    format_id="rekordbox_xml",
    display_name="Rekordbox XML",
    file_extension=".xml",
    max_hot_cues=8,
    supports_memory_cues=True,
    supports_loops=True,
    supports_loop_colors=True,
    supports_cue_names=True,
    color_mode=ColorMode.PALETTE,
    palette=[(c["r"], c["g"], c["b"]) for c in get_palette("rekordbox_hotcue")],
    notes=("Imports as a separate tree in Rekordbox (File > Import > Import "
           "Collection). Hot cues A-H map to Num 0-7; anything beyond that is "
           "written as a memory cue."),
)


def _file_uri(path: str) -> str:
    """Rekordbox expects `file://localhost/` plus a percent-encoded absolute path."""
    p = Path(path).resolve()
    quoted = urllib.parse.quote(str(p).replace("\\", "/"), safe="/:")
    if not quoted.startswith("/"):
        quoted = "/" + quoted
    return f"file://localhost{quoted}"


class RekordboxXmlExporter:
    caps = CAPS

    def __init__(self, loop_type: str = LOOP_TYPE_COMPAT,
                 playlist_name: str = "djprep") -> None:
        self.loop_type = loop_type
        self.playlist_name = playlist_name

    def export(self, prep: TrackPreparation, analysis: TrackAnalysis,
               audio_path: str, **options) -> tuple[bytes, LoweringReport]:
        report = LoweringReport(target=self.caps.format_id)
        palette = get_palette("rekordbox_hotcue")
        meta, grid = analysis.meta, analysis.beat_grid

        root = ET.Element("DJ_PLAYLISTS", {"Version": "1.0.0"})
        ET.SubElement(root, "PRODUCT", {
            "Name": "djprep", "Version": "0.1.0", "Company": "djprep",
        })
        collection = ET.SubElement(root, "COLLECTION", {"Entries": "1"})
        track = ET.SubElement(collection, "TRACK", {
            "TrackID": "1",
            "Name": meta.title or Path(meta.filename).stem,
            "Artist": meta.artist or "",
            "Kind": Path(meta.filename).suffix.lstrip(".").upper() + " File",
            "TotalTime": str(round(meta.duration_sec)),
            "AverageBpm": f"{grid.bpm:.2f}",
            "SampleRate": str(meta.sample_rate),
            "Location": _file_uri(audio_path),
        })

        # --- beat grid -------------------------------------------------------
        # One TEMPO element expresses the whole grid because the analyser fits a
        # constant tempo. Battito is 1-based within the bar.
        ET.SubElement(track, "TEMPO", {
            "Inizio": f"{grid.anchor_sec + grid.downbeat_offset * grid.beat_period:.6f}",
            "Bpm": f"{grid.bpm:.2f}",
            "Metro": f"{grid.beats_per_bar}/4",
            "Battito": "1",
        })
        if not grid.is_constant:
            report.add(Severity.DROPPED, "variable_tempo",
                       "Track has variable tempo; exported as a single constant-tempo "
                       "grid, which will drift. Check the grid in Rekordbox.")
        if grid.downbeat_confidence < 0.4:
            report.add(Severity.INFO, "low_downbeat_confidence",
                       f"Downbeat phase confidence is only "
                       f"{grid.downbeat_confidence:.0%}; the bar-1 position may be "
                       f"off by a beat.")

        # --- hot cue slot allocation ----------------------------------------
        # Two separate decisions, and conflating them produces something a DJ
        # cannot use:
        #
        #   *Which* markers get one of the eight hot cue slots is a question of
        #   importance -- answered by the user's priority order (confidence by
        #   default, reorderable in the UI).
        #
        #   *Which slot each one gets* is a question of ergonomics, and the
        #   answer is always chronological. Pads A-H run left to right under the
        #   DJ's hand; if slot A is the outro and slot B is the intro, the
        #   controller is unusable under pressure. An earlier version allocated
        #   slots straight down the priority list and produced exactly that.
        cues = prep.prioritised_cues()
        loops = prep.active_loops()
        max_hot = self.caps.max_hot_cues if self.caps.max_hot_cues is not None else 10**6

        rank = {c.id: i for i, c in enumerate(cues)}
        loop_bias = len(cues) if options.get("cues_before_loops", True) else 0
        markers: list[tuple[float, object, bool]] = (
            [(float(rank[c.id]), c, False) for c in cues]
            + [(loop_bias + i - (1e6 if options.get("loops_first") else 0), lp, True)
               for i, lp in enumerate(loops)]
        )
        by_priority = sorted(markers, key=lambda m: m[0])
        hot = by_priority[:max_hot]
        overflow = by_priority[max_hot:]

        def _time(m) -> float:
            return m.start_sec if hasattr(m, "start_sec") else m.time_sec

        slot_of: dict[str, int] = {}
        for i, (_, m, _is_loop) in enumerate(sorted(hot, key=lambda x: _time(x[1]))):
            slot_of[m.id] = i
        for _, m, is_loop in overflow:
            report.add(Severity.ADJUSTED, "hot_cue_overflow",
                       f"{'Loop' if is_loop else 'Cue'} '{m.label}' at "
                       f"{_mmss(_time(m))} did not fit the {max_hot} hot cue slots "
                       f"and was written as a memory "
                       f"{'loop' if is_loop else 'cue'}.", m.id)

        def _colour(marker, is_loop: bool):
            colour, entry, d = snap(marker.color, palette)
            if d > DELTA_E_NOTABLE:
                report.add(Severity.ADJUSTED, "color_snapped",
                           f"{'Loop' if is_loop else 'Cue'} '{marker.label}' colour "
                           f"{marker.color.to_hex()} snapped to palette colour "
                           f"{entry.get('name', '?')} {colour.to_hex()}.", marker.id)
            return colour

        # Emit in chronological order so the XML reads the way the track plays.
        emit = sorted(
            [(c.time_sec, c, False) for c in cues]
            + [(lp.start_sec, lp, True) for lp in loops],
            key=lambda x: x[0])
        for _, marker, is_loop in emit:
            colour = _colour(marker, is_loop)
            attrs = {
                "Name": marker.label,
                "Type": self.loop_type if is_loop else "0",
                "Start": f"{(marker.start_sec if is_loop else marker.time_sec):.3f}",
                "Num": str(slot_of.get(marker.id, -1)),
                "Red": str(colour.r), "Green": str(colour.g), "Blue": str(colour.b),
            }
            if is_loop:
                attrs["End"] = f"{marker.end_sec:.3f}"
            ET.SubElement(track, "POSITION_MARK", attrs)

        # --- things the format simply cannot carry ---------------------------
        n_reasons = sum(len(c.reasons) for c in cues)
        if n_reasons:
            report.add(Severity.DROPPED, "explanations_not_representable",
                       f"{n_reasons} recommendation explanations and all confidence "
                       f"scores were dropped: the Rekordbox XML schema has nowhere "
                       f"to put them. Use the JSON export to keep them.")
        kinds = {c.kind.value for c in cues}
        if kinds:
            report.add(Severity.DROPPED, "cue_kind_not_representable",
                       "Cue types (drop, breakdown, ...) survive only as cue names; "
                       "Rekordbox has no typed-cue concept.")

        # --- playlist --------------------------------------------------------
        playlists = ET.SubElement(root, "PLAYLISTS")
        rootnode = ET.SubElement(playlists, "NODE",
                                 {"Type": "0", "Name": "ROOT", "Count": "1"})
        node = ET.SubElement(rootnode, "NODE", {
            "Name": self.playlist_name, "Type": "1", "KeyType": "0", "Entries": "1",
        })
        ET.SubElement(node, "TRACK", {"Key": "1"})

        raw = ET.tostring(root, encoding="utf-8")
        pretty = minidom.parseString(raw).toprettyxml(indent="  ", encoding="UTF-8")
        return pretty, report


def _mmss(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:05.2f}"


_check: Exporter = RekordboxXmlExporter()
