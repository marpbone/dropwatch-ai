#!/usr/bin/env python3
"""Replace the shipped Rekordbox palette approximation with the real values.

    pip install pyrekordbox
    python scripts/extract_rekordbox_palette.py

Reads the colour table out of a local Rekordbox installation and rewrites the
`rekordbox_hotcue` entry in `backend/djprep/data/palettes.json`. No other code
changes: the exporter reads whatever is in that file, which is precisely why the
palette lives in data rather than in a serialiser.

Rekordbox 6+ stores its library in an encrypted SQLite database; pyrekordbox
handles locating and decrypting it. If it cannot (a new Rekordbox version, a
non-standard install path), the shipped approximation stays in place and export
still works -- colours will just occasionally snap to a neighbouring hue.
"""
from __future__ import annotations

import json
from pathlib import Path

PALETTES = Path(__file__).resolve().parents[1] / "backend/djprep/data/palettes.json"


def main() -> None:
    try:
        from pyrekordbox import Rekordbox6Database
    except ImportError as exc:
        raise SystemExit("pip install pyrekordbox") from exc

    try:
        db = Rekordbox6Database()
    except Exception as exc:
        raise SystemExit(
            f"could not open the Rekordbox database ({exc}).\n"
            "The shipped approximation remains in use; nothing is broken."
        ) from exc

    colors = []
    try:
        for row in db.get_my_tag() if False else db.get_color():
            name = getattr(row, "Commnt", None) or getattr(row, "Name", "") or ""
            val = getattr(row, "ColorCode", None)
            cid = getattr(row, "ID", None)
            if val is None:
                continue
            v = int(val)
            colors.append({"id": int(cid) if cid is not None else len(colors) + 1,
                           "name": str(name).strip() or f"Colour {len(colors) + 1}",
                           "r": (v >> 16) & 0xFF, "g": (v >> 8) & 0xFF, "b": v & 0xFF})
    except Exception as exc:
        raise SystemExit(f"could not read the colour table: {exc}") from exc

    if not colors:
        raise SystemExit("no colours found; leaving the shipped palette in place")

    data = json.loads(PALETTES.read_text())
    data["rekordbox_hotcue"]["colors"] = colors
    data["rekordbox_hotcue"]["source"] = "extracted from a local Rekordbox database"
    PALETTES.write_text(json.dumps(data, indent=2))
    print(f"wrote {len(colors)} colours to {PALETTES}")


if __name__ == "__main__":
    main()
