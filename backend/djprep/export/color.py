"""Colour quantisation for targets with a fixed palette.

Rekordbox surfaces a 16-colour hot cue palette. The application lets the user
pick any colour they like, so export has to map an arbitrary RGB value onto the
nearest palette entry -- and "nearest" must mean *perceptually* nearest, not
nearest in RGB space.

RGB Euclidean distance is a poor perceptual metric: it treats a step in green as
equal to a step in blue, when human vision is far more sensitive to the former.
Converting to CIE L*a*b* first (via linear sRGB and XYZ, D65 white point) gives a
space where Euclidean distance approximates perceived difference, so a user's
teal maps to the palette's teal rather than to a green that happens to be closer
in raw channel values.

The palette itself is data, not code (`djprep/data/palettes.json`), because it is
a property of a specific version of a specific piece of third-party software.
`scripts/extract_rekordbox_palette.py` can read the real palette out of a local
Rekordbox installation and overwrite the shipped approximation.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from djprep.models.common import RGB

PALETTES_PATH = Path(__file__).resolve().parent.parent / "data" / "palettes.json"


def _srgb_to_linear(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(v) for v in rgb)
    # sRGB -> XYZ (D65)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    # Normalise by the D65 white point
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    def f(t: float) -> float:
        return t ** (1 / 3) if t > 216 / 24389 else (841 / 108) * t + 4 / 29
    fx, fy, fz = f(x / xn), f(y / yn), f(z / zn)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    """CIE76 colour difference. Adequate for snapping to 16 well-separated hues."""
    la, aa, ba = rgb_to_lab(a)
    lb, ab, bb = rgb_to_lab(b)
    return ((la - lb) ** 2 + (aa - ab) ** 2 + (ba - bb) ** 2) ** 0.5


@lru_cache(maxsize=1)
def load_palettes() -> dict:
    if PALETTES_PATH.exists():
        return json.loads(PALETTES_PATH.read_text())
    return {}


def get_palette(name: str) -> list[dict]:
    return load_palettes().get(name, {}).get("colors", [])


def snap(color: RGB, palette: list[dict]) -> tuple[RGB, dict, float]:
    """Snap `color` to its perceptually nearest palette entry.

    Returns (snapped colour, palette entry, delta-E). The distance is returned so
    the caller can decide whether the change is worth reporting: a delta-E under
    about 5 is imperceptible to most people and does not deserve a warning, while
    a delta-E of 40 means the user's pink came out orange and they should know.
    """
    if not palette:
        return color, {}, 0.0
    src = (color.r, color.g, color.b)
    best, best_d = palette[0], float("inf")
    for entry in palette:
        d = delta_e(src, (entry["r"], entry["g"], entry["b"]))
        if d < best_d:
            best, best_d = entry, d
    return RGB(r=best["r"], g=best["g"], b=best["b"]), best, best_d
