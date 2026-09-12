#!/usr/bin/env python3
"""Generate the Dropwatch AI mark.

Drawn procedurally rather than hand-designed, so it is reproducible, versionable
and tweakable by changing numbers instead of reopening a vector editor.

The mark encodes what the product does. A watch bezel carries 12 tick marks with
every third one long -- a bar grid with the downbeats emphasised, which is the
structure the whole analyser is built on. Inside it, levels ramp up through a
build, cut to almost nothing, and spike into a drop picked out in signal amber
with a cue marker capping it. So: a watch that is watching for the drop.

Everything is supersampled 4x and downscaled, because at 32 px in a browser tab
the antialiasing is the difference between a logo and a smudge.

    python scripts/make_logo.py --out frontend/src/assets/dropwatch.png
"""
from __future__ import annotations

import argparse
import base64
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SS = 4                      # supersampling factor
INK = (232, 163, 61)        # signal amber - the accent
INK_SOFT = (240, 188, 108)
BODY = (150, 168, 196)      # cool steel for the ordinary bars
RING = (92, 108, 132)
BG_OUTER = (18, 22, 30)
BG_INNER = (30, 38, 52)


def _radial_background(size: int) -> Image.Image:
    """Dark disc with a soft inner glow, so the mark has depth at small sizes."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    steps = 64
    for i in range(steps, 0, -1):
        f = i / steps
        r = size / 2 * f
        mix = 1.0 - f
        col = tuple(
            int(BG_OUTER[c] + (BG_INNER[c] - BG_OUTER[c]) * (mix ** 1.6))
            for c in range(3)
        )
        d.ellipse([size / 2 - r, size / 2 - r, size / 2 + r, size / 2 + r],
                  fill=(*col, 255))
    return img


def _bezel(d: ImageDraw.ImageDraw, size: int) -> None:
    """12 ticks, every third long: a bar grid with the downbeats emphasised."""
    cx = cy = size / 2
    r_out = size * 0.455
    for i in range(12):
        ang = -math.pi / 2 + (i / 12) * 2 * math.pi
        downbeat = i % 3 == 0
        length = size * (0.080 if downbeat else 0.040)
        width = max(1, int(size * (0.026 if downbeat else 0.015)))
        col = INK if downbeat else RING
        x1 = cx + math.cos(ang) * (r_out - length)
        y1 = cy + math.sin(ang) * (r_out - length)
        x2 = cx + math.cos(ang) * r_out
        y2 = cy + math.sin(ang) * r_out
        d.line([x1, y1, x2, y2], fill=(*col, 255), width=width)


def _envelope() -> list[float]:
    """The silhouette: a build, the gap, the drop, the body settling.

    Seven bars, not twenty. An earlier version used enough bars to look like a
    real waveform and at 32 px it collapsed into a solid block. The shape has to
    survive being 20 pixels wide, which means exaggerating far past anything a
    real signal looks like and using as few marks as will still tell the story.
    """
    return [
        0.26,   # intro
        0.44,   # build
        0.64,
        0.09,   # the gap before it lands -- the most important bar here
        1.00,   # DROP
        0.72,
        0.46,   # settling
    ]


def _waveform(d: ImageDraw.ImageDraw, size: int) -> None:
    """Bars standing on a baseline, with the drop picked out in amber.

    Deliberately *not* mirrored about a centre line. A symmetric waveform is what
    an audio editor draws, but as a mark it reads as a lens-shaped blob: the eye
    sees the outline, not the individual levels. Standing the bars on a baseline
    turns the same data into a profile you can read instantly -- the ramp, the
    gap, the spike -- which is the entire point of the logo.
    """
    env = _envelope()
    n = len(env)
    drop_i = max(range(n), key=lambda i: env[i])
    cx = size / 2
    baseline = size * 0.655
    span = size * 0.54
    step = span / n
    bar_w = step * 0.56
    max_h = size * 0.40

    for i, a in enumerate(env):
        if i == drop_i:
            continue                        # drawn last, on top
        x = cx - span / 2 + step * (i + 0.5)
        h = max(max_h * a, bar_w * 0.85)
        d.rounded_rectangle([x - bar_w / 2, baseline - h, x + bar_w / 2, baseline],
                            radius=bar_w / 2, fill=(*BODY, 255))

    x = cx - span / 2 + step * (drop_i + 0.5)
    h = max_h * 1.0
    w = bar_w * 1.25
    d.rounded_rectangle([x - w / 2, baseline - h, x + w / 2, baseline],
                        radius=w / 2, fill=(*INK, 255))
    # Cue marker capping the drop bar.
    r = w * 0.78
    d.ellipse([x - r, baseline - h - r * 1.25, x + r, baseline - h + r * 0.75],
              fill=(*INK_SOFT, 255))


def render(px: int = 128) -> Image.Image:
    size = px * SS
    img = _radial_background(size)
    d = ImageDraw.Draw(img)
    _bezel(d, size)
    _waveform(d, size)


    img = img.resize((px, px), Image.LANCZOS)
    return img.filter(ImageFilter.UnsharpMask(radius=1.1, percent=55, threshold=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="frontend/src/assets/dropwatch.png")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--data-uri", action="store_true",
                    help="also write a .txt containing a base64 data URI")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = render(args.size)
    img.save(out, "PNG", optimize=True)
    print(f"wrote {out} ({out.stat().st_size} bytes, {args.size}x{args.size})")

    if args.data_uri:
        uri = "data:image/png;base64," + base64.b64encode(out.read_bytes()).decode()
        uri_path = out.with_suffix(".datauri.txt")
        uri_path.write_text(uri)
        print(f"wrote {uri_path} ({len(uri)} chars)")


if __name__ == "__main__":
    main()
