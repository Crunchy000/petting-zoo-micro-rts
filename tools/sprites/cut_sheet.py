#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Cut ``sprite_sheet.jpg`` and ``spider_drones.jpg`` into transparent PNGs in ``microrts_pz/render/sprites``.

The main sheet is a grid (5 columns x 4 rows) on a flat dark background; the
Light unit comes from the drill column of ``spider_drones.jpg``. Each cell is cropped, the
background connected to the cell border is keyed out (with a soft, colour
corrected edge) and the sprite is trimmed and scaled to at most 112 px.
Run ``embed.py`` afterwards to inline the result into the viewer.
Needs Pillow, NumPy and SciPy.
"""

import os

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "microrts_pz", "render", "sprites")
XS = [35, 231, 427, 623, 819, 1015]  # grid line positions in the sheet
YS = [27, 163, 299, 436, 570]
BG = np.array([34.0, 39.0, 45.0])
CELLS = {
    (0, 0): "Base_0", (0, 1): "Barracks_0", (0, 2): "Worker_0", (0, 3): "Resource", (0, 4): "Resource_small",
    (1, 1): "Ranged_0", (1, 2): "Heavy_0",
    (2, 0): "Base_1", (2, 1): "Barracks_1", (2, 2): "Worker_1",
    (3, 1): "Ranged_1", (3, 2): "Heavy_1",
}
#: spider_drones.jpg: name -> (top, bottom, left, right) of the drill drone, below the column title
DRONES = {"Light_0": (100, 316, 546, 1018), "Light_1": (328, 568, 546, 1018)}


def cut(cell: np.ndarray, shadow: bool = False) -> Image.Image:
    """``shadow`` also keys out a grey floor shadow a little darker than the background."""
    d = np.sqrt(((cell - BG) ** 2).sum(-1))
    flat = d < 16
    if shadow:
        m = cell.mean(-1)
        flat |= (m >= 24) & (m <= 37) & (np.ptp(cell, axis=-1) < 10)
    lab, _ = ndimage.label(flat)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(edge))
    alpha = np.where(bg, 0.0, 1.0)
    rim = ndimage.binary_dilation(bg, iterations=2) & ~bg
    alpha[rim] = np.clip(d[rim] / 40, 0, 1)
    a = alpha[..., None]
    rgb = np.where(a > 0.02, (cell - (1 - a) * BG) / np.maximum(a, 0.02), 0)  # un-blend the background
    img = Image.fromarray(np.dstack([np.clip(rgb, 0, 255), alpha * 255]).astype(np.uint8), "RGBA")
    img = img.crop(img.getchannel("A").point(lambda v: 255 if v > 20 else 0).getbbox())
    s = 112 / max(img.size)
    return img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS) if s < 1 else img


def main() -> None:
    sheet = np.array(Image.open(os.path.join(HERE, "sprite_sheet.jpg")).convert("RGB")).astype(float)
    os.makedirs(OUT, exist_ok=True)
    for (r, c), name in CELLS.items():
        cut(sheet[YS[r] + 8 : YS[r + 1] - 8, XS[c] + 8 : XS[c + 1] - 8]).save(os.path.join(OUT, name + ".png"), optimize=True)
    drones = np.array(Image.open(os.path.join(HERE, "spider_drones.jpg")).convert("RGB")).astype(float)
    for name, (t, b, l, r) in DRONES.items():
        cut(drones[t:b, l:r], shadow=True).save(os.path.join(OUT, name + ".png"), optimize=True)


if __name__ == "__main__":
    main()
