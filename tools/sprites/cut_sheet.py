#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Cut ``sprite_sheet.jpg`` into transparent PNGs in ``microrts_pz/render/sprites``.

The sheet is a 4x4 grid on a flat dark background. Each cell is cropped, the
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
XS = [20, 183, 348, 511, 675, 839]  # grid line positions in the sheet
YS = [22, 156, 291, 424, 556]
BG = np.array([36.0, 40.0, 49.0])
CELLS = {
    (0, 0): "Base_0", (0, 1): "Barracks_0", (0, 2): "Worker_0", (0, 3): "Resource",
    (1, 0): "Light_0", (1, 1): "Ranged_0", (1, 2): "Heavy_0",
    (2, 0): "Base_1", (2, 1): "Barracks_1", (2, 2): "Worker_1", (2, 4): "Resource_small",
    (3, 0): "Light_1", (3, 1): "Ranged_1", (3, 2): "Heavy_1",
}


def cut(cell: np.ndarray) -> Image.Image:
    d = np.sqrt(((cell - BG) ** 2).sum(-1))
    lab, _ = ndimage.label(d < 16)
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


if __name__ == "__main__":
    main()
