#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Inline ``microrts_pz/render/sprites/*.png`` into ``viewer.html`` as WebP data URIs.

The viewer stays a single self-contained file (replays embed it), so the
sprites live in it between the ``__SPRITES_BEGIN__``/``__SPRITES_END__`` markers.
"""

import base64
import glob
import io
import json
import os
import re

from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "microrts_pz", "render")


def main() -> None:
    sprites = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "sprites", "*.png"))):
        buf = io.BytesIO()
        Image.open(path).save(buf, "WEBP", quality=88, method=6)
        sprites[os.path.splitext(os.path.basename(path))[0]] = "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()
    viewer = os.path.join(ROOT, "viewer.html")
    html = open(viewer, encoding="utf-8").read()
    block = "/*__SPRITES_BEGIN__*/ " + json.dumps(sprites, separators=(",", ":")).replace("},", "},\n") + " /*__SPRITES_END__*/"
    html, n = re.subn(r"/\*__SPRITES_BEGIN__\*/.*?/\*__SPRITES_END__\*/", lambda _: block, html, flags=re.S)
    assert n == 1, "sprite markers not found in viewer.html"
    open(viewer, "w", encoding="utf-8").write(html)


if __name__ == "__main__":
    main()
