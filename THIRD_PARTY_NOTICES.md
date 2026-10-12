# Third-party notices

This project is licensed under the GNU General Public License, version 3
(see [`LICENSE`](LICENSE)), because it is a derivative work of MicroRTS. It
also incorporates design and interface elements from MicroRTS-Py, which is
MIT licensed. The details are below.

## MicroRTS (GPL-3.0)

- Source: https://github.com/Farama-Foundation/MicroRTS (originally https://github.com/santiontanon/microrts)
- Author: Santiago Ontañón and contributors
- License: GNU General Public License v3 (the full text is in [`LICENSE`](LICENSE))

What is derived from it:

- `microrts_pz/engine.py`, `microrts_pz/unit_types.py` and `microrts_pz/game.py`
  are Python ports of the game rules and data in `rts.GameState`,
  `rts.PhysicalGameState`, `rts.UnitAction`, `rts.ResourceUsage`,
  `rts.PlayerAction`, `rts.PartiallyObservableGameState`, `rts.units.*`,
  `ai.reward.*`, `ai.jni.JNIAI` and `tests.JNIGridnetClientSelfPlay`.
- `microrts_pz/maps/` holds unmodified map files copied from MicroRTS's `maps/` directory.
- `microrts_pz/bots.py` includes a simplified version of MicroRTS's `WorkerRush` strategy.
- `tools/parity/ParityHarness.java` links against MicroRTS to compare the two engines.
  It is a development tool and is not part of the Python package.

No MicroRTS Java code or JAR is bundled or needed at runtime.

## MicroRTS-Py / Gym-µRTS (MIT)

- Source: https://github.com/Farama-Foundation/MicroRTS-Py (originally https://github.com/vwxyzjn/gym-microrts)
- License: MIT, reproduced below.

What is derived from it: the gridnet observation encoding (one-hot feature
planes), the action and action-mask layout, the reward-component weighting
interface and the general environment design.

```
MIT License

Copyright (c) 2020 gym-microrts developers
Copyright (c) 2022 Farama Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Sounds and graphics

The browser viewer's sounds are synthesised at runtime with the Web Audio
API, and its terrain and effects are drawn with the Canvas API. No
third-party audio or font assets are included.

The unit, building and resource sprites (`tools/sprites/sprite_sheet.jpg` and
`tools/sprites/spider_drones.jpg`, the PNGs cut from them in `microrts_pz/render/sprites/`, and their WebP copies
embedded in `microrts_pz/render/viewer.html`) are AI-generated artwork
supplied by the project maintainers and dedicated to the public domain under
[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
They may be reused for any purpose without attribution.

## Runtime dependencies

NumPy (BSD-3-Clause), Gymnasium (MIT) and PettingZoo (MIT) are installed
from PyPI and are not redistributed here.
