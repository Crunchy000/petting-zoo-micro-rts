# SPDX-License-Identifier: GPL-3.0-only
"""Turn the board 180 degrees so a policy sees every game from one corner.

A grid policy reads absolute positions, so it learns separate habits for
each starting corner and can end up much weaker in one of them. Rotating
player 0's view by 180 degrees makes it look like player 1's on the usual
point-symmetric maps, so one set of habits serves both sides.

Observations ``(..., H, W, C)`` reverse their rows and columns;
masks and actions also reverse their cells and remap each direction
``d -> (d + 2) % 4`` and each attack offset ``(dx, dy) -> (-dx, -dy)``.
Rotating twice gives back the original, so the same functions map an
action chosen on the rotated board back to the real one. They take NumPy
or JAX arrays (pass ``xp=jax.numpy`` for JAX).
"""

from __future__ import annotations

import numpy as np

#: Action components holding a direction (move, harvest, return, produce).
DIR_COMPONENTS = (1, 2, 3, 4)
#: Action component holding the attack target, an index into the attack grid.
ATTACK_COMPONENT = 6


def mask_permutation(action_dims) -> np.ndarray:
    """Index array that rotates the last axis of a ``sum(action_dims)``-wide mask or logit vector."""
    dims = [int(d) for d in action_dims]
    offs = np.concatenate([[0], np.cumsum(dims)])
    perm = np.arange(offs[-1])
    for k in DIR_COMPONENTS:
        perm[offs[k] : offs[k + 1]] = offs[k] + (np.arange(4) + 2) % 4
    a = dims[ATTACK_COMPONENT]
    perm[offs[ATTACK_COMPONENT] : offs[ATTACK_COMPONENT + 1]] = offs[ATTACK_COMPONENT] + (a - 1 - np.arange(a))
    return perm


def rotate_obs(obs, xp=np):
    """Rotate observations ``(..., H, W, C)``."""
    return xp.flip(obs, axis=(-3, -2))


def rotate_mask(mask, perm, xp=np):
    """Rotate masks or logits ``(..., HW, sum(action_dims))``."""
    return xp.flip(mask, axis=-2)[..., perm]


def rotate_action(action, action_dims, xp=np):
    """Rotate actions ``(..., HW, 7)`` (a rotated action played on the real board, or the reverse)."""
    a = xp.flip(action, axis=-2)
    n_att = int(action_dims[ATTACK_COMPONENT])
    cols = []
    for k in range(a.shape[-1]):
        c = a[..., k]
        if k in DIR_COMPONENTS:
            c = (c + 2) % 4
        elif k == ATTACK_COMPONENT:
            c = n_att - 1 - c
        cols.append(c)
    return xp.stack(cols, axis=-1)
