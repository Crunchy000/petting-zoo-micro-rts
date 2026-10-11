# SPDX-License-Identifier: GPL-3.0-only
"""Gymnasium space for the gridnet action space with flat action masks."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.spaces import MultiDiscrete


class GridnetActionSpace(MultiDiscrete):
    """``MultiDiscrete`` with ``nvec = tile(component_dims, num_cells)`` (flat, 1-D).

    Each grid cell carries one 7-component action
    ``[type, move_dir, harvest_dir, return_dir, produce_dir, produce_type, attack_target]``
    which is only read when the cell holds an idle unit of the acting player.

    :meth:`sample` additionally accepts the environment's flat ``action_mask``
    (``num_cells * sum(component_dims)`` int8 values, components concatenated
    per cell), as PettingZoo's test utilities pass it directly.
    """

    def __init__(self, num_cells: int, component_dims: np.ndarray, seed: Any = None):
        self.num_cells = int(num_cells)
        self.component_dims = np.asarray(component_dims, dtype=np.int64)
        self.mask_offsets = np.concatenate([[0], np.cumsum(self.component_dims)])
        super().__init__(np.tile(self.component_dims, self.num_cells), dtype=np.int64, seed=seed)

    @property
    def mask_size(self) -> int:
        return int(self.mask_offsets[-1]) * self.num_cells

    def sample(self, mask: Any = None, probability: Any = None) -> np.ndarray:  # type: ignore[override]
        if isinstance(mask, np.ndarray):
            return self.sample_from_flat_mask(mask)
        return super().sample(mask=mask, probability=probability)

    def sample_from_flat_mask(self, mask: np.ndarray) -> np.ndarray:
        """Uniformly sample every component among its valid values (0 when none is valid)."""
        mask = np.asarray(mask).reshape(self.num_cells, -1)
        if mask.shape[1] != self.mask_offsets[-1]:
            raise ValueError(f"expected an action mask with {self.mask_size} entries, got {mask.size}")
        out = np.empty((self.num_cells, len(self.component_dims)), dtype=np.int64)
        for k in range(len(self.component_dims)):
            sub = mask[:, self.mask_offsets[k] : self.mask_offsets[k + 1]] > 0
            noise = self.np_random.random(sub.shape) + 1.0  # strictly positive
            out[:, k] = np.argmax(np.where(sub, noise, 0.0), axis=1)  # 0 when no valid entry
        return out.reshape(-1)

    def __repr__(self) -> str:
        return f"GridnetActionSpace(num_cells={self.num_cells}, component_dims={self.component_dims.tolist()})"
