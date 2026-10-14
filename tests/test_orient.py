"""Turning the board 180 degrees (``training/orient.py``)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from orient import mask_permutation, rotate_action, rotate_mask, rotate_obs  # noqa: E402

from microrts_pz.bots import WorkerRushBot  # noqa: E402
from microrts_pz.game import MicroRTSGame  # noqa: E402


def test_player0_turned_sees_what_player1_sees():
    game = MicroRTSGame("maps/24x24/basesWorkers24x24.xml")  # a point-symmetric map
    game.reset()
    perm = mask_permutation(game.action_dims)
    bot = WorkerRushBot()
    for _ in range(60):
        assert np.array_equal(rotate_obs(game.observation(0)), game.observation(1))
        assert np.array_equal(rotate_mask(game.action_mask(0), perm), game.action_mask(1))
        # player 0 plays player 1's choice, turned back onto its own corner
        a1 = np.asarray(bot.act(game, 1)).reshape(game.num_cells, -1)
        game.step(rotate_action(a1, game.action_dims), a1)


def test_rotation_is_its_own_inverse():
    dims = np.array([6, 4, 4, 4, 4, 7, 49])
    rng = np.random.default_rng(0)
    act = np.stack([rng.integers(0, d, 64) for d in dims], axis=-1)
    assert np.array_equal(rotate_action(rotate_action(act, dims), dims), act)
    perm = mask_permutation(dims)
    assert np.array_equal(perm[perm], np.arange(dims.sum()))
