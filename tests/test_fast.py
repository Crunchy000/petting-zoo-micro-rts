"""The compiled batch engine (``microrts_pz.fast``) against the reference Python engine, tick by tick."""

import numpy as np
import pytest

pytest.importorskip("numba")

from microrts_pz.bots import RandomBot, WorkerRushBot  # noqa: E402
from microrts_pz.fast import BatchGame  # noqa: E402
from microrts_pz.game import MicroRTSGame  # noqa: E402
from microrts_pz.unit_types import UnitTypeTable  # noqa: E402

ARMIES = [None, "Light", ("Light", "Ranged", "Heavy"), ("Heavy", "Ranged")]


def bot_args(spec):
    kind, harvesters, army = spec
    return kind, harvesters, army


def run_pair(map_path, specs, seed, ticks, conflict=1):
    """Play the same game in both engines; ``specs`` give each player's bot (``("rush", h, army)`` or ``("random",)``)."""
    utt = UnitTypeTable(move_conflict_resolution=conflict)
    ref = MicroRTSGame(map_path, utt, rng=np.random.default_rng(seed))
    ref.reset()
    fast = BatchGame([map_path], num_games=1, max_steps=10**9, move_conflict_resolution=conflict, auto_reset=False)
    fast.reset(seed=seed)
    py_bots = [WorkerRushBot(s[1], s[2]) if s[0] == "rush" else RandomBot(seed + i) for i, s in enumerate(specs)]
    for tick in range(ticks):
        for p in (0, 1):
            obs, mask = fast.observe(p)
            np.testing.assert_array_equal(obs[0], ref.observation(p), err_msg=f"observation p{p} tick {tick}")
            np.testing.assert_array_equal(mask[0], ref.action_mask(p), err_msg=f"mask p{p} tick {tick}")
        actions = []
        for p, (s, bot) in enumerate(zip(specs, py_bots)):
            a = np.asarray(bot.act(ref, p)).reshape(ref.num_cells, -1)
            if s[0] == "rush":
                b = fast.bot_actions(p, "rush", s[1], s[2])[0]
                np.testing.assert_array_equal(b, a, err_msg=f"bot p{p} tick {tick}")
            actions.append(a)
        raw, over = ref.step(actions[0], actions[1])
        fraw, fover, _, fwin = fast.step(actions[0][None], actions[1][None])
        np.testing.assert_array_equal(fraw[0], raw, err_msg=f"rewards tick {tick}")
        assert fover[0] == over, f"gameover tick {tick}"
        assert fast.time[0] == ref.gs.time
        if over:
            assert fwin[0] == ref.winner
            return tick
    return ticks


@pytest.mark.parametrize("map_path", [
    "maps/16x16/basesWorkers16x16.xml",
    "maps/8x8/basesWorkers8x8A.xml",
    "maps/letMeOut.xml",
    "maps/16x16/TwoBasesBarracks16x16.xml",
    "maps/NoWhereToRun9x8.xml",
])
@pytest.mark.parametrize("matchup", range(4))
def test_bots_match(map_path, matchup):
    specs = [
        [("rush", 1, None), ("rush", 2, ("Light", "Ranged", "Heavy"))],
        [("rush", 2, "Heavy"), ("rush", 2, "Ranged")],
        [("random",), ("rush", 2, ("Heavy", "Ranged", "Light"))],
        [("random",), ("random",)],
    ][matchup]
    run_pair(map_path, specs, seed=matchup, ticks=1500)


@pytest.mark.parametrize("conflict", [1, 3])
def test_conflict_rules(conflict):
    run_pair("maps/8x8/basesWorkers8x8A.xml", [("random",), ("random",)], seed=5, ticks=1500, conflict=conflict)


def test_batch_auto_reset_and_truncation():
    games = BatchGame(["maps/8x8/basesWorkers8x8A.xml"], num_games=8, max_steps=50)
    games.reset(seed=0)
    zero = np.zeros((8, 64, 7), dtype=np.int64)
    for t in range(50):
        raw, over, trunc, winner = games.step(zero, zero)
    assert trunc.all() and not over.any()
    assert (games.episode_steps == 0).all() and (games.time == 0).all()
