"""The JAX engine (``microrts_pz.jax_engine``) against the reference Python engine, tick by tick."""

import numpy as np
import pytest

pytest.importorskip("jax")

from microrts_pz.bots import RandomBot, WorkerRushBot  # noqa: E402
from microrts_pz.engine import TYPE_ATTACK_LOCATION, TYPE_NONE, TYPE_PRODUCE  # noqa: E402
from microrts_pz.game import MicroRTSGame  # noqa: E402
from microrts_pz.jax_engine import JaxBatchGame  # noqa: E402
from microrts_pz.unit_types import UnitTypeTable  # noqa: E402


class ChaosBot:
    """Mostly legal play with a share of arbitrary (often illegal) action components, to exercise the engine's quirks."""

    def __init__(self, seed, chaos=0.3):
        self.rng = np.random.default_rng(seed)
        self.random = RandomBot(seed + 1000)
        self.chaos = chaos

    def act(self, game, player):
        a = self.random.act(game, player)
        wild = np.stack([self.rng.integers(n, size=game.num_cells) for n in game.action_dims], 1)
        pick = self.rng.random(game.num_cells) < self.chaos
        a[pick] = wild[pick]
        return a


def make_bot(spec, seed):
    if spec[0] == "rush":
        return WorkerRushBot(spec[1], spec[2])
    if spec[0] == "chaos":
        return ChaosBot(seed, spec[1])
    return RandomBot(seed)


def assert_same_state(fast: JaxBatchGame, ref: MicroRTSGame, where: str):
    s = fast.state
    gs = ref.gs
    W, AG = gs.width, fast.attack_grid
    center = AG // 2
    utype = np.asarray(s.utype[0])
    occupied = np.flatnonzero(utype >= 0)
    by_pos = {u.x + u.y * W: u for u in gs.units}
    assert sorted(by_pos) == sorted(occupied.tolist()), f"unit positions {where}"
    birth = np.asarray(s.birth_t[0]).astype(np.int64) * (1 << 31) + np.asarray(s.birth_s[0])
    order = occupied[np.argsort(birth[occupied], kind="stable")]
    assert order.tolist() == [u.x + u.y * W for u in gs.units], f"unit order {where}"
    f = {k: np.asarray(getattr(s, k)[0]) for k in ("player", "hp", "res", "a_has", "a_type", "a_param", "a_ut", "a_tgt", "a_time")}
    for c, u in by_pos.items():
        got = (int(utype[c]), int(f["player"][c]), int(f["hp"][c]), int(f["res"][c]))
        assert got == (u.type.id, u.player, u.hp, u.resources), f"unit {u} {where}: {got}"
        a = gs.assignments.get(u.id)
        assert bool(f["a_has"][c]) == (a is not None), f"assignment of {u} {where}"
        if a is None:
            continue
        ua = a.action
        assert int(f["a_type"][c]) == ua.type and int(f["a_time"][c]) == a.time, f"action of {u} {where}"
        if ua.type == TYPE_ATTACK_LOCATION:
            k = int(f["a_tgt"][c])
            assert (u.x + k % AG - center, u.y + k // AG - center) == (ua.x, ua.y), f"attack target of {u} {where}"
        else:
            assert int(f["a_param"][c]) == ua.parameter, f"action parameter of {u} {where}"
        if ua.type == TYPE_PRODUCE:
            assert int(f["a_ut"][c]) == ua.unit_type.id
    assert np.asarray(s.resources[0]).tolist() == gs.player_resources, f"resources {where}"
    assert int(s.time[0]) == gs.time


def run_pair(map_path, specs, seed, ticks, version=1, conflict=1, check_every=1):
    utt = UnitTypeTable(version, conflict)
    ref = MicroRTSGame(map_path, utt, rng=np.random.default_rng(seed))
    ref.reset()
    fast = JaxBatchGame([map_path], num_games=1, max_steps=10**9, unit_type_table_version=version,
                        move_conflict_resolution=conflict, auto_reset=False)
    fast.reset()
    bots = [make_bot(s, seed + 17 * i) for i, s in enumerate(specs)]
    for tick in range(ticks):
        if tick % check_every == 0:
            assert_same_state(fast, ref, f"before tick {tick}")
            for p in (0, 1):
                obs, mask = fast.observe(p)
                np.testing.assert_array_equal(obs[0], ref.observation(p), err_msg=f"observation p{p} tick {tick}")
                np.testing.assert_array_equal(mask[0], ref.action_mask(p), err_msg=f"mask p{p} tick {tick}")
        actions = [np.asarray(b.act(ref, p)).reshape(ref.num_cells, -1) for p, b in enumerate(bots)]
        raw, over = ref.step(actions[0], actions[1])
        fraw, fover, _, fwin = fast.step(actions[0][None], actions[1][None])
        np.testing.assert_array_equal(fraw[0], raw, err_msg=f"rewards tick {tick}")
        assert fover[0] == over, f"gameover tick {tick}"
        if over:
            assert fwin[0] == ref.winner
            assert_same_state(fast, ref, "at the end")
            return tick
    return ticks


MAPS = [
    "maps/8x8/basesWorkers8x8A.xml",
    "maps/16x16/basesWorkers16x16.xml",
    "maps/16x16/TwoBasesBarracks16x16.xml",
    "maps/NoWhereToRun9x8.xml",
    "maps/letMeOut.xml",
    "maps/melee14x12Mixed18.xml",
]


@pytest.mark.parametrize("map_path", MAPS)
@pytest.mark.parametrize("matchup", range(4))
def test_matches_python_engine(map_path, matchup):
    specs = [
        [("rush", 1, None), ("rush", 2, ("Light", "Ranged", "Heavy"))],
        [("rush", 2, "Ranged"), ("random",)],
        [("random",), ("random",)],
        [("chaos", 0.3), ("chaos", 0.6)],
    ][matchup]
    run_pair(map_path, specs, seed=matchup, ticks=1500)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("conflict", [1, 2, 3])
def test_unit_tables_and_conflict_rules(version, conflict):
    run_pair("maps/8x8/basesWorkers8x8A.xml", [("chaos", 0.2), ("rush", 2, ("Heavy", "Ranged"))], seed=5,
             ticks=1500, version=version, conflict=conflict)


def test_batch_auto_reset_and_truncation():
    games = JaxBatchGame(["maps/8x8/basesWorkers8x8A.xml"], num_games=8, max_steps=50)
    games.reset()
    zero = np.zeros((8, 64, 7), dtype=np.int64)
    for _ in range(50):
        raw, over, trunc, winner = games.step(zero, zero)
    assert trunc.all() and not over.any()
    assert (games.episode_steps == 0).all() and (games.time == 0).all()


# ------------------------------------------------------------ hand-made corner cases
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3


def grid(width, height, *cells):
    """``(x, y, type, value)`` -> gridnet action array; ``value`` is the direction, ``(dir, unit type)`` or attack index."""
    a = np.zeros((width * height, 7), dtype=np.int64)
    for x, y, t, v in cells:
        row = a[x + y * width]
        row[0] = t
        if t == TYPE_PRODUCE:
            row[4], row[5] = v
        else:
            row[{1: 1, 2: 2, 3: 3, 5: 6}[t]] = v
    return a


def run_script(spec, script, ticks=60):
    """Play ``script`` (``{tick: (actions0, actions1)}``, waits otherwise) in both engines, comparing every tick."""
    utt = UnitTypeTable()
    ref = MicroRTSGame(map_spec=spec, utt=utt)
    ref.reset()
    fast = JaxBatchGame([spec], num_games=1, max_steps=10**9, auto_reset=False)
    fast.reset()
    zero = np.zeros((ref.num_cells, 7), dtype=np.int64)
    for tick in range(ticks):
        a0, a1 = script.get(tick, (zero, zero))
        raw, over = ref.step(a0, a1)
        fraw, fover, _, _ = fast.step(a0[None], a1[None])
        np.testing.assert_array_equal(fraw[0], raw, err_msg=f"rewards tick {tick}")
        assert_same_state(fast, ref, f"after tick {tick}")
        if over:
            break
    return ref


def spec(width, height, units, resources=(5, 5)):
    from microrts_pz.engine import MapSpec

    return MapSpec(width, height, [0] * (width * height), list(resources), units)


def test_harvest_race_for_the_last_resource():
    units = [(0, "Resource", -1, 2, 2, 1, 1), (1, "Worker", 0, 1, 2, 0, 1), (2, "Worker", 0, 3, 2, 0, 1),
             (3, "Worker", 1, 2, 1, 0, 1), (4, "Base", 1, 5, 5, 0, 10), (5, "Base", 0, 0, 5, 0, 10)]
    W = 6
    ref = run_script(spec(W, 6, units), {0: (grid(W, 6, (1, 2, 2, RIGHT), (3, 2, 2, LEFT)), grid(W, 6, (2, 1, 2, DOWN)))}, 25)
    assert sorted(u.resources for u in ref.gs.units if u.type.name == "Worker") == [0, 0, 1]


def test_moves_into_one_cell_and_wrapped_positions():
    W, H = 5, 5
    units = [(1, "Worker", 0, 1, 2, 0, 1), (2, "Worker", 0, 3, 2, 0, 1), (3, "Worker", 1, 2, 3, 0, 1),
             (4, "Worker", 0, 4, 0, 0, 1), (5, "Worker", 0, 1, 1, 0, 1), (6, "Base", 1, 4, 4, 0, 10)]
    a0 = grid(W, H, (1, 2, 1, RIGHT), (3, 2, 1, LEFT), (4, 0, 1, RIGHT), (1, 1, 1, LEFT))  # (4,0) right "claims" (0,1)
    a1 = grid(W, H, (2, 3, 1, UP))
    run_script(spec(W, H, units), {0: (a0, a1)}, 30)


def test_unaffordable_produce_blocks_the_rest_of_the_player():
    W, H = 6, 6
    units = [(1, "Base", 0, 1, 1, 0, 10), (2, "Worker", 0, 3, 3, 0, 1), (3, "Worker", 0, 4, 4, 0, 1),
             (4, "Base", 1, 5, 0, 0, 10)]
    a0 = grid(W, H, (3, 3, 4, (RIGHT, 1)), (4, 4, 1, LEFT), (1, 1, 4, (DOWN, 3)))  # a base costs 10, only 2 in stock
    run_script(spec(W, H, units, resources=(2, 2)), {0: (a0, grid(W, H)), 12: (grid(W, H, (1, 1, 4, (DOWN, 3))), grid(W, H))}, 80)


def test_attacks_and_targets_walking_away():
    W, H = 7, 7
    units = [(1, "Ranged", 0, 1, 3, 0, 1), (2, "Light", 0, 3, 2, 0, 4), (3, "Heavy", 1, 4, 3, 0, 4),
             (4, "Worker", 1, 3, 3, 0, 1), (5, "Base", 1, 6, 6, 0, 10), (6, "Base", 0, 0, 0, 0, 10)]
    center = 3
    at = lambda dx, dy: (center + dy) * 7 + (center + dx)  # noqa: E731
    script = {
        0: (grid(W, H, (3, 2, 5, at(0, 1))), grid(W, H, (3, 3, 1, LEFT), (4, 3, 1, UP))),
        5: (grid(W, H, (1, 3, 5, at(1, 0))), grid(W, H)),
    }
    run_script(spec(W, H, units), script, 60)


@pytest.mark.parametrize("map_path", ["maps/8x8/basesWorkers8x8A.xml", "maps/16x16/basesWorkers16x16.xml"])
def test_fast_forward_matches_the_env_option(map_path):
    max_steps = 400
    ref = MicroRTSGame(map_path, UnitTypeTable())
    ref.reset()
    fast = JaxBatchGame([map_path], num_games=1, max_steps=max_steps, auto_reset=False, fast_forward=True)
    fast.reset()
    bots = [WorkerRushBot(1), WorkerRushBot(1)]  # bases always busy, so whole ticks can be skipped
    steps = 0
    skipped = 0
    while steps < max_steps:
        actions = [np.asarray(b.act(ref, p)).reshape(ref.num_cells, -1) for p, b in enumerate(bots)]
        raw, over = ref.step(actions[0], actions[1])
        steps += 1
        ticks = 1
        while not over and steps < max_steps and not ref.needs_decision():  # as parallel_env(fast_forward=True)
            r, over = ref.step_without_decision()
            raw += r
            steps += 1
            ticks += 1
        fraw, fover, ftrunc, _ = fast.step(actions[0][None], actions[1][None])
        np.testing.assert_array_equal(fraw[0], raw)
        assert fast.last_ticks[0] == ticks and fover[0] == over and ftrunc[0] == (not over and steps >= max_steps)
        assert_same_state(fast, ref, f"at step {steps}")
        skipped += ticks - 1
        if over:
            break
    assert skipped > 0
