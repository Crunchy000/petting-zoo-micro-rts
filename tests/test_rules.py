"""Hand-checked MicroRTS rules on tiny scenarios."""

import numpy as np

from microrts_pz.engine import (
    TYPE_ATTACK_LOCATION,
    TYPE_HARVEST,
    TYPE_MOVE,
    TYPE_PRODUCE,
    TYPE_RETURN,
    MapSpec,
    list_maps,
    load_map,
)
from microrts_pz.game import MicroRTSGame
from microrts_pz.unit_types import UnitTypeTable

UTT = UnitTypeTable()
RIGHT, DOWN, LEFT, UP = 1, 2, 3, 0


def make_game(width, height, units, resources=(5, 5), terrain=None):
    spec = MapSpec(width, height, terrain or [0] * (width * height), list(resources), units)
    g = MicroRTSGame(map_spec=spec, utt=UTT)
    g.reset()
    return g


def act(g, *cells):
    """cells: (x, y, type, param) -> action grid."""
    a = np.zeros((g.num_cells, 7), dtype=np.int64)
    for x, y, t, p in cells:
        row = a[x + y * g.width]
        row[0] = t
        if t == TYPE_MOVE:
            row[1] = p
        elif t == TYPE_HARVEST:
            row[2] = p
        elif t == TYPE_RETURN:
            row[3] = p
        elif t == TYPE_PRODUCE:
            row[4], row[5] = p
        elif t == TYPE_ATTACK_LOCATION:
            row[6] = p
    return a


def run(g, a0=None, a1=None, ticks=1):
    out = None
    for _ in range(ticks):
        z = np.zeros((g.num_cells, 7), dtype=np.int64)
        out = g.step(z if a0 is None else a0, z if a1 is None else a1)
        a0 = a1 = None
    return out


def worker(uid, player, x, y, res=0):
    return (uid, "Worker", player, x, y, res, 1)


def test_all_bundled_maps_load():
    maps = list_maps()
    assert len(maps) > 50
    for m in maps:
        spec = load_map(m)
        gs = spec.new_game(UTT)
        assert gs.width * gs.height == len(spec.terrain)


def test_move_takes_move_time():
    g = make_game(4, 1, [worker(0, 0, 0, 0), worker(1, 1, 3, 0)])
    run(g, act(g, (0, 0, TYPE_MOVE, RIGHT)))
    u = g.gs.units[0]
    for _ in range(UTT["Worker"].move_time - 2):
        run(g)
        assert (u.x, u.y) == (0, 0)
    run(g)
    assert (u.x, u.y) == (1, 0)


def test_harvest_and_return_cycle():
    units = [(0, "Resource", -1, 0, 0, 3, 1), (1, "Base", 0, 2, 0, 0, 10), worker(2, 0, 1, 0), worker(3, 1, 3, 3)]
    g = make_game(4, 4, units, resources=(0, 0))
    raw, _ = g.step(act(g, (1, 0, TYPE_HARVEST, LEFT)), np.zeros((16, 7), int))
    assert raw[0][1] == 1  # resource-gather reward is paid when the action is issued
    run(g, ticks=UTT["Worker"].harvest_time)
    w = g.gs.units[2]
    assert w.resources == 1 and g.gs.units[0].resources == 2
    run(g, act(g, (1, 0, TYPE_RETURN, RIGHT)), ticks=UTT["Worker"].move_time + 1)
    assert g.gs.player_resources[0] == 1 and w.resources == 0


def test_produce_spends_resources_on_completion():
    units = [(0, "Base", 0, 0, 0, 0, 10), worker(1, 1, 3, 3)]
    g = make_game(4, 4, units, resources=(1, 0))
    produce = act(g, (0, 0, TYPE_PRODUCE, (RIGHT, UTT["Worker"].id)))
    raw, _ = g.step(produce, np.zeros((16, 7), int))
    assert raw[0][2] == 1
    assert g.gs.player_resources[0] == 1  # paid at completion, only reserved now
    run(g, ticks=UTT["Worker"].produce_time)
    assert g.gs.player_resources[0] == 0
    assert any(u.type.name == "Worker" and u.player == 0 and (u.x, u.y) == (1, 0) for u in g.gs.units)


def test_attack_kills_and_ends_game():
    g = make_game(3, 1, [worker(0, 0, 0, 0), worker(1, 1, 1, 0)])
    center = g.attack_grid // 2
    target = center * g.attack_grid + center + 1  # one cell to the right
    raw, over = g.step(act(g, (0, 0, TYPE_ATTACK_LOCATION, target)), np.zeros((3, 7), int))
    assert raw[0][4] == 1 and not over
    raw, over = run(g, ticks=UTT["Worker"].attack_time)
    assert over and g.gs.winner() == 0
    assert raw[0][0] == 1 and raw[1][0] == -1


def test_simultaneous_attacks_both_land():
    """A unit killed during a tick still finishes its own action that tick (Java engine quirk)."""
    g = make_game(2, 1, [worker(0, 0, 0, 0), worker(1, 1, 1, 0)])
    c = g.attack_grid // 2
    a0 = act(g, (0, 0, TYPE_ATTACK_LOCATION, c * g.attack_grid + c + 1))
    a1 = act(g, (1, 0, TYPE_ATTACK_LOCATION, c * g.attack_grid + c - 1))
    g.step(a0, a1)
    _, over = run(g, ticks=UTT["Worker"].attack_time)
    assert over and g.gs.winner() == -1 and not g.gs.units


def test_move_conflict_player0_has_priority():
    g = make_game(3, 1, [worker(0, 0, 0, 0), worker(1, 1, 2, 0)])
    g.step(act(g, (0, 0, TYPE_MOVE, RIGHT)), act(g, (2, 0, TYPE_MOVE, LEFT)))
    p0, p1 = g.gs.units
    assert g.gs.assignments[p0.id].action.type == TYPE_MOVE
    assert p1.id not in g.gs.assignments  # player 1's move was dropped for a 1-tick wait, already over
    run(g, ticks=UTT["Worker"].move_time)
    assert (p0.x, p1.x) == (1, 2)


def test_illegal_action_becomes_wait_of_same_duration():
    g = make_game(3, 1, [worker(0, 0, 0, 0), worker(1, 1, 2, 0)])
    g.step(act(g, (0, 0, TYPE_MOVE, UP)), np.zeros((3, 7), int))  # off the map
    a = g.gs.assignments[0]
    assert a.action.type == 0 and a.action.parameter == UTT["Worker"].move_time


def test_masks_match_legal_actions():
    g = MicroRTSGame("maps/8x8/basesWorkers8x8A.xml", UTT)
    g.reset()
    mask = g.action_mask(0)
    src = g.source_unit_mask(0)
    assert set(np.flatnonzero(mask.any(1))) == set(np.flatnonzero(src))
    for u in g.gs.units:
        if u.player == 0:
            row = mask[u.x + u.y * g.width]
            legal_types = {a.type for a in g.gs.unit_actions(u)}
            assert set(np.flatnonzero(row[:6])) == legal_types


def test_partial_observability_hides_distant_enemies():
    units = [worker(0, 0, 0, 0), worker(1, 1, 9, 9), (2, "Resource", -1, 9, 0, 5, 1)]
    g = MicroRTSGame(map_spec=MapSpec(10, 10, [0] * 100, [0, 0], units), utt=UTT, partial_obs=True)
    g.reset()
    raw = g.raw_observation(0)
    assert raw[3, 9, 9] == 0 and raw[3, 0, 9] == 0  # enemy and far resource hidden
    assert raw[3, 0, 0] == UTT["Worker"].id + 1
    assert raw[6, 0, 0] == 1 and raw[6, 9, 9] == 0  # own visibility plane
    assert g.observation(0).shape == (10, 10, 31)


def test_terrain_blocks_movement():
    spec = MapSpec(3, 1, [0, 1, 0], [0, 0], [worker(0, 0, 0, 0), worker(1, 1, 2, 0)])
    g = MicroRTSGame(map_spec=spec, utt=UTT)
    g.reset()
    assert not any(a.type == TYPE_MOVE for a in g.gs.unit_actions(g.gs.units[0]))
