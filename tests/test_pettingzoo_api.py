"""PettingZoo's own conformance tests, run against the Parallel and AEC versions."""

import functools

import numpy as np
import pytest
from pettingzoo.test import api_test, parallel_api_test, parallel_seed_test, render_test, seed_test

from microrts_pz import microrts_v0

MAPS = [
    "maps/4x4/base4x4.xml",
    "maps/8x8/basesWorkers8x8A.xml",
    "maps/10x10/basesWorkers10x10.xml",
    "maps/16x16/basesWorkers16x16.xml",
    "maps/8x8/basesWorkers8x8Obstacle.xml",
]


@pytest.mark.parametrize("map_path", MAPS)
@pytest.mark.parametrize("partial_obs", [False, True])
def test_parallel_api(map_path, partial_obs):
    env = microrts_v0.parallel_env(map_path=map_path, max_steps=300, partial_obs=partial_obs)
    parallel_api_test(env, num_cycles=400)


@pytest.mark.parametrize("map_path", MAPS[:3])
def test_aec_api(map_path):
    api_test(microrts_v0.env(map_path=map_path, max_steps=300), num_cycles=400)


def test_seeding():
    kwargs = dict(map_path="maps/8x8/basesWorkers8x8A.xml", max_steps=200)
    seed_test(functools.partial(microrts_v0.env, **kwargs), num_cycles=200)
    parallel_seed_test(functools.partial(microrts_v0.parallel_env, **kwargs), num_cycles=200)


def test_seeding_non_deterministic_unit_table():
    """Version 3 rolls damage at random; a seed must make it reproducible."""
    kwargs = dict(map_path="maps/8x8/melee8x8Mixed6.xml", max_steps=300, unit_type_table_version=3)
    parallel_seed_test(functools.partial(microrts_v0.parallel_env, **kwargs), num_cycles=300)


def test_render_modes():
    render_test(lambda render_mode=None: microrts_v0.env(map_path="maps/8x8/basesWorkers8x8A.xml", render_mode=render_mode))


def test_spaces_and_mask_layout():
    env = microrts_v0.parallel_env(map_path="maps/16x16/basesWorkers16x16.xml")
    obs, infos = env.reset(seed=1)
    space = env.action_space("player_0")
    assert space.shape == (16 * 16 * 7,)
    assert list(space.nvec[:7]) == [6, 4, 4, 4, 4, 7, 49]
    o = obs["player_0"]
    assert o["observation"].shape == (16, 16, 29) and o["observation"].dtype == np.int8
    assert o["action_mask"].shape == (16 * 16 * 78,) and o["action_mask"].dtype == np.int8
    # every cell has exactly one active plane per feature group
    planes = [5, 5, 3, 8, 6, 2]
    offs = np.cumsum([0] + planes)
    for a, b in zip(offs[:-1], offs[1:]):
        assert (o["observation"][:, :, a:b].sum(-1) == 1).all()
    assert env.state().shape == env.state_space.shape


def test_masked_sampling_only_picks_valid_components():
    env = microrts_v0.parallel_env(map_path="maps/16x16/basesWorkers16x16.xml")
    obs, _ = env.reset(seed=3)
    rng_space = env.action_space("player_0")
    for _ in range(50):
        actions = {}
        for agent in env.agents:
            mask = obs[agent]["action_mask"].reshape(-1, 78)
            a = env.action_space(agent).sample(obs[agent]["action_mask"]).reshape(-1, 7)
            offs = rng_space.mask_offsets
            for cell in np.flatnonzero(mask.any(1)):
                for k in range(7):
                    sub = mask[cell, offs[k] : offs[k + 1]]
                    if sub.any():  # components with no valid value are sampled as 0
                        assert sub[a[cell, k]] == 1
            actions[agent] = a.reshape(-1)
        obs, *_ = env.step(actions)


def test_missing_actions_are_noops_and_episode_ends():
    env = microrts_v0.parallel_env(map_path="maps/4x4/base4x4.xml", max_steps=5)
    env.reset(seed=0)
    for t in range(5):
        obs, rew, term, trunc, info = env.step({})
    assert all(trunc.values()) and not any(term.values())
    assert env.agents == []
    with pytest.raises(RuntimeError):
        env.step({})
