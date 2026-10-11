# SPDX-License-Identifier: GPL-3.0-only
"""PettingZoo environments for MicroRTS (Parallel API natively, AEC via conversion)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from .engine import load_map
from .game import REWARD_NAMES, MicroRTSGame
from .spaces import GridnetActionSpace
from .unit_types import MOVE_CONFLICT_RESOLUTION_CANCEL_BOTH, VERSION_ORIGINAL, UnitTypeTable

DEFAULT_MAP = "maps/16x16/basesWorkers16x16.xml"
#: Weights for ``[win_loss, resource_gather, produce_worker, produce_building, attack, produce_combat_unit]``
#: (the shaped reward used by CleanRL's gym-microrts PPO baselines).
DEFAULT_REWARD_WEIGHT = (10.0, 1.0, 1.0, 0.2, 1.0, 4.0)


class MicroRTSParallelEnv(ParallelEnv):
    """Two-player, simultaneous-move MicroRTS.

    Agents ``"player_0"`` and ``"player_1"`` act every tick using the *gridnet*
    action space of Gym-MicroRTS: one 7-component action per map cell, read only
    where the agent has an idle unit.

    Observation (per agent) is a dict:

    * ``"observation"``: ``int8`` one-hot planes of shape ``(H, W, C)`` with
      ``C = 5 hp + 5 carried/remaining resources + 3 owner (none/self/enemy) +
      (n_unit_types + 1) unit type + 6 current action + 2 terrain`` (``= 29``),
      plus 2 visibility planes when ``partial_obs=True``. Ownership is
      ego-centric: "self" always means the observing agent.
    * ``"action_mask"``: flat ``int8`` vector of length ``H * W * 78``; for each
      cell the valid values of each action component, concatenated. All zero
      for cells without an idle unit of the agent.

    Rewards are ``reward_weight @ raw_rewards`` where the six raw reward
    components (see ``infos[agent]["raw_rewards"]``) are win/loss, resources
    gathered, workers produced, buildings produced, attacks issued against
    enemies and combat units produced, as in the reference engine.

    An episode terminates when one side has no units left and is truncated
    after ``max_steps`` ticks.
    """

    metadata = {
        "name": "microrts_v0",
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 30,
        "is_parallelizable": True,
    }

    def __init__(
        self,
        map_path: str = DEFAULT_MAP,
        max_steps: int = 2000,
        reward_weight: Sequence[float] = DEFAULT_REWARD_WEIGHT,
        partial_obs: bool = False,
        unit_type_table_version: int = VERSION_ORIGINAL,
        move_conflict_resolution: int = MOVE_CONFLICT_RESOLUTION_CANCEL_BOTH,
        render_mode: str | None = None,
        record_replay: bool = False,
        viewer_port: int = 0,
        open_browser: bool = True,
    ):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}, got {render_mode!r}")
        self.map_path = map_path
        self.max_steps = int(max_steps)
        self.reward_weight = np.asarray(reward_weight, dtype=np.float64)
        if self.reward_weight.shape != (len(REWARD_NAMES),):
            raise ValueError(f"reward_weight must have {len(REWARD_NAMES)} entries")
        self.partial_obs = partial_obs
        self.render_mode = render_mode
        self.record_replay = record_replay or render_mode == "human"
        self._viewer_port = viewer_port
        self._open_browser = open_browser
        self._viewer = None

        self.utt = UnitTypeTable(unit_type_table_version, move_conflict_resolution)
        self._rng = np.random.default_rng()
        self.game = MicroRTSGame(map_path, self.utt, partial_obs, self._rng, map_spec=load_map(map_path))
        g = self.game
        self.height, self.width = g.height, g.width

        self.possible_agents = ["player_0", "player_1"]
        self.agents: list[str] = []
        self._agent_ids = {a: i for i, a in enumerate(self.possible_agents)}

        obs_space = spaces.Dict(
            {
                "observation": spaces.Box(0, 1, shape=(g.height, g.width, g.num_channels), dtype=np.int8),
                "action_mask": spaces.Box(0, 1, shape=(g.num_cells * g.mask_size,), dtype=np.int8),
            }
        )
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: GridnetActionSpace(g.num_cells, g.action_dims) for a in self.possible_agents}
        self.state_space = spaces.Box(0, 1, shape=(g.height, g.width, sum(g.num_planes)), dtype=np.int8)

        self.num_steps = 0
        self.replay_frames: list[dict] = []

    # --------------------------------------------------------------- spaces
    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> GridnetActionSpace:
        return self.action_spaces[agent]

    # ------------------------------------------------------------------ api
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            for i, a in enumerate(self.possible_agents):
                self.action_spaces[a].seed(seed + i)
        self.game.rng = self._rng
        self.game.reset()
        self.agents = self.possible_agents[:]
        self.num_steps = 0
        self.replay_frames = []
        if self.record_replay:
            self._record()
        if self.render_mode == "human":
            self.render()
        observations = {a: self._observe(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions: dict[str, Any]):
        if not self.agents:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        grids = []
        for agent in self.possible_agents:
            act = actions.get(agent)
            if act is None:
                grids.append(np.zeros((self.game.num_cells, len(self.game.action_dims)), dtype=np.int64))
            else:
                grids.append(np.asarray(act, dtype=np.int64).reshape(self.game.num_cells, -1))
        raw, gameover = self.game.step(grids[0], grids[1])
        self.num_steps += 1
        truncated = not gameover and self.num_steps >= self.max_steps
        winner = self.game.winner if gameover else None

        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        for agent in self.agents:
            i = self._agent_ids[agent]
            observations[agent] = self._observe(agent)
            rewards[agent] = float(self.reward_weight @ raw[i])
            terminations[agent] = gameover
            truncations[agent] = truncated
            info = {"raw_rewards": raw[i].copy(), "time": self.game.gs.time}
            if gameover:
                info["winner"] = None if winner < 0 else self.possible_agents[winner]
            infos[agent] = info

        if self.record_replay:
            self._record()
        if self.render_mode == "human":
            self.render()
        if gameover or truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def _observe(self, agent: str) -> dict[str, np.ndarray]:
        i = self._agent_ids[agent]
        return {
            "observation": self.game.observation(i),
            "action_mask": self.game.action_mask(i).reshape(-1),
        }

    def state(self) -> np.ndarray:
        """Fully observed one-hot planes from player 0's point of view."""
        partial = self.game.partial_obs
        self.game.partial_obs = False
        try:
            obs = self.game.observation(0)
        finally:
            self.game.partial_obs = partial
        return obs[:, :, : self.state_space.shape[-1]]

    # -------------------------------------------------------------- render
    def _record(self) -> None:
        from .render import encode_frame

        self.replay_frames.append(encode_frame(self.game.gs))

    def replay_meta(self) -> dict:
        from .render import encode_meta

        return encode_meta(self.game, self.map_path, self.possible_agents)

    def save_replay(self, path: str, title: str | None = None) -> str:
        """Write the recorded episode as a self-contained HTML player (needs ``record_replay=True``)."""
        from .render import save_replay_html

        if not self.replay_frames:
            raise RuntimeError("no frames recorded; create the env with record_replay=True (or render_mode='human')")
        return save_replay_html(self.replay_meta(), self.replay_frames, path, title=title)

    def render(self):
        if self.render_mode is None:
            import warnings

            warnings.warn("render() called without a render_mode; pass render_mode= to the constructor")
            return None
        if self.game.gs is None:
            return None
        if self.render_mode == "rgb_array":
            from .render import render_rgb

            return render_rgb(self.game.gs)
        if self.render_mode == "ansi":
            from .render import render_ansi

            return render_ansi(self.game.gs)
        # human: stream frames to the browser viewer
        from .render import LiveViewer

        if self._viewer is None:
            self._viewer = LiveViewer(port=self._viewer_port, open_browser=self._open_browser)
        self._viewer.send(self.replay_meta(), self.replay_frames[-1] if self.replay_frames else None)
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


def raw_env(**kwargs):
    """Unwrapped AEC environment, converted from the native parallel one."""
    return parallel_to_aec(MicroRTSParallelEnv(**kwargs))


def parallel_env(**kwargs) -> MicroRTSParallelEnv:
    return MicroRTSParallelEnv(**kwargs)


def env(**kwargs):
    """AEC version: players take turns choosing actions, the tick runs after the last one."""
    return wrappers.OrderEnforcingWrapper(raw_env(**kwargs))


__all__ = ["MicroRTSParallelEnv", "env", "parallel_env", "raw_env", "DEFAULT_MAP", "DEFAULT_REWARD_WEIGHT"]
