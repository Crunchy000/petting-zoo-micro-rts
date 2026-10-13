# SPDX-License-Identifier: GPL-3.0-only
"""Evaluate a trained checkpoint against the scripted bots and save HTML replays.

    python training/play.py runs/16x16/latest.pt --opponent mixed --games 20
    python training/play.py runs/16x16/latest.pt --opponent mixed --record replays/ --record-games 3

Pass a second checkpoint as ``--opponent path/to/other.pt`` for agent-vs-agent games.
Checkpoints can be PyTorch ``.pt`` files (``train_selfplay.py``, or the
``latest.pt`` that ``train_jax.py`` exports) or ``train_jax.py``'s ``.pkl``
files, which play through JAX and need no PyTorch.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opponents import BOTS  # noqa: E402

from microrts_pz import microrts_v0  # noqa: E402


def load_agent(path: str, device: str, deterministic: bool = False, seed: int = 0, rotate_side="auto"):
    """A bot that plays the checkpoint at ``path`` (PyTorch for ``.pt``, JAX for ``.pkl``)."""
    if path.endswith(".pkl"):
        from jax_model import JaxPolicyBot

        return JaxPolicyBot(path, deterministic=deterministic, seed=seed, rotate_side=rotate_side)
    from model import PolicyBot

    return PolicyBot(path, device=device, deterministic=deterministic, seed=seed, rotate_side=rotate_side)


def make_opponent(name: str, seed: int, device: str):
    if name.endswith((".pt", ".pkl")):
        return load_agent(name, device, seed=seed)
    return BOTS[name](seed)


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--map", default="maps/16x16/basesWorkers16x16.xml")
    ap.add_argument("--opponent", default="mixed", help=f"one of {sorted(BOTS)} or a .pt/.pkl checkpoint")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--deterministic", action="store_true", help="take the most likely action instead of sampling")
    ap.add_argument("--record", help="directory for HTML replays")
    ap.add_argument("--record-games", type=int, default=2, help="how many replays to keep (wins first, then the most unit types)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rotate-side", choices=("auto", "0", "1", "none"), default="auto",
                    help="turn this side's board 180 degrees before the agent sees it (orient.py); "
                         "auto = what the checkpoint was trained with")
    ap.add_argument("--device", default=None, help="PyTorch device for .pt checkpoints (default: cuda when available)")
    args = ap.parse_args(argv)

    args.device = args.device or default_device()
    rot = int(args.rotate_side) if args.rotate_side.isdigit() else {"auto": "auto", "none": None}[args.rotate_side]
    agent = load_agent(args.checkpoint, args.device, deterministic=args.deterministic, seed=args.seed, rotate_side=rot)
    env = microrts_v0.parallel_env(map_path=args.map, max_steps=args.max_steps, record_replay=bool(args.record))
    results, kept = [], []
    for g in range(args.games):
        side = g % 2  # alternate sides
        opp = make_opponent(args.opponent, args.seed + g, args.device)
        env.reset(seed=args.seed + g)
        game = env.unwrapped.game
        players = {side: agent, 1 - side: opp}
        built = set()  # unit and structure types the agent owned at some point
        while env.agents:
            actions = {a: np.asarray(players[i].act(game, i)).reshape(-1) for i, a in enumerate(env.possible_agents)}
            _, _, _, _, infos = env.step(actions)
            built.update(u.type.name for u in game.gs.units if u.player == side)
        winner = infos["player_0"].get("winner")
        me = env.possible_agents[side]
        result = 0 if winner is None else (1 if winner == me else -1)
        results.append(result)
        label = {1: "win", 0: "draw", -1: "loss"}[result]
        print(f"game {g + 1}: agent as {me} vs {args.opponent}: {label} at tick {game.gs.time}, "
              f"built {', '.join(sorted(built))}")
        if args.record:
            kept.append((-result, -len(built), g, side, label, list(env.replay_frames), env.replay_meta()))
            kept.sort(key=lambda k: k[:3])
            del kept[args.record_games :]

    r = np.array(results)
    print(f"\n{len(r)} games vs {args.opponent}: {np.mean(r == 1):.0%} wins, {np.mean(r == 0):.0%} draws, {np.mean(r == -1):.0%} losses")
    if args.record:
        from microrts_pz.render import save_replay_html

        os.makedirs(args.record, exist_ok=True)
        opp = os.path.splitext(os.path.basename(args.opponent))[0]
        for _, _, g, side, label, frames, meta in kept:
            path = os.path.join(args.record, f"agent_vs_{opp}_game{g + 1}_{label}.html")
            save_replay_html(meta, frames, path, title=f"Trained agent (player {side}) vs {opp}: {label}")
            print("replay written to", path)
    env.close()


if __name__ == "__main__":
    main()
