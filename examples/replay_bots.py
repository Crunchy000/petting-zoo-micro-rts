"""Play a worker rush against a light-unit rush (or another bot) and save an HTML replay with sound.

    python examples/replay_bots.py --map maps/16x16/basesWorkers16x16.xml --out replay.html
    python examples/replay_bots.py --live          # watch in the browser while it runs
"""

import argparse

from microrts_pz import microrts_v0
from microrts_pz.bots import RandomBot, WorkerRushBot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="maps/16x16/basesWorkers16x16.xml")
    ap.add_argument("--out", default="replay.html")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--opponent", choices=["light", "rush", "random"], default="light")
    ap.add_argument("--live", action="store_true", help="stream to the browser viewer instead of writing a file")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = microrts_v0.parallel_env(
        map_path=args.map,
        max_steps=args.max_steps,
        render_mode="human" if args.live else None,
        record_replay=True,
    )
    bots = {"player_0": WorkerRushBot(), "player_1": {
        "light": WorkerRushBot(harvesters=2, army="Light"),
        "rush": WorkerRushBot(),
        "random": RandomBot(args.seed),
    }[args.opponent]}
    env.reset(seed=args.seed)
    game = env.unwrapped.game
    if args.live:
        import time
    while env.agents:
        actions = {a: bots[a].act(game, i) for i, a in enumerate(env.possible_agents)}
        _, rewards, terms, truncs, infos = env.step(actions)
        if args.live:
            time.sleep(1 / 40)
    print(f"finished at tick {game.gs.time}, winner: {infos['player_0'].get('winner', 'none (truncated)')}")
    if not args.live:
        print("replay written to", env.save_replay(args.out))
    else:
        input("press enter to close the viewer")
    env.close()


if __name__ == "__main__":
    main()
