import json
import urllib.request

import numpy as np

from microrts_pz import microrts_v0
from microrts_pz.bots import RandomBot, WorkerRushBot
from microrts_pz.render import DATA_MARKER, LiveViewer


def play(env, steps=None):
    game = env.unwrapped.game
    bots = [WorkerRushBot(), WorkerRushBot(harvesters=2, army="Light")]
    n = 0
    while env.agents and (steps is None or n < steps):
        env.step({a: bots[i].act(game, i) for i, a in enumerate(env.possible_agents)})
        n += 1
    return n


def test_replay_html_is_self_contained(tmp_path):
    env = microrts_v0.parallel_env(map_path="maps/8x8/basesWorkers8x8A.xml", max_steps=400, record_replay=True)
    env.reset(seed=0)
    n = play(env)
    out = env.save_replay(str(tmp_path / "replay.html"), title="test")
    html = open(out, encoding="utf-8").read()
    assert DATA_MARKER not in html
    start = html.index("window.MICRORTS_REPLAY = ") + len("window.MICRORTS_REPLAY = ")
    data = json.loads(html[start : html.index(";\n</script>", start)])
    assert len(data["frames"]) == n + 1
    assert data["meta"]["width"] == 8 and len(data["meta"]["terrain"]) == 64
    assert "<script src=" not in html  # no external assets


def test_rgb_and_ansi():
    env = microrts_v0.parallel_env(map_path="maps/16x16/basesWorkers16x16.xml", render_mode="rgb_array")
    env.reset(seed=0)
    img = env.render()
    assert img.shape == (16 * 32, 16 * 32, 3) and img.dtype == np.uint8
    assert len(np.unique(img.reshape(-1, 3), axis=0)) > 4
    env = microrts_v0.parallel_env(map_path="maps/8x8/basesWorkers8x8A.xml", render_mode="ansi")
    env.reset(seed=0)
    text = env.render()
    assert "B" in text and "b" in text and "$" in text


def test_live_viewer_streams_frames():
    viewer = LiveViewer(port=0, open_browser=False)
    try:
        page = urllib.request.urlopen(viewer.url, timeout=5).read().decode()
        assert "window.MICRORTS_LIVE = true;" in page
        viewer.send({"map": "x", "width": 1, "height": 1, "terrain": "0", "players": [], "unitTypes": []}, {"t": 0, "r": [0, 0], "u": []})
        viewer.send({}, {"t": 1, "r": [1, 0], "u": []})
        stream = urllib.request.urlopen(viewer.url + "events", timeout=5)
        lines = [stream.readline().decode() for _ in range(6)]
        assert lines[0].startswith("event: reset")
        frames = [json.loads(line[6:]) for line in lines if line.startswith("data: {\"t\"")]
        assert [f["t"] for f in frames] == [0, 1]
        stream.close()
    finally:
        viewer.close()


def test_bots_produce_valid_games():
    env = microrts_v0.parallel_env(map_path="maps/16x16/basesWorkers16x16.xml", max_steps=1500)
    env.reset(seed=0)
    game = env.unwrapped.game
    bots = [WorkerRushBot(), RandomBot(0)]
    infos = {}
    while env.agents:
        _, _, _, _, infos = env.step({a: bots[i].act(game, i) for i, a in enumerate(env.possible_agents)})
    assert infos["player_0"].get("winner") == "player_0"
