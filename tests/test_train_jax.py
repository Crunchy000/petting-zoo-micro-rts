"""The JAX trainer (``training/train_jax.py``) and its network (``training/jax_model.py``)."""

import os
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
import jax_model as P  # noqa: E402
import train_jax  # noqa: E402
from opponents import parse_weights  # noqa: E402

from microrts_pz.bots import WorkerRushBot  # noqa: E402
from microrts_pz.game import MicroRTSGame  # noqa: E402

DIMS = (6, 4, 4, 4, 4, 7, 49)


def test_slot_layout_follows_the_weights():
    names = train_jax.slot_layout(64, parse_weights("mixed:1,light:1,random:1,self:1"), keep_snapshots=4)
    assert len(names) == 64 and names.count("self") == 16
    assert names[:16] == ["self"] * 16 and names[-1] == "random"  # self-play first, random last
    assert {n: names.count(n) for n in ("mixed", "light", "random")} == {"mixed": 16, "light": 16, "random": 16}
    assert train_jax.slot_layout(8, parse_weights("self:1"), keep_snapshots=4) == ["self"] * 8
    assert "self" not in train_jax.slot_layout(8, parse_weights("rush:1"), keep_snapshots=4)


def _game_inputs(ticks=40):
    g = MicroRTSGame("maps/8x8/basesWorkers8x8A.xml")
    g.reset()
    bot = WorkerRushBot(2, "Light")
    for _ in range(ticks):
        g.step(bot.act(g, 0), bot.act(g, 1))
    return g, bot


def test_sampling_and_scoring_agree():
    g, bot = _game_inputs()
    params = P.init_params(jax.random.PRNGKey(0), g.num_channels, DIMS, width=16)
    obs = jnp.asarray(g.observation(0)[None])
    mask = jnp.asarray(g.action_mask(0)[None]).astype(bool)
    logits, _ = P.apply(params, obs)
    act, logp = P.sample(jax.random.PRNGKey(1), logits, mask, DIMS)
    source = np.asarray(mask[0, :, :6].any(-1))
    assert (np.asarray(act[0])[~source] == 0).all()
    lp2, ent = P.evaluate(logits, mask, act, DIMS)
    np.testing.assert_allclose(np.asarray(lp2), np.asarray(logp), rtol=1e-5, atol=1e-5)
    assert float(ent[0]) > 0
    packed = P.pack_mask(mask)
    assert packed.dtype == jnp.uint8 and packed.shape[-1] == 10
    assert (P.unpack_mask(packed, mask.shape[-1]) == mask).all()


def test_matches_the_pytorch_policy():
    torch = pytest.importorskip("torch")
    from model import GridnetPolicy, masked_policy

    g, bot = _game_inputs()
    net = GridnetPolicy(in_channels=g.num_channels, width=16)
    params = P.from_torch(net.state_dict())
    for k, v in P.to_torch(params).items():
        np.testing.assert_array_equal(v, net.state_dict()[k].numpy())
    obs, mask = g.observation(0)[None], g.action_mask(0)[None].astype(bool)
    act = bot.act(g, 0)[None]
    with torch.no_grad():
        lt, vt = net(torch.as_tensor(obs))
    _, lpt, et, _ = masked_policy(lt, torch.as_tensor(mask), torch.as_tensor(act).long())
    lj, vj = P.apply(params, jnp.asarray(obs))
    lpj, ej = P.evaluate(lj, jnp.asarray(mask), jnp.asarray(act), DIMS)
    np.testing.assert_allclose(np.asarray(lj), lt.detach().numpy(), atol=1e-4)
    np.testing.assert_allclose(float(vj[0]), float(vt[0]), atol=1e-4)
    np.testing.assert_allclose(float(lpj[0]), float(lpt[0]), atol=1e-4)
    np.testing.assert_allclose(float(ej[0]), float(et[0]), atol=1e-4)


def test_trains_saves_resumes_and_plays(tmp_path):
    out = str(tmp_path / "run")
    common = ["--map", "maps/8x8/basesWorkers8x8A.xml", "--out", out, "--num-envs", "16", "--num-steps", "8",
              "--minibatch-size", "64", "--width", "8", "--keep-snapshots", "2", "--snapshot-every", "1",
              "--opponents", "self:1,mixed:1,random:1"]
    train_jax.main(common + ["--total-steps", "256", "--bc-steps", "128"])
    ckpt = P.load_checkpoint(os.path.join(out, "latest.pkl"))
    assert ckpt["global_step"] == 256 and ckpt["n_snapshots"] == 1
    rows = open(os.path.join(out, "metrics.csv")).read().splitlines()
    assert [r.split(",")[1] for r in rows[1:]] == ["bc", "ppo"]
    train_jax.main(common + ["--total-steps", "384", "--resume"])
    assert P.load_checkpoint(os.path.join(out, "latest.pkl"))["global_step"] == 384

    import play

    play.main([os.path.join(out, "latest.pkl"), "--map", "maps/8x8/basesWorkers8x8A.xml", "--opponent", "rush",
               "--games", "1", "--max-steps", "30"])
    if os.path.exists(os.path.join(out, "latest.pt")):  # written when PyTorch is installed
        play.main([os.path.join(out, "latest.pt"), "--map", "maps/8x8/basesWorkers8x8A.xml", "--opponent",
                   os.path.join(out, "latest.pkl"), "--games", "1", "--max-steps", "30"])


def test_rotated_side_and_frozen_opponents(tmp_path):
    first = str(tmp_path / "first")
    common = ["--map", "maps/8x8/basesWorkers8x8A.xml", "--num-envs", "16", "--num-steps", "8",
              "--minibatch-size", "64", "--width", "8", "--keep-snapshots", "2", "--opponents", "self:1,rush:1"]
    train_jax.main(common + ["--out", first, "--total-steps", "128"])
    out = str(tmp_path / "run")
    frozen = os.path.join(first, "latest.pkl")
    train_jax.main(common + ["--out", out, "--total-steps", "256", "--bc-steps", "128", "--rotate-side", "0",
                             "--frozen", frozen])
    ckpt = P.load_checkpoint(os.path.join(out, "latest.pkl"))
    assert ckpt["rotate_side"] == 0
    assert jax.tree.leaves(ckpt["snapshots"])[0].shape[0] == 3  # two ring slots and the frozen checkpoint
    train_jax.main(common + ["--out", out, "--total-steps", "384", "--rotate-side", "0", "--frozen", frozen, "--resume"])

    bot = P.JaxPolicyBot(os.path.join(out, "latest.pkl"))
    assert bot.rotate_side == 0
