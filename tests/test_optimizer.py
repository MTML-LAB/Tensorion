import copy
import io
import math

import pytest
import torch
from torch import nn

from tensorion import Tensorion
from tensorion.orthogonalize import ORTHOGONALIZERS, newton_schulz
from tensorion.unfolding import STRATEGIES


def _net():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3),
        nn.ReLU(),
        nn.Conv2d(8, 4, (3, 5)),
        nn.Flatten(),
        nn.Linear(4 * 6 * 6, 10),
    )


def _train(model, optimizer, steps=3):
    torch.manual_seed(1)
    for _ in range(steps):
        optimizer.zero_grad()
        model(torch.randn(4, 3, 10, 12)).square().mean().backward()
        optimizer.step()


def test_defaults_follow_paper_algorithm():
    """Algorithm 1 of the paper: offline (max-rank) unfolding fixed by the shape, Newton-Schulz."""
    w = nn.Parameter(torch.randn(16, 8, 3, 3))
    w0 = w.detach().clone()
    w.grad = torch.randn_like(w)
    lr, wd = 0.01, 0.1
    opt = Tensorion([w], lr=lr, weight_decay=wd)
    assert (opt.defaults["unfolding"], opt.defaults["orthogonalizer"]) == ("offline", "newton_schulz")
    opt.step()

    # (16, 8, 3, 3): rows {0, 2} give a 48 x 24 matricization, the largest possible rank.
    G = w.grad.permute(0, 2, 1, 3).reshape(48, 24)
    update = newton_schulz(G).reshape(16, 3, 8, 3).permute(0, 2, 1, 3)
    expected = w0 * (1 - lr * wd) - lr * 0.2 * math.sqrt(48) * update
    torch.testing.assert_close(w.detach(), expected)


def test_fixed_mode_svd_update_matches_reference():
    """One step on a conv kernel: momentum, fixed mode-2 unfolding, exact polar factor, AdamW-matched lr."""
    w = nn.Parameter(torch.randn(16, 8, 3, 3))
    w0 = w.detach().clone()
    w.grad = torch.randn_like(w)
    lr, wd = 0.01, 0.1
    Tensorion([("w", w)], lr=lr, weight_decay=wd, unfolding={"w": 2}, orthogonalizer="svd").step()

    G = w.grad.permute(2, 0, 1, 3).reshape(3, -1)
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    update = (U @ Vh).reshape(3, 16, 8, 3).permute(1, 2, 0, 3)
    expected = w0 * (1 - lr * wd) - lr * 0.2 * math.sqrt(G.shape[1]) * update
    torch.testing.assert_close(w.detach(), expected)


def test_matrix_update_is_muon():
    w = nn.Parameter(torch.randn(12, 20))
    w0 = w.detach().clone()
    w.grad = torch.randn_like(w)
    Tensorion([w], lr=0.1, weight_decay=0.0, adjust_lr_fn=None, orthogonalizer="svd").step()
    U, _, Vh = torch.linalg.svd(w.grad, full_matrices=False)
    torch.testing.assert_close(w.detach(), w0 - 0.1 * U @ Vh)


def test_parameter_routing():
    model = _net()
    opt = Tensorion(model.parameters(), min_order=3)
    _train(model, opt, steps=1)
    for p in model.parameters():
        if p.ndim >= 3:
            assert set(opt.state[p]) == {"momentum_buffer"}
        else:
            assert set(opt.state[p]) == {"step", "moment1", "moment2"}


def test_muon_style_param_groups():
    model = _net()
    head = list(model[-1].parameters())
    body = [p for p in model.parameters() if all(p is not h for h in head)]
    groups = [
        dict(params=[p for p in body if p.ndim >= 2], use_muon=True),
        dict(params=[p for p in body if p.ndim < 2] + head, use_muon=False, lr=3e-4, betas=(0.8, 0.9), eps=1e-10),
    ]
    opt = Tensorion(groups)
    _train(model, opt, steps=1)
    assert "momentum_buffer" not in opt.state[model[-1].weight]
    assert "momentum_buffer" in opt.state[model[0].weight]
    adam_group = opt.param_groups[1]
    assert (adam_group["lr"], adam_group["adamw_betas"], adam_group["adamw_eps"]) == (3e-4, (0.8, 0.9), 1e-10)


def test_rejects_vectors_forced_onto_tensorion():
    with pytest.raises(ValueError, match="cannot get the Tensorion update"):
        Tensorion([dict(params=[nn.Parameter(torch.zeros(3))], use_muon=True)])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(unfolding="smallest"),
        dict(orthogonalizer="qr"),
        dict(adjust_lr_fn="spectral"),
        dict(fallback="adam"),
        dict(unfolding=0),
        dict(unfolding={"0.weight": 0}),  # names needed, but plain parameters were passed
    ],
)
def test_rejects_bad_options_early(kwargs):
    with pytest.raises((TypeError, ValueError)):
        Tensorion(_net().parameters(), **kwargs)


@pytest.mark.parametrize("unfolding", sorted(STRATEGIES) + [{"0.weight": (0, 2), "2.weight": -1, "4.weight": 0}])
@pytest.mark.parametrize("orthogonalizer", sorted(ORTHOGONALIZERS))
def test_all_options_train(unfolding, orthogonalizer):
    model = _net()
    opt = Tensorion(
        model.named_parameters(),
        lr=1e-3,
        unfolding=unfolding,
        orthogonalizer=orthogonalizer,
        nesterov=True,
    )
    before = copy.deepcopy(model.state_dict())
    _train(model, opt)
    after = model.state_dict()
    assert all(torch.isfinite(v).all() for v in after.values())
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_sgd_fallback():
    model = _net()
    opt = Tensorion(model.parameters(), fallback="sgd")
    _train(model, opt, steps=1)
    assert set(opt.state[model[0].bias]) == {"step", "momentum"}


def test_per_group_unfolding():
    model = _net()
    groups = [
        dict(params=list(model[0].parameters()), unfolding="online"),
        dict(params=list(model[2].parameters()) + list(model[4].parameters())),
    ]
    opt = Tensorion(groups, unfolding="offline")
    assert [g["unfolding"] for g in opt.param_groups] == ["online", "offline"]
    _train(model, opt, steps=1)


def test_state_dict_roundtrip():
    model = _net()
    opt = Tensorion(model.parameters())
    _train(model, opt, steps=2)

    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer)

    clone = _net()
    clone.load_state_dict(saved["model"])
    clone_opt = Tensorion(clone.parameters())
    clone_opt.load_state_dict(saved["opt"])

    _train(model, opt, steps=1)
    _train(clone, clone_opt, steps=1)
    for p, q in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(p, q)


def test_closure_returns_loss():
    model = _net()
    opt = Tensorion(model.parameters())

    def closure():
        opt.zero_grad()
        loss = model(torch.randn(2, 3, 10, 12)).square().mean()
        loss.backward()
        return loss

    assert torch.is_tensor(opt.step(closure))


def test_per_layer_unfolding_by_name():
    model = _net()  # weights: 0.weight (8, 3, 3, 3), 2.weight (4, 8, 3, 5), 4.weight (10, 144)
    opt = Tensorion(model.named_parameters(), unfolding={"0.weight": (0, 1), "2.*": 2})
    probe = torch.zeros(1)
    assert opt._unfoldings[model[0].weight](probe) == (0, 1)
    assert opt._unfoldings[model[2].weight](probe) == (2,)
    # Unmatched layers use the offline strategy.
    assert opt._unfoldings[model[4].weight](model[4].weight) == (0,)
    _train(model, opt)


def test_per_layer_axes_are_checked_against_each_shape():
    with pytest.raises(ValueError, match="'4.weight' of shape \\(10, 144\\)"):
        Tensorion(_net().named_parameters(), unfolding={"*.weight": (0, 1)})


def test_unmatched_unfolding_keys_are_reported():
    with pytest.raises(ValueError, match="conv9"):
        Tensorion(_net().named_parameters(), unfolding={"conv9.weight": 0})


def test_named_parameters_with_strategy():
    model = _net()
    _train(model, Tensorion(model.named_parameters(), unfolding="online"))


def test_per_layer_state_dict_roundtrip():
    spec = {"0.weight": (0, 1), "2.weight": 2}
    model = _net()
    opt = Tensorion(model.named_parameters(), unfolding=spec)
    _train(model, opt, steps=2)

    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer)

    clone = _net()
    clone.load_state_dict(saved["model"])
    clone_opt = Tensorion(clone.named_parameters(), unfolding=spec)
    clone_opt.load_state_dict(saved["opt"])

    _train(model, opt, steps=1)
    _train(clone, clone_opt, steps=1)
    for p, q in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(p, q)


def test_online_every_reuses_the_split_between_selections(monkeypatch):
    from tensorion import unfolding

    calls = []
    original = unfolding.STRATEGIES["online"]

    def counting(x):
        calls.append(tuple(x.shape))
        return original(x)

    monkeypatch.setitem(unfolding.STRATEGIES, "online", counting)
    model = _net()
    opt = Tensorion(model.named_parameters(), unfolding="online", online_every=3)
    _train(model, opt, steps=7)
    # 3 layers get the Tensorion update, selections happen at steps 0, 3 and 6.
    assert len(calls) == 3 * 3
    state = opt.state[model[0].weight]
    assert state["unfolding_age"] == 7 and isinstance(state["row_modes"], tuple)

    # The stored split survives a checkpoint.
    clone = _net()
    clone.load_state_dict(model.state_dict())
    clone_opt = Tensorion(clone.named_parameters(), unfolding="online", online_every=3)
    clone_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    _train(model, opt, steps=2)
    _train(clone, clone_opt, steps=2)
    for p, q in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(p, q)


def test_online_every_defaults_to_ten():
    a, b = _net(), _net()
    opt = Tensorion(a.parameters(), unfolding="online")
    assert opt.defaults["online_every"] == 10
    _train(a, opt, steps=12)
    _train(b, Tensorion(b.parameters(), unfolding="online", online_every=10), steps=12)
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.equal(p, q)


def test_online_every_one_selects_at_every_step(monkeypatch):
    from tensorion import unfolding

    calls = []
    original = unfolding.STRATEGIES["online"]
    monkeypatch.setitem(unfolding.STRATEGIES, "online", lambda x: (calls.append(1), original(x))[1])
    model = _net()
    opt = Tensorion(model.parameters(), unfolding="online", online_every=1)
    _train(model, opt, steps=4)
    assert len(calls) == 3 * 4
    assert "row_modes" not in opt.state[model[0].weight]


@pytest.mark.parametrize("value", [0, -1, 2.5])
def test_rejects_bad_online_every(value):
    with pytest.raises(ValueError, match="online_every"):
        Tensorion(_net().parameters(), online_every=value)


def test_parameters_that_are_vectors_up_to_singleton_axes_use_the_fallback():
    shapes = {"depthwise": (8, 1, 3, 3), "conv1x1": (16, 8, 1, 1), "pos_embed": (1, 10, 16), "cls": (1, 1, 16), "scale": (16, 1, 1)}
    params = {name: nn.Parameter(torch.randn(shape)) for name, shape in shapes.items()}
    opt = Tensorion(list(params.items()))
    for p in params.values():
        p.grad = torch.randn_like(p)
    opt.step()
    uses_tensorion = {name: "momentum_buffer" in opt.state[p] for name, p in params.items()}
    assert uses_tensorion == {"depthwise": True, "conv1x1": True, "pos_embed": True, "cls": False, "scale": False}

    # min_order counts axes larger than 1: a 1x1 convolution and (1, n, d) embeddings are matrices.
    params = {name: nn.Parameter(torch.randn(shape)) for name, shape in shapes.items()}
    opt = Tensorion(list(params.items()), min_order=3)
    for p in params.values():
        p.grad = torch.randn_like(p)
    opt.step()
    uses_tensorion = {name: "momentum_buffer" in opt.state[p] for name, p in params.items()}
    assert uses_tensorion == {"depthwise": True, "conv1x1": False, "pos_embed": False, "cls": False, "scale": False}

    with pytest.raises(ValueError, match="cannot get the Tensorion update"):
        Tensorion([dict(params=[nn.Parameter(torch.zeros(1, 1, 16))], use_muon=True)])
    with pytest.raises(ValueError, match="match no parameter"):
        Tensorion([("cls", nn.Parameter(torch.zeros(1, 1, 16)))], unfolding={"cls": 2})


def test_same_weights_stored_as_conv1x1_or_linear_get_the_same_update():
    """The optimizer must not depend on axes of size 1."""
    w = torch.randn(16, 8)
    g = torch.randn(16, 8)
    results = []
    for shape in [(16, 8), (16, 8, 1, 1), (1, 16, 8)]:
        p = nn.Parameter(w.reshape(shape).clone())
        p.grad = g.reshape(shape).clone()
        Tensorion([p]).step()
        results.append(p.detach().reshape(16, 8))
    torch.testing.assert_close(results[0], results[1])
    torch.testing.assert_close(results[0], results[2])
