"""The JAX implementation must agree with the PyTorch one."""

import itertools

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
optax = pytest.importorskip("optax")
import jax.numpy as jnp  # noqa: E402

import tensorion_jax  # noqa: E402
from tensorion import Tensorion  # noqa: E402
from tensorion import orthogonalize as torch_orthogonalize  # noqa: E402
from tensorion import unfolding as torch_unfolding  # noqa: E402
from tensorion.nuclear import nuclear_norm_exact as torch_nuclear_norm_exact  # noqa: E402

SHAPES = {
    "conv1/kernel": (8, 3, 3, 3),
    "conv2/kernel": (4, 8, 3, 5),
    "conv2/bias": (4,),
    "fc/kernel": (10, 24),
    "fc/bias": (10,),
}


def _nested(flat):
    tree = {}
    for path, value in flat.items():
        module, name = path.split("/")
        tree.setdefault(module, {})[name] = value
    return tree


def _run(steps=4, **kwargs):
    """Train the same parameters on the same gradients with both implementations."""
    rng = np.random.default_rng(0)
    initial = {path: rng.standard_normal(shape).astype(np.float32) for path, shape in SHAPES.items()}
    grads = [
        {path: rng.standard_normal(shape).astype(np.float32) for path, shape in SHAPES.items()} for _ in range(steps)
    ]

    torch_params = {path: torch.nn.Parameter(torch.from_numpy(value.copy())) for path, value in initial.items()}
    torch_optimizer = Tensorion(list(torch_params.items()), lr=1e-2, **kwargs)
    for g in grads:
        for path, p in torch_params.items():
            p.grad = torch.from_numpy(g[path].copy())
        torch_optimizer.step()

    jax_kwargs = dict(kwargs)
    if "adamw_betas" in jax_kwargs:
        jax_kwargs["adamw_b1"], jax_kwargs["adamw_b2"] = jax_kwargs.pop("adamw_betas")
    optimizer = tensorion_jax.tensorion(1e-2, **jax_kwargs)
    params = _nested({path: jnp.asarray(value) for path, value in initial.items()})
    state = optimizer.init(params)

    @jax.jit
    def step(params, state, g):
        updates, state = optimizer.update(g, state, params)
        return optax.apply_updates(params, updates), state

    for g in grads:
        params, state = step(params, state, _nested({path: jnp.asarray(value) for path, value in g.items()}))

    for path, p in torch_params.items():
        module, name = path.split("/")
        np.testing.assert_allclose(np.asarray(params[module][name]), p.detach().numpy(), rtol=2e-3, atol=2e-5, err_msg=path)
    return state


@pytest.mark.parametrize("orthogonalizer", ["newton_schulz", "polar_express", "cans", "svd"])
def test_matches_pytorch_for_every_orthogonalizer(orthogonalizer):
    _run(orthogonalizer=orthogonalizer)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(nesterov=True, nuclear=True, orthogonalizer="svd"),
        dict(adjust_lr_fn="original"),
        dict(adjust_lr_fn=None, weight_decay=0.0),
        dict(min_ndim=3),
        dict(fallback="sgd"),
        dict(fallback="sgd", sgd_momentum=0.0, weight_decay=0.0),
        dict(ns_steps=3, momentum=0.8),
        dict(adamw_betas=(0.8, 0.9)),
    ],
)
def test_matches_pytorch_for_other_options(kwargs):
    _run(**kwargs)


@pytest.mark.parametrize(
    "unfolding",
    [
        {"conv1/kernel": (0, 1), "conv2/*": 2},
        {"conv*/kernel": -1, "fc/kernel": 1},
        {"conv2/kernel": "online_exact", "conv1/kernel": 0},
    ],
)
def test_matches_pytorch_for_per_layer_unfoldings(unfolding):
    _run(unfolding=unfolding, orthogonalizer="svd", online_every=1)


@pytest.mark.parametrize("unfolding", ["online", "online_exact"])
@pytest.mark.parametrize("online_every", [1, 3])
def test_matches_pytorch_for_online_unfoldings(unfolding, online_every):
    state = _run(steps=7, unfolding=unfolding, online_every=online_every, orthogonalizer="svd")
    assert int(state.inner_states["tensorion"].inner_state[0].count) == 7


def test_same_candidates_as_pytorch():
    for shape in itertools.product([1, 2, 3, 8, 64], [1, 3, 4, 64], [1, 3, 7], [1, 3]):
        assert tensorion_jax.candidates(shape) == torch_unfolding._candidates(shape), shape
    assert tensorion_jax.offline_rows((64, 32, 3, 3)) == (0, 2)


@pytest.mark.parametrize("name", ["newton_schulz", "polar_express", "cans", "svd"])
@pytest.mark.parametrize("shape", [(24, 60), (60, 24)])
def test_orthogonalizers_match_pytorch(name, shape):
    G = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
    ours = np.asarray(tensorion_jax.ORTHOGONALIZERS[name](jnp.asarray(G)))
    theirs = torch_orthogonalize.ORTHOGONALIZERS[name](torch.from_numpy(G)).numpy()
    np.testing.assert_allclose(ours, theirs, rtol=1e-3, atol=1e-4)


def _with_spectrum(rows, cols, singular_values, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = len(singular_values)
    U, _ = torch.linalg.qr(torch.randn(rows, k, generator=g))
    V, _ = torch.linalg.qr(torch.randn(cols, k, generator=g))
    return U @ torch.diag(singular_values) @ V.T


@pytest.mark.parametrize(
    "matrix",
    [
        torch.randn(512, 768, generator=torch.Generator().manual_seed(0)),
        torch.randn(900, 300, generator=torch.Generator().manual_seed(0)),
        _with_spectrum(512, 768, 1 / torch.arange(1, 513.0)),
        _with_spectrum(512, 768, torch.cat([torch.ones(12), torch.full((500,), 1e-3)])),
        _with_spectrum(512, 768, torch.tensor([3.0, 2.0, 1.0])),
        torch.zeros(300, 400),
    ],
    ids=["gaussian", "tall", "decay", "cliff", "rank 3", "zero"],
)
def test_nuclear_norm_estimate(matrix):
    lower, upper = jax.jit(tensorion_jax.nuclear_norm_bounds)(jnp.asarray(matrix.numpy()))
    exact = torch_nuclear_norm_exact(matrix)
    assert float(lower) <= float(upper)
    assert (float(lower) + float(upper)) / 2 == pytest.approx(exact, rel=0.1, abs=1e-6)


@pytest.mark.parametrize("function", ["nuclear_norm_bounds", "nuclear_norm_exact"])
def test_nuclear_norm_of_a_very_wide_matrix_fits_in_memory(function):
    """A 3 x N unfolding must not allocate an N x N matrix, with the Gram matrix or the SVD."""
    M = np.random.default_rng(0).standard_normal((3, 400_000)).astype(np.float32)
    lower, upper = jax.jit(getattr(tensorion_jax, function))(jnp.asarray(M))
    assert float(lower) == float(upper) == pytest.approx(np.linalg.svd(M, compute_uv=False).sum(), rel=1e-4)


def test_vectors_up_to_singleton_axes_use_the_fallback_like_pytorch():
    shapes = {"depthwise": (8, 1, 3, 3), "conv1x1": (16, 8, 1, 1), "pos_embed": (1, 10, 16), "cls": (1, 1, 16), "scale": (16, 1, 1)}
    params = {name: jnp.ones(shape) for name, shape in shapes.items()}
    state = tensorion_jax.tensorion(1e-3).init(params)
    momentum = state.inner_states["tensorion"].inner_state[0].momentum
    assert sorted(k for k, v in momentum.items() if isinstance(v, jax.Array)) == ["conv1x1", "depthwise", "pos_embed"]
    assert tensorion_jax.offline_dimension_numbers(params)["cls"] is None
    with pytest.raises(ValueError, match="cannot get the Tensorion update"):
        tensorion_jax.scale_by_tensorion().init({"cls": jnp.ones((1, 1, 16))})
    assert tensorion_jax.candidates((64, 1, 3, 3)) == (((0,), 9), ((2,), 3), ((3,), 3))


def test_online_finds_the_best_split_of_a_large_tensor():
    """Large enough for the SVD-free estimate: low rank in split (0, 2), so another one must win."""
    g = torch.Generator().manual_seed(0)
    low = torch.randn(768, 4, generator=g) @ torch.randn(4, 384, generator=g)
    x = low.reshape(256, 3, 128, 3).permute(0, 2, 1, 3) + 0.01 * torch.randn(256, 128, 3, 3, generator=g)
    nuclear = {
        rows: torch_nuclear_norm_exact(torch_unfolding.matricize(x, rows)) for rows in torch_unfolding.two_way_splits(4)
    }

    optimizer = tensorion_jax.scale_by_tensorion(unfolding="online", online_every=1, momentum=0.0)
    params = {"w": jnp.asarray(x.numpy())}
    _, state = jax.jit(optimizer.update)(params, optimizer.init(params))
    chosen = tensorion_jax.candidates(tuple(x.shape))[int(state.choice["w"])][0]
    assert nuclear[chosen] >= 0.99 * max(nuclear.values())
    assert chosen != (0, 2)


def test_online_every_reuses_the_choice():
    optimizer = tensorion_jax.scale_by_tensorion(unfolding="online_exact", online_every=4, orthogonalizer="svd")
    rng = np.random.default_rng(0)
    params = {"w": jnp.asarray(rng.standard_normal((6, 5, 4)).astype(np.float32))}
    state = optimizer.init(params)
    update = jax.jit(optimizer.update)
    choices = []
    for _ in range(8):
        g = {"w": jnp.asarray(rng.standard_normal((6, 5, 4)).astype(np.float32))}
        _, state = update(g, state)
        choices.append(int(state.choice["w"]))
    assert len(set(choices[:4])) == 1 and len(set(choices[4:])) == 1


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(unfolding="min"), ValueError),
        (dict(unfolding=0), TypeError),
        (dict(orthogonalizer="qr"), ValueError),
        (dict(adjust_lr_fn="spectral"), ValueError),
        (dict(online_every=0), ValueError),
        (dict(fallback="adam"), ValueError),
    ],
)
def test_rejects_bad_options(kwargs, error):
    with pytest.raises(error):
        tensorion_jax.tensorion(1e-3, **kwargs)


def test_bad_per_layer_unfoldings_are_reported_at_init():
    params = _nested({path: jnp.zeros(shape) for path, shape in SHAPES.items()})
    with pytest.raises(ValueError, match="'fc/kernel' of shape \\(10, 24\\)"):
        tensorion_jax.tensorion(1e-3, unfolding={"*/kernel": (0, 1)}).init(params)
    with pytest.raises(ValueError, match="conv9"):
        tensorion_jax.tensorion(1e-3, unfolding={"conv9/kernel": 0}).init(params)


def test_param_labels_override_the_routing():
    params = _nested({path: jnp.ones(shape) for path, shape in SHAPES.items()})
    labels = jax.tree.map(lambda p: "fallback", params)
    labels["conv1"]["kernel"] = "tensorion"
    optimizer = tensorion_jax.tensorion(1e-3, param_labels=labels)
    state = optimizer.init(params)
    momentum = state.inner_states["tensorion"].inner_state[0].momentum
    assert [p.shape for p in jax.tree.leaves(momentum)] == [(8, 3, 3, 3)]


def test_offline_dimension_numbers_work_with_optax_muon():
    params = {"conv": jnp.ones((64, 32, 3, 3)), "fc": jnp.ones((10, 20)), "bias": jnp.zeros(10)}
    numbers = tensorion_jax.offline_dimension_numbers(params)
    assert numbers["conv"] == optax.contrib.MuonDimensionNumbers(reduction_axis=(0, 2), output_axis=(1, 3))
    assert numbers["fc"] == optax.contrib.MuonDimensionNumbers(reduction_axis=(0,), output_axis=(1,))
    assert numbers["bias"] is None

    optimizer = optax.contrib.muon(1e-3, muon_weight_dimension_numbers=tensorion_jax.offline_dimension_numbers)
    grads = jax.tree.map(lambda p: jnp.asarray(np.random.default_rng(0).standard_normal(p.shape), p.dtype), params)
    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    assert all(bool(jnp.isfinite(u).all()) for u in jax.tree.leaves(updates))
