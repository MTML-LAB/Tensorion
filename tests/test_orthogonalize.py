import pytest
import torch

from tensorion.orthogonalize import ORTHOGONALIZERS, resolve_orthogonalizer


def _matrix(rows, cols, cond, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = min(rows, cols)
    U, _ = torch.linalg.qr(torch.randn(rows, k, generator=g))
    V, _ = torch.linalg.qr(torch.randn(cols, k, generator=g))
    s = torch.logspace(0, -torch.log10(torch.tensor(float(cond))).item(), k)
    return U @ torch.diag(s) @ V.T


@pytest.mark.parametrize("shape", [(32, 96), (96, 32)])
def test_svd_is_exact_polar_factor(shape):
    G = _matrix(*shape, cond=1e3)
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    torch.testing.assert_close(ORTHOGONALIZERS["svd"](G), U @ Vh)


@pytest.mark.parametrize(
    "name, steps, low, high",
    [
        # Newton-Schulz trades accuracy for speed by design.
        ("newton_schulz", 8, 0.6, 1.2),
        ("polar_express", 8, 0.99, 1.01),
        ("cans", 10, 0.99, 1.01),
    ],
)
@pytest.mark.parametrize("shape", [(32, 96), (96, 32)])
def test_iterative_methods_orthogonalize(name, steps, low, high, shape):
    G = _matrix(*shape, cond=1e3)
    s = torch.linalg.svdvals(ORTHOGONALIZERS[name](G, steps))
    assert low <= s.min() and s.max() <= high


@pytest.mark.parametrize("name", sorted(ORTHOGONALIZERS))
def test_shape_dtype_and_input_untouched(name):
    G = torch.randn(16, 40, dtype=torch.bfloat16)
    G_copy = G.clone()
    out = ORTHOGONALIZERS[name](G, 5)
    assert out.shape == G.shape and out.dtype == G.dtype
    assert torch.equal(G, G_copy)


@pytest.mark.parametrize("name", ["newton_schulz", "polar_express", "cans"])
def test_zero_matrix_stays_finite(name):
    assert torch.isfinite(ORTHOGONALIZERS[name](torch.zeros(8, 12), 5)).all()


def test_resolve_orthogonalizer():
    custom = lambda G, steps: G
    assert resolve_orthogonalizer(custom) is custom
    with pytest.raises(ValueError, match="Unknown orthogonalizer"):
        resolve_orthogonalizer("qr")
