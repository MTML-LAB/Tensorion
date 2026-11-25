import pytest
import torch

from tensorion.orthogonalize import ORTHOGONALIZERS, _cans_coefficients, resolve_orthogonalizer


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


def test_cans_coefficients_match_paper():
    """Degree-3 CANS composition for Muon, 7 steps (Grishina et al., 2025, Appendix J)."""
    paper = [
        (5.181702879894027, -5.177039351076183),
        (2.5854225645668487, -0.6478627820075661),
        (2.565592012027513, -0.6452645701961278),
        (2.5162233474315263, -0.6387826202434335),
        (2.401068707564606, -0.6235851252726741),
        (2.1708447617901196, -0.5928497805346629),
        (1.8394377168195162, -0.5476683622291173),
    ]
    ours = [c for a, b, _ in _cans_coefficients(7, 9e-4) for c in (a, b)]
    assert ours == pytest.approx([c for pair in paper for c in pair], rel=1e-9)


@pytest.mark.parametrize("name", ["newton_schulz", "polar_express", "cans"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_reduced_precision_is_close_to_float32(name, dtype):
    G = _matrix(48, 96, cond=30)
    reference = ORTHOGONALIZERS[name](G)
    reduced = ORTHOGONALIZERS[name](G.to(dtype))
    assert reduced.dtype == dtype
    cosine = torch.nn.functional.cosine_similarity(reference.flatten(), reduced.float().flatten(), dim=0)
    assert cosine > 0.99
