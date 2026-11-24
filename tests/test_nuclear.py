import pytest
import torch

from tensorion.nuclear import nuclear_norm, nuclear_norm_bounds, nuclear_norm_exact


def _with_spectrum(rows, cols, singular_values, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = len(singular_values)
    U, _ = torch.linalg.qr(torch.randn(rows, k, generator=g))
    V, _ = torch.linalg.qr(torch.randn(cols, k, generator=g))
    return U @ torch.diag(singular_values) @ V.T


def _matrices():
    g = torch.Generator().manual_seed(0)
    i = torch.arange(1, 513.0)
    return {
        "gaussian": torch.randn(512, 768, generator=g),
        "tall": torch.randn(900, 300, generator=g),
        "decay 1/i": _with_spectrum(512, 768, 1 / i),
        "decay 1/i^2": _with_spectrum(512, 768, 1 / i**2),
        # A few dominant directions followed by a cliff, as in small-batch gradients.
        "cliff": _with_spectrum(512, 768, torch.cat([torch.ones(12), torch.full((500,), 1e-3)])),
        "rank 3": _with_spectrum(512, 768, torch.tensor([3.0, 2.0, 1.0])),
        "many zeros": torch.block_diag(torch.randn(200, 200, generator=g), torch.zeros(400, 400)),
    }


@pytest.mark.parametrize("name", sorted(_matrices()))
def test_estimate_is_close_to_svd(name):
    M = _matrices()[name]
    assert nuclear_norm(M) == pytest.approx(nuclear_norm_exact(M), rel=0.1)


@pytest.mark.parametrize("name", sorted(_matrices()))
def test_bounds_are_ordered_and_bracket_loosely(name):
    M = _matrices()[name]
    lower, upper = nuclear_norm_bounds(M)
    exact = nuclear_norm_exact(M)
    assert lower <= upper
    # Bounds hold up to probe noise.
    assert lower <= 1.1 * exact and upper >= 0.9 * exact


def test_small_side_uses_the_gram_matrix():
    M = torch.randn(64, 4096)
    lower, upper = nuclear_norm_bounds(M)
    assert lower == upper == pytest.approx(nuclear_norm_exact(M), rel=1e-5)
    assert nuclear_norm_bounds(M.T) == pytest.approx((lower, upper), rel=1e-6)


def test_very_wide_matrix_is_cheap():
    """A 3 x N unfolding needs a 3 x 3 eigenvalue problem, not an SVD."""
    M = torch.randn(3, 2_000_000)
    assert nuclear_norm(M) == pytest.approx(nuclear_norm_exact(M), rel=1e-4)


@pytest.mark.parametrize(
    "singular_values",
    [torch.cat([torch.ones(1), torch.zeros(99)]), torch.logspace(0, -6, 100), torch.ones(100)],
    ids=["rank 1", "geometric decay", "flat"],
)
def test_gram_matrix_is_accurate_enough_for_small_singular_values(singular_values):
    M = _with_spectrum(100, 5000, singular_values)
    assert nuclear_norm(M) == pytest.approx(singular_values.sum().item(), rel=0.02)


def test_zero_matrix():
    assert nuclear_norm_bounds(torch.zeros(300, 400)) == (0.0, 0.0)


def test_scale_equivariance():
    M = torch.randn(256, 512)
    assert nuclear_norm(1e-6 * M) == pytest.approx(1e-6 * nuclear_norm(M), rel=1e-4)
    assert nuclear_norm(1e6 * M) == pytest.approx(1e6 * nuclear_norm(M), rel=1e-4)


def test_deterministic_and_leaves_the_global_rng_alone():
    M = torch.randn(256, 512)
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    first = nuclear_norm(M)
    assert torch.equal(torch.rand(3), expected)
    assert nuclear_norm(M) == first


def test_input_is_not_modified():
    M = torch.randn(300, 200)
    copy = M.clone()
    nuclear_norm(M)
    assert torch.equal(M, copy)
