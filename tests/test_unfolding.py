import pytest
import torch

from tensorion.unfolding import matricize, resolve_unfolding, two_way_splits, unmatricize


@pytest.mark.parametrize("row_modes", [(0,), (1,), (3,), (0, 2), (1, 3), (0, 1, 2)])
def test_matricize_roundtrip(row_modes):
    x = torch.randn(4, 5, 2, 3)
    m = matricize(x, row_modes)
    rows = 1
    for r in row_modes:
        rows *= x.shape[r]
    assert m.shape == (rows, x.numel() // rows)
    assert torch.equal(unmatricize(m, x.shape, row_modes), x)


def test_mode_unfolding_rows_are_mode_fibers():
    x = torch.randn(4, 5, 6)
    m = matricize(x, (1,))
    assert torch.equal(m[2], x[:, 2, :].reshape(-1))


def test_two_way_splits_are_unique():
    assert two_way_splits(2) == [(0,)]
    assert two_way_splits(3) == [(0,), (1,), (2,)]
    # 4 axes: 4 singletons + 3 balanced pairs, complements removed.
    assert two_way_splits(4) == [(0,), (1,), (2,), (3,), (0, 1), (0, 2), (0, 3)]


def test_builtin_strategies():
    x = torch.randn(64, 32, 3, 3)
    assert resolve_unfolding("min")(x) == (2,)
    assert resolve_unfolding("max")(x) == (0,)
    # {0, 2} gives a 192 x 96 matrix, the largest possible rank (ties keep the first split).
    assert resolve_unfolding("maxrank")(x) == (0, 2)


def test_dual_picks_largest_nuclear_norm():
    # x[i, j, k] = A[i, k] * b[j]: the mode-1 unfolding has rank one, the others do not.
    A, b = torch.randn(6, 4), torch.randn(5)
    x = torch.einsum("ik,j->ijk", A, b)
    nuclear = {r: torch.linalg.svdvals(matricize(x, r)).sum().item() for r in two_way_splits(3)}
    choice = resolve_unfolding("dual")(x)
    assert choice == max(nuclear, key=nuclear.get)
    assert choice != (1,)


def test_fixed_and_callable_specs():
    x = torch.randn(3, 4, 5)
    assert resolve_unfolding(1)(x) == (1,)
    assert resolve_unfolding(-1)(x) == (2,)
    assert resolve_unfolding([2, 0])(x) == (2, 0)
    assert resolve_unfolding(lambda t: t.ndim - 1)(x) == (2,)


@pytest.mark.parametrize("spec", [3, (0, 0), (0, 1, 2), ()])
def test_invalid_fixed_specs(spec):
    with pytest.raises(ValueError):
        resolve_unfolding(spec)(torch.randn(3, 4, 5))


def test_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown unfolding"):
        resolve_unfolding("smallest")
