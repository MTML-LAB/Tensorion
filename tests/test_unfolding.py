import pytest
import torch

from tensorion.unfolding import STRATEGIES, check_unfolding, matricize, resolve, select, two_way_splits, unmatricize


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


def test_offline_picks_largest_rank():
    x = torch.randn(64, 32, 3, 3)
    # {0, 2} gives a 192 x 96 matrix, the largest possible rank (ties keep the first split).
    assert resolve("offline", x.ndim)(x) == (0, 2)


def test_online_picks_largest_nuclear_norm():
    # x[i, j, k] = A[i, k] * b[j]: the mode-1 unfolding has rank one, the others do not.
    A, b = torch.randn(6, 4), torch.randn(5)
    x = torch.einsum("ik,j->ijk", A, b)
    nuclear = {r: torch.linalg.svdvals(matricize(x, r)).sum().item() for r in two_way_splits(3)}
    choice = resolve("online", x.ndim)(x)
    assert choice == max(nuclear, key=nuclear.get)
    assert choice != (1,)


def test_fixed_axes():
    x = torch.randn(3, 4, 5)
    assert resolve(1, 3)(x) == (1,)
    assert resolve(-1, 3)(x) == (2,)
    assert resolve([2, 0], 3)(x) == (2, 0)


@pytest.mark.parametrize("spec", [3, (0, 0), (0, 1, 2), ()])
def test_invalid_fixed_axes_for_shape(spec):
    with pytest.raises(ValueError):
        resolve(spec, 3)


@pytest.mark.parametrize("name", ["smallest", "min", "max", "maxrank", "dual"])
def test_unknown_strategy(name):
    with pytest.raises(ValueError, match="unknown unfolding strategy"):
        check_unfolding(name)


@pytest.mark.parametrize("spec", [0, (0, 1), [2]])
def test_fixed_axes_must_be_per_parameter(spec):
    with pytest.raises(TypeError, match="per parameter"):
        check_unfolding(spec)


@pytest.mark.parametrize("spec", [lambda x: 0, {"conv.weight": "min"}, {"conv.weight": 1.5}, {0: "offline"}])
def test_invalid_unfolding_arguments(spec):
    with pytest.raises((TypeError, ValueError)):
        check_unfolding(spec)


def test_select():
    spec = {"layer1.conv.weight": (0, 1), "layer*.conv.weight": 2, "*.weight": "online"}
    assert select(spec, "layer1.conv.weight") == ((0, 1), "layer1.conv.weight")  # exact name wins
    assert select(spec, "layer2.conv.weight") == (2, "layer*.conv.weight")  # then the first matching pattern
    assert select(spec, "head.weight") == ("online", "*.weight")
    assert select(spec, "head.bias") == ("offline", None)  # unmatched: the default
    assert select("online", "anything") == ("online", None)
    assert sorted(STRATEGIES) == ["offline", "online"]
