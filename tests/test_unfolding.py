import pytest
import torch

from tensorion.nuclear import nuclear_norm_bounds, nuclear_norm_exact
from tensorion.unfolding import (
    STRATEGIES,
    _candidates,
    check_unfolding,
    matricize,
    resolve,
    select,
    select_online,
    two_way_splits,
    unmatricize,
)


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


def _planted(shape, rows, rank=4, noise=0.01, seed=0):
    """A tensor that has low rank in the matricization with the given row modes."""
    g = torch.Generator().manual_seed(seed)
    r = 1
    for m in rows:
        r *= shape[m]
    numel = 1
    for n in shape:
        numel *= n
    low = torch.randn(r, rank, generator=g) @ torch.randn(rank, numel // r, generator=g)
    perm = list(rows) + [m for m in range(len(shape)) if m not in rows]
    x = low.reshape(*(shape[m] for m in perm)).permute(*(perm.index(m) for m in range(len(shape))))
    return x + noise * torch.randn(shape, generator=g)


def _brute_force(x):
    nuclear = {r: nuclear_norm_exact(matricize(x, r)) for r in two_way_splits(x.ndim)}
    return max(nuclear, key=nuclear.get), nuclear


@pytest.mark.parametrize("strategy", ["online", "online_exact"])
def test_online_picks_largest_nuclear_norm(strategy):
    # x[i, j, k] = A[i, k] * b[j]: the mode-1 unfolding has rank one, the others do not.
    A, b = torch.randn(6, 4), torch.randn(5)
    x = torch.einsum("ik,j->ijk", A, b)
    best, _ = _brute_force(x)
    choice = resolve(strategy, x.ndim)(x)
    assert choice == best
    assert choice != (1,)


@pytest.mark.parametrize("seed", range(5))
def test_online_exact_matches_brute_force(seed):
    """Pruning by the rank bound never changes the exact choice."""
    g = torch.Generator().manual_seed(seed)
    shape = [(6, 5, 4), (8, 8, 3, 3), (4, 9, 2, 3), (16, 2, 2, 2), (3, 4, 5, 2, 2)][seed]
    x = torch.randn(shape, generator=g)
    best, nuclear = _brute_force(x)
    assert nuclear[STRATEGIES["online_exact"](x)] == pytest.approx(nuclear[best], rel=1e-5)


@pytest.mark.parametrize("low_rank_rows", [(0,), (0, 2), (0, 3)])
def test_online_estimate_finds_the_best_split_of_large_tensors(low_rank_rows):
    """Matrices here are large enough for the SVD-free estimate to be used."""
    x = _planted((256, 128, 3, 3), low_rank_rows)
    best, nuclear = _brute_force(x)
    choice = STRATEGIES["online"](x)
    assert nuclear[choice] >= 0.99 * nuclear[best]
    assert set(choice) != set(low_rank_rows)


def test_online_skips_splits_ruled_out_by_the_rank_bound():
    seen = []

    def counting(M):
        seen.append(tuple(M.shape))
        return nuclear_norm_bounds(M)

    x = torch.randn(256, 256, 3, 3)
    select_online(x, bounds=counting)
    # Only the two 768 x 768 matricizations are formed, the other five are pruned.
    assert seen == [(768, 768), (768, 768)]


@pytest.mark.parametrize("shape", [(64, 128), (128, 64, 1, 1), (1, 32, 1, 5)])
def test_layers_with_one_distinct_unfolding_cost_nothing(shape):
    def fail(M):
        raise AssertionError("nothing to compare")

    assert len(_candidates(shape)) == 1
    select_online(torch.randn(shape), bounds=fail)


def test_candidates_are_sorted_and_deduplicated():
    assert _candidates((512, 512, 3, 3)) == (
        ((0, 2), 1536),
        ((0, 3), 1536),
        ((0,), 512),
        ((1,), 512),
        ((0, 1), 9),
        ((2,), 3),
        ((3,), 3),
    )
    assert _candidates((128, 64, 1, 1)) == (((0,), 64),)


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
    assert sorted(STRATEGIES) == ["offline", "online", "online_exact"]
