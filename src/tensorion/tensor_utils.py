import torch
import einops
import string
from itertools import combinations
import numpy as np


Tensor = torch.Tensor


def tucker_reconstruct(core, factors):
    """
    core.shape   = (r1, r2, ..., rN)
    factors[k].shape = (n_k, r_k)
    result.shape = (n1, n2, ..., nN)
    """
    n_modes = core.ndim
    letters = string.ascii_lowercase  # enough for typical N
    core_idx = letters[:n_modes]      # e.g. 'abc'
    out_idx  = letters[n_modes:2*n_modes]  # e.g. 'def'

    eq = core_idx + ',' + ','.join(
        f'{out_idx[i]}{core_idx[i]}' for i in range(n_modes)
    ) + '->' + out_idx

    return einops.einsum(core, *factors, eq, )

def partitions(n: int):

    if n < 2:
        return
    items = set(range(1, n))
    yield from (
        [list((0, *c)), list(items - set(c))]
        for r in range(n)
        if (rest := n - 1 - r) > 0
        for c in combinations(items, r)
    )


def invert_permutation(perm: list[int]) -> list[int]:
    """Return the inverse of a permutation."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv
    

def _get_unfold_transpose_order(d: int, mode: int) -> list[int]:
    enum = list(range(d))
    order = [mode] + enum[:mode] + enum[mode + 1 :]

    return order


def unfold(X: Tensor, mode: int) -> Tensor:
    # n1 x ... x nd -> n_mode x (N/n_mode), N = n1 * ... * nd
    order = _get_unfold_transpose_order(X.ndim, mode)

    x = X.permute(*order)
    x = x.reshape((x.shape[0], -1))

    return x


def fold(x: Tensor, original_shape: tuple[int], mode: int):
    order = _get_unfold_transpose_order(len(original_shape), mode)
    inv_order = np.argsort(order).tolist()
    perm_shape = tuple(original_shape[j] for j in order)

    # a = A.reshape(np.array(original_shape)[order])
    # a = np.transpose(a, inv_order)

    X = x.reshape(*perm_shape)
    X = X.permute(*inv_order)

    return X


def get_unfoldings_sigvals(x: Tensor) -> list[Tensor]:
    """
    :return: list of sigvals tensors for x unfoldings
    """

    d = x.dim()
    unfolding_sigvals = []

    for i in range(d):
        F = unfold(x, mode=i)
        unfolding_sigvals.append(torch.linalg.svdvals(F))

    return unfolding_sigvals
