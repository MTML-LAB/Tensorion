import torch
import math
import itertools

from typing import Union

Tensor = torch.Tensor


def _get_unfold_transpose_order(d: int, mode: int) -> list[int]:
    enum = list(range(d))
    order = [mode] + enum[:mode] + enum[mode + 1 :]

    return order


def _get_matricization_transpose_order(d: int, modes: list[int]) -> list[int]:
    if not modes:
        raise ValueError(f"Incorrect {modes=}\nModes mustn't be empty")

    if (max(modes) >= d) or (min(modes) < 0):
        raise ValueError(f"Incorrect {modes=}\nAll elements must be in [0, d-1]")

    if len(set(modes)) != len(modes):
        raise ValueError(f"Incorrect {modes=}\nAll elements must be unique")

    order = modes + list(filter(
        lambda m: m not in modes,
        range(d)
    ))

    return order


def inverse_order(d: int, order: list[int]) -> list[int]:
    inv_order = [0]*d
    
    for i, j in enumerate(order): 
        inv_order[j] = i
    
    return inv_order


def unfold(X: Tensor, mode: int) -> Tensor:
    # n1 x ... x nd -> n_mode x (N/n_mode), N = n1 * ... * nd
    order = _get_unfold_transpose_order(X.ndim, mode)

    x = X.permute(*order)
    x = x.reshape((x.shape[0], -1))

    return x


def fold(x: Tensor, original_shape: Union[tuple[int], torch.Size], mode: int):
    order = _get_unfold_transpose_order(len(original_shape), mode)
    inv_order = inverse_order(len(original_shape), order)
    perm_shape = tuple(original_shape[j] for j in order)

    # a = A.reshape(np.array(original_shape)[order])
    # a = np.transpose(a, inv_order)

    X = x.reshape(*perm_shape)
    X = X.permute(*inv_order)

    return X


def matricization(X: Tensor, modes: list[int]) -> Tensor:
    """
    Multi-mode tensor unfolding (matricization).

    Permutes `X` so that axes in `modes` come first (in the given order), then
    flattens the first `len(modes)` axes into the row dimension and the rest
    into the column dimension.

    Parameters
    ----------
    X : Tensor
        Input tensor with shape (I_0, ..., I_{d-1}).
    modes : list[int]
        Non-empty list of unique axes in [0, X.ndim-1]. Their order defines the
        row-wise linearization.

    Returns
    -------
    Tensor
        Matrix of shape:
          (prod_{k in modes} I_k, prod_{k not in modes} I_k),
        where I_k = X.shape[k].
    """

    order = _get_matricization_transpose_order(X.ndim, modes)

    x = X.permute(*order)

    m = math.prod(x.shape[:len(modes)])
    x = x.reshape((m, -1))

    return x


def un_matricization(x: Tensor, original_shape: Union[tuple[int], torch.Size], modes: list[int]) -> Tensor:
    """
    Inverse of `matricization`: fold a matrix back into a tensor.

    Reshapes `x` into the permuted tensor shape where axes in `modes` are first
    (same order as in `modes`), then applies the inverse permutation to recover
    `original_shape`.

    Parameters
    ----------
    x : Tensor
        2D matrix with total size equal to prod(original_shape).
    original_shape : tuple[int]
        Target tensor shape (I_0, ..., I_{d-1}).
    modes : list[int]
        Same `modes` used for unfolding.

    Returns
    -------
    Tensor
        Reconstructed tensor of shape `original_shape` (returned contiguous).
    """

    order = _get_matricization_transpose_order(len(original_shape), modes)
    inv_order = inverse_order(len(original_shape), order)
    perm_shape = tuple(original_shape[j] for j in order)

    X = x.reshape(*perm_shape)
    X = X.permute(*inv_order)
    X = X.contiguous()

    return X


def all_two_bucket_partitions(d: int) -> list[tuple[list[int], list[int]]]:
    """
    Build all unique partitions of range(d) into two (unordered) buckets.

    Returns pairs (A, B) such that:
      - A ∪ B = {0, ..., d-1}
      - A ∩ B = ∅
      - both buckets are non-empty
      - (A, B) and (B, A) are considered the same partition and returned once
    """
    if d < 2:
        return []

    elems = list(range(d))
    out: list[tuple[list[int], list[int]]] = []

    # To avoid duplicates (A,B) vs (B,A), fix bucket A to have size <= d//2
    for k in range(1, d // 2 + 1):
        for A in itertools.combinations(elems, k):
            B = [e for e in elems if e not in A]

            # If d is even and k == d/2, keep only one of each complementary pair
            # by requiring 0 to be in A (arbitrary but consistent tie-breaker).
            if d % 2 == 0 and k * 2 == d and 0 not in A:
                continue

            out.append((list(A), B))

    return out


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
