import numpy as np
import torch


Tensor = torch.Tensor


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
    inv_order = list(np.argsort(order))

    # a = A.reshape(np.array(original_shape)[order])
    # a = np.transpose(a, inv_order)

    X = x.reshape(tuple(np.array(original_shape, dtype=int)[order]))
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
