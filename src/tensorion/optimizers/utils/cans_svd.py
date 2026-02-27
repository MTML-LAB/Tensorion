import torch

from tensorion.optimizers.utils.cans import cans_iteration


def cans_svd(
    matrix: torch.Tensor,
    degree: int = 3,
    preprocess: bool = True,
    preprocess_iters: int = 2,
    delta: float = 0.99,
    a: float = 1e-3,
    num_iters: int = 50,
) -> torch.Tensor:
    """
    Compute the SVD of a matrix using the CANS method.

    Args:
        matrix (torch.Tensor): The input matrix.
        degree (int): The degree of the polynomial approximation.
        preprocess (bool): Whether to use preprocessing.
        preprocess_iters (int): Number of preprocessing iterations.
        delta (float): The delta parameter for CANS.
        a (float): The scaling parameter for CANS.
        num_iters (int): Number of iterations for the CANS method.

    Returns:
        U: torch.Tensor: Left singular vectors.
        S: torch.Tensor: Singular values.
        Vt: torch.Tensor: Right singular vectors (transposed).
    """

    transpose = False
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
        transpose = True

    W = cans_iteration(
        matrix,
        n=num_iters,
        a=a,
        degree=degree,
        preprocess=preprocess,
        preprocess_iters=preprocess_iters,
        delta=delta,
    )

    H = W.T @ matrix
    H = 0.5 * (H + H.T)

    S, V = torch.linalg.eigh(H)  # ascending order
    U = W @ V

    U, R = torch.linalg.qr(U, mode="reduced")
    s = torch.diagonal(R) * S

    sign = torch.sign(s)
    sign[sign == 0] = 1.0

    V = V * sign.unsqueeze(0)
    s = s.abs()

    idx = torch.argsort(s, descending=True)

    U = U[:, idx]
    V = V[:, idx]
    s = s[idx]

    if not transpose:
        return U, s, V.T
    else:
        return V, s, U.T


def cans_polar_iteration(
    matrix: torch.Tensor,
    degree: int = 3,
    preprocess: bool = True,
    preprocess_iters: int = 2,
    delta: float = 0.99,
    a: float = 1e-3,
    num_iters: int = 50,
) -> torch.Tensor:
    """
    Compute the polar iteration of a matrix using the CANS method.
    """

    transpose = False
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
        transpose = True

    W = cans_iteration(
        matrix,
        n=num_iters,
        a=a,
        degree=degree,
        preprocess=preprocess,
        preprocess_iters=preprocess_iters,
        delta=delta,
    )

    if transpose:
        return W.T
    return W
