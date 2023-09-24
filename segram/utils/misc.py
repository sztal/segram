import numpy as np
from numpy.linalg import norm


def cosine_similarity(
    X: np.ndarray[tuple[int] | tuple[int, int], np.floating],
    Y: np.ndarray[tuple[int] | tuple[int, int], np.floating],
    *,
    nan_as_zero: bool = True
) -> float | np.ndarray[tuple[int, ...], np.floating]:
    """Cosine similarity between two vectors.

    When 2D arrays are passed it is assumed that vectors
    for calculating similarities are arranged in rows.
    """
    Xnorm = norm(X.T, axis=0)
    Ynorm = norm(Y.T, axis=0)
    cos = np.clip(X@Y.T / np.outer(Xnorm, Ynorm), -1, 1)
    if nan_as_zero:
        cos[np.isnan(cos)] = 0
    if cos.size == 1:
        return float(cos[0][0])
    return cos.squeeze()
