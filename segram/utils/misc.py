import numpy as np
from numpy.linalg import norm


def cosine_similarity(
    X: np.ndarray[tuple[int] | tuple[int, int], np.floating],
    Y: np.ndarray[tuple[int] | tuple[int, int], np.floating],
    *,
    nan_as_zero: bool = True,
    aligned: bool = False
) -> float | np.ndarray[tuple[int, ...], np.floating]:
    """Cosine similarity between two vectors.

    When 2D arrays are passed it is assumed that vectors
    for calculating similarities are arranged in rows.

    Parameters
    ----------
    X, Y
        Vectors or arrays of vectors.
    nan_as_zero
        Should NaN values be treated as zero similarities.
    aligned
        If ``True`` then ``X`` and ``Y`` have to be 2D and of the
        same shape and row-by-row similarities are calculated.
    """
    if aligned:
        if X.ndim != 2:
            raise ValueError("'X' and 'Y' must be 2D when 'aligned=True'")
        if X.shape != Y.shape:
            raise ValueError("'X' and 'Y' have to be of the same shape when 'aligned=True'")
        Xnorm = np.linalg.norm(X, axis=1)
        Ynorm = np.linalg.norm(Y, axis=1)
        sim = (X*Y).sum(axis=1) / (Xnorm*Ynorm)
        return sim
    Xnorm = norm(X.T, axis=0)
    Ynorm = norm(Y.T, axis=0)
    cos = np.clip(X@Y.T / np.outer(Xnorm, Ynorm), -1, 1)
    if nan_as_zero:
        cos[np.isnan(cos)] = 0
    if cos.size == 1:
        return float(cos[0][0])
    return cos.squeeze()
