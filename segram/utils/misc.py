import numpy as np
from numpy.linalg import norm


def cosine_similarity(
    x: np.ndarray[tuple[int], np.floating],
    y: np.ndarray[tuple[int], np.floating],
    *,
    nan_as_zero: bool = True
) -> float:
    """Cosine similarity between two vectors."""
    xnorm = norm(x)
    ynorm = norm(y)
    if nan_as_zero and 0 in (xnorm, ynorm):
        # TODO: add warning
        return 0.0
    return np.clip(x.dot(y) / (norm(x)*norm(y)), 0, 1)
