from typing import Any, Callable, Iterable, Mapping
from itertools import product
from more_itertools import unique_everseen
import numpy as np
from numpy.linalg import norm


def cosine_similarity(
    X: np.ndarray[tuple[int] | tuple[int, int], np.floating],
    Y: np.ndarray[tuple[int] | tuple[int, int], np.floating],
    *,
    aligned: bool = False
) -> float | np.ndarray[tuple[int, ...], np.floating]:
    """Cosine similarity between two vectors.

    When 2D arrays are passed it is assumed that vectors
    for calculating similarities are arranged in rows.

    Parameters
    ----------
    X, Y
        Vectors or arrays of vectors.
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
        sim = (X*Y).sum(axis=1)
        mask = (Xnorm != 0) & (Ynorm != 0)
        sim = sim[mask] / (Xnorm*Ynorm)[mask]
        return sim
    Xnorm = norm(X.T, axis=0)
    Ynorm = norm(Y.T, axis=0)
    Xnz = Xnorm != 0
    Ynz = Ynorm != 0
    cos = (X@Y.T)[Xnz][:, Ynz]
    cos = np.clip(cos / np.outer(Xnorm[Xnz], Ynorm[Ynz]), -1, 1)
    if cos.size == 1:
        return float(cos[0][0])
    return cos.squeeze()


def best_matches(
    objs: Iterable,
    others: Iterable,
    func: Callable[[Any, Any], int | float],
    *args: Any,
    **kwds: Any
) -> Iterable[tuple[int | float, Any, Any]]:
    objs = tuple(objs)
    others = tuple(others)
    idx = 1 if len(objs) <= len(others) else 2
    pairs = sorted((
        (func(obj, other, *args, **kwds), obj, other)
        for obj, other in product(objs, others)
    ), key=lambda x: -x[0])
    yield from unique_everseen(pairs, key=lambda x: x[idx])


def sort_map(mapping: Mapping) -> Mapping:
    return mapping.__class__({
        k: v for k, v
        in sorted(mapping.items(), key=lambda x: x[1])
    })

def stringify(obj: Any, **kwds: Any) -> str:
    """Convert ``obj`` to string.

    If ``obj`` exposes ``to_str()`` then it is used
    with keyword arguments passed in ``**kwds``.
    Otherwise the plain ``__repr__()`` is used.
    """
    if (to_str := getattr(obj, "to_str", None)):
        return to_str(**kwds)
    return repr(obj)
