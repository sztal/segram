# pylint: disable=no-name-in-module
from typing import Any, Callable, Iterable, Mapping
from itertools import product
from more_itertools import unique_everseen
import numpy as np
from numpy.linalg import norm
from spacy.vocab import Vocab


def cosine_similarity(
    X: np.ndarray[tuple[int] | tuple[int, int], np.floating],
    Y: np.ndarray[tuple[int] | tuple[int, int], np.floating],
    *,
    aligned: bool = False,
    nans_as_zeros: bool = True
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
    nans_as_zeros
        Should NaN values arising from zero vector norm
        be interpreted as zero similarities.
    """
    if aligned:
        if X.ndim != 2:
            raise ValueError("'X' and 'Y' must be 2D when 'aligned=True'")
        if X.shape != Y.shape:
            raise ValueError("'X' and 'Y' have to be of the same shape when 'aligned=True'")
        Xnorm = np.linalg.norm(X, axis=1)
        Ynorm = np.linalg.norm(Y, axis=1)
        sim = (X*Y).sum(axis=1)
        if nans_as_zeros:
            mask = (Xnorm != 0) & (Ynorm != 0)
            sim = sim[mask] / (Xnorm*Ynorm)[mask]
        else:
            sim /= Xnorm*Ynorm
        return sim
    Xnorm = norm(X.T, axis=0)
    Ynorm = norm(Y.T, axis=0)
    if nans_as_zeros:
        Xnz = Xnorm != 0
        Ynz = Ynorm != 0
        cos = (X@Y.T)[Xnz][:, Ynz]
        cos = np.clip(cos / np.outer(Xnorm[Xnz], Ynorm[Ynz]), -1, 1)
    else:
        cos = X@Y.T
        cos = np.clip(cos / np.outer(Xnorm, Ynorm), -1, 1)
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
    return mapping.__class__(sorted(mapping.items(), key=lambda x: x[0]))


def stringify(obj: Any, **kwds: Any) -> str:
    """Convert ``obj`` to string.

    If ``obj`` exposes ``to_str()`` then it is used
    with keyword arguments passed in ``**kwds``.
    Otherwise the plain ``__repr__()`` is used.
    """
    if (to_str := getattr(obj, "to_str", None)):
        return to_str(**kwds)
    return repr(obj)


def ensure_cpu_vectors(vocab: Vocab | Any) -> None:
    """Ensure that word vectors are stored on CPU.

    Parameters
    ----------
    vocab
        Vocabulary object.
        If an arbitrary object is passed then an attempt
        at retrieving ``.vocab`` attribute is made.
    """
    if not isinstance(vocab, Vocab):
        vocab = vocab.vocab
    if not isinstance(vocab.vectors.data, np.ndarray):
        vocab.vectors.data = vocab.vectors.data.get()

def prefer_gpu_vectors(
    vocab: Vocab | Any,
    device_id: int | None = None
) -> bool:
    """Store word vectors on GPU if possible.

    Parameters
    ----------
    Vocabulary object.
        If an arbitrary object is passed then an attempt
        at retrieving ``.vocab`` attribute is made.
    device_id
        GPU device id. If ``None`` then the default device
        is used (typically it is with id ``0``).

    Returns
    -------
    bool
        Specifies whether the vectors where successfully moved to GPU.
    """
    if not isinstance(vocab, Vocab):
        vocab = vocab.vocab
    data = vocab.vectors.data
    if isinstance(data, np.ndarray):
        try:
            import cupy as cp # pylint: disable=import-outside-toplevel
        except ImportError:
            return False
        if device_id is not None:
            with cp.cuda.Device(device_id):
                data = cp.asarray(data)
        else:
            data = cp.asarray(data)
        vocab.vectors.data = data
    return True
