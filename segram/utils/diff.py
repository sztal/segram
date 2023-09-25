from typing import Any, Sized, Iterable
from collections.abc import Mapping
from functools import singledispatch
from .meta import get_ppath


DiffType = tuple[str, Any, Any]
IDiffType = Iterable[DiffType]


@singledispatch
def equal(obj, other, *, strict: bool = True) -> bool:
    """Compare two arbitrary objects.

    Objects are compared using standard equality operator,
    unless they are :class:`~segram.util.meta.SegramABC`
    instance, in which case the :meth:`~segram.util.meta.SegramABC.equal`
    method is used.

    Parameters
    ----------
    obj, other
        Arbitrary objects.
    strict
        Passed to ``.equal()`` method.
    """
    # pylint: disable=unused-argument
    return not any(iter_diffs(obj, other, strict=strict))

@singledispatch
def iter_diffs(obj, other, *, strict: bool = True) -> IDiffType:
    """Single dispatch generic function for iterating
    over differences between two objects.

    Parameters
    ----------
    obj, other
        Two arbitrary objects.
        The method dispatch is done on ``obj`` type.
    strict
        Used for comparing NLP token objects and grammar/semantic
        objects. When ``strict=True`` only exact matches on classes
        are accepted. This means, for instance, that only tokens
        from the same document can be equal, regardless of any
        text and index equivalences between two tokens and documents.

    Yields
    ------
    qname
        Qualified name of ``obj`` or its type.
    name
        Name of the attribute/property that differs between ``obj`` and ``other``.
    v1, v2
        Values that differ.
    """
    # pylint: disable=unused-argument
    if isinstance(other, type(obj)):
        if obj != other:
            yield "VALUE", obj, other
    else:
        yield "TYPE", get_ppath(obj), get_ppath(other)

@iter_diffs.register
def _(obj: tuple, other: tuple, *, strict: bool = True) -> IDiffType:
    # pylint: disable=unused-argument
    if (diff := next(_iter_diff_size(obj, other), None)):
        yield diff
        return
    yield from _iter_diff_iterable(obj, other, strict=strict)

@iter_diffs.register
def _(obj: list, other: list, *, strict: bool = True) -> IDiffType:
    # pylint: disable=unused-argument
    return iter_diffs(tuple(obj), tuple(other), strict=strict)

@iter_diffs.register
def _(obj: Mapping, other: Mapping, *, strict: bool = True) -> IDiffType:
    # pylint: disable=unused-argument
    if (diff := next(_iter_diff_size(obj, other), None)):
        yield diff
        return
    yield from _iter_diff_mapping(obj, other, strict=strict)

# Internals ---------------------------------------------------------------

def _iter_diff_size(obj: Sized, other: Sized) -> IDiffType:
    if len(obj) != len(other):
        yield "SIZE", obj, other

def _iter_diff_iterable(obj: Iterable, other: Iterable, *, strict: bool = True) -> IDiffType:
    for i, xy in enumerate(zip(obj, other)):
        x, y = xy
        if not equal(x, y, strict=strict):
            yield f"INDEX={i}", x, y
            yield from iter_diffs(x, y, strict=strict)

def _iter_diff_mapping(obj: Mapping, other: Mapping, *, strict: bool = True) -> IDiffType:
    for kv1, kv2 in zip(
        sorted(obj.items(), key=lambda x: x[0]),
        sorted(other.items(), key=lambda x: x[0])
    ):
        k1, v1 = kv1
        k2, v2 = kv2
        if not equal(k1, k2, strict=strict):
            yield "KEYS", k1, k2
            continue
        if not equal(v1, v2, strict=strict):
            yield k1, v1 ,v2
            yield from iter_diffs(v1, v2, strict=strict)
