"""Utility functions for matching object attributes."""
from typing import Any, Mapping, Literal, Callable, Optional


def match_spec(
    obj: Any,
    spec: Optional[Mapping] = None,
    /,
    _missing: Literal["ignore", "warn", "raise"] = "raise",
    **kwds: Any
) -> bool:
    """Match object properties with specification(s).

    Parameters
    ----------
    obj
        Object to test.
    spec
        Mapping from attribute names to desired values.
        Callables are used as predicate functions applied to attribute values
        that should return ``True``.
    _missing
        What to do when there fields in a specification
        which are not present in the object.
    **kwds
        Alternative way to pass spec.

    Raises
    ------
    ValueError
        When ``spec`` and ``kwds`` are used at the same time.
    """
    if spec and kwds:
        raise ValueError("'spec' and 'kwds' cannot be used at the same time")
    if spec is None:
        spec = kwds
    match = True
    for k, v in spec.items():
        try:
            attr = getattr(obj, k)
        except AttributeError as exc:
            if _missing == "ignore":
                pass
            elif _missing == "warn":
                # TODO: add warning
                pass
            elif _missing == "raise":
                raise exc
            else:
                raise ValueError(f"unknown '_missing' value '{_missing}'") from exc
        if isinstance(v, Callable):
            match &= v(attr)
        else:
            match &= v == attr
    if match:
        return True
    return False
