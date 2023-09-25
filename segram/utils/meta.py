"""Metaprogramming utilities."""
from typing import Any
from types import FunctionType, MethodType


def get_cname(obj: Any) -> str:
    """Get class name."""
    if not isinstance(obj, type):
        obj = type(obj)
    return obj.__name__

def get_ppath(obj: Any) -> str:
    """Get full python path a named python object."""
    if not isinstance(obj, type | FunctionType | MethodType):
        obj = type(obj)
    return f"{obj.__module__}.{obj.__qualname__}"

def init_class_attrs(
    cls,
    attrs: dict[str, str],
    *,
    check_slots: bool = True
) -> None:
    """Initialize special class attributes if they are not
    already defined and set final values.

    Parameters
    ----------
    attrs
        Dictionary from class special attribute names
        to the names of final attributes.
    check_slots
        Check if attribute names are correctly
        declared as slots.
    """
    for attr in attrs:
        if attr not in cls.__dict__:
            setattr(cls, attr, ())
    for attr, final in attrs.items():
        slots = getattr(cls, attr)
        if check_slots and (incorrect := set(slots) - set(cls.__slots__)):
            raise TypeError(
                f"names in '{attr}' are not declared "
                f"in '__slots__': {tuple(incorrect)}"
            )
        names = []
        for typ in reversed(cls.mro()):
            names.extend(typ.__dict__.get(attr, ()))
        if len(names) != len(set(names)):
            raise TypeError(f"repeated '{attr}' slots: {names}")
        setattr(cls, final, tuple(names))
