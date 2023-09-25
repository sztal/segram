from __future__ import annotations
from typing import Any, Callable


class Matcher:
    """Matcher class.

    Attributes
    ----------
    func
        Matching function.
        If ``None`` then fallback to :meth:`match` is attempted.
        The `match` method should be defined as a static method.
    """
    def __init__(self, func: Callable | None = None) -> None:
        self.func = func

    def __call__(self, obj: Any) -> bool:
        if self.func is not None:
            return self.func(obj)
        return self.match(obj)

    def __and__(self, other: Matcher) -> Matcher:
        if isinstance(other, Matcher):
            return Matcher(lambda obj: self(obj) and other(obj))
        if isinstance(other, Callable):
            return self & Matcher(other)
        return NotImplemented

    def __rand__(self, other: Matcher) -> Matcher:
        return self & other

    def __or__(self, other: Matcher) -> Matcher:
        if isinstance(other, Matcher):
            return Matcher(lambda obj: self(obj) or other(obj))
        if isinstance(other, Callable):
            return self | Matcher(other)
        return NotImplemented

    def __ror__(self, other: Matcher) -> Matcher:
        return self | other

    @staticmethod
    def match(obj: None) -> bool:
        """Default matching function."""
        raise NotImplementedError("default matching function not implemented")
