from __future__ import annotations
from typing import Any, Final, Callable
from functools import total_ordering
from abc import ABC, abstractmethod
from ...utils.meta import init_class_attrs


def attr(prop: Callable) -> Callable:
    """Mark property as token attribute,
    so its name is stored in ``__attrs__``.
    """
    prop.fget.__is_attr__ = True
    return prop


class NLP(ABC):
    """Abstract base class for NLP objects."""
    __slots__ = ()
    __attrs__: Final[tuple[str, ...]] = ()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "__slots__" not in cls.__dict__:
            raise TypeError(f"'{cls.__name__}' does not define '__slots__'")
        init_class_attrs(cls, {
            "__slots__": "slot_names"
        })
        try:
            total_ordering(cls)
        except ValueError:
            pass
        if "__attrs__" in cls.__dict__:
            raise AttributeError("'__attrs__' class attribute cannot be defined on subclasses")
        __attrs__ = list(cls.__attrs__)
        for n, a in vars(cls).items():
            if isinstance(a, property) and getattr(a.fget, "__is_attr__", False):
                if n in __attrs__:
                    raise AttributeError(f"'{n}' is already defined in '__attrs__': {__attrs__}")
                __attrs__.append(n)
        cls.__attrs__ = tuple(__attrs__)


class NLPToken(NLP):
    """ABC for defning generic NLP tokens."""
    __slots__ = ()

    def __repr__(self) -> str:
        """String representation."""
        return self.text

    def __eq__(self, other: NLPToken) -> bool:
        """Check equality with another token of the same type."""
        if self.is_comparable_with(other) is True:
            return self.doc == other.doc
        return NotImplemented

    @abstractmethod
    def __hash__(self) -> int:
        pass

    # Properties --------------------------------------------------------------

    @property
    def lang(self) -> str:
        return self.doc.lang

    @property
    def attrs(self) -> tuple[Any, ...]:
        return tuple(getattr(self, attr) for attr in self.__attrs__)

    # Abstract methods --------------------------------------------------------

    @attr
    @property
    @abstractmethod
    def text(self) -> str:
        """Raw text."""

    @property
    @abstractmethod
    def doc(self) -> "DocABC":
        """Parent document object (or ``self`` for documents)."""

    # Methods -----------------------------------------------------------------

    def is_comparable_with(self, other: Any) -> bool:
        """Check if ``self`` defines the same abstract interface as ``other``."""
        if not isinstance(other, NLPToken):
            return NotImplemented
        if self.doc is not other.doc:
            raise ValueError("'self' and 'other' are based on different documents")
        return isinstance(other, self.__class__) or NotImplemented
