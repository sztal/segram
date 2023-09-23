from __future__ import annotations
from typing import Any, Optional, Callable
from functools import total_ordering
from collections.abc import MutableMapping
from collections.abc import Sequence, Iterable, Iterator
from .meta import get_cname
from ..abc import SegramABC


class Namespace(MutableMapping):
    """Namespace class.

    It behaves like a dictionary with both item and attribute
    getters, setters and deletters.
    """
    def __init__(self, *args: Any, **kwds: Any) -> None:
        try:
            dct = dict(*args, **kwds)
        except TypeError as exc:
            raise TypeError(f"'{get_cname(self)}' {str(exc)}") from exc
        self.__dict__.update(dct)

    def __repr__(self) -> str:
        return f"{get_cname(self)}({self.__dict__})"

    def __iter__(self) -> Iterator[str]:
        yield from self.__dict__

    def __len__(self) -> int:
        return len(self.__dict__)

    def __getitem__(self, name: str) -> Any :
        return self.__dict__[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value

    def __delitem__(self, name: str) -> None:
        del self.__dict__[name]

    def __contains__(self, name: str) -> bool:
        return name in self.__dict__

    @property
    def names(self) -> list[str]:
        return list(self)


@total_ordering
class Group(Sequence, SegramABC):
    """Group of arbitrary coordinated objects.

    Subclasses can define additional slots corresponding
    to coordinating objects (e.g. coordinating conjunctions),
    which can be jointly returned as a single tuple through
    ``cconjs`` property, in which case they will be included
    automatically in comparison methods and ``__repr__()``.

    Attributes
    ----------
    members
        Sequence of objects.
        Stored as a tuple, so it is not mutable and can be safely hashed
        if the stored objects are hashable themselves.
    """
    __slots__ = ("members",)

    def __init__(self, members: Iterable[Any] = ()) -> None:
        self.members = tuple(members)

    def __repr__(self) -> str:
        return self.to_str()

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Any) -> bool:
        if self.is_comparable_with(other):
            return tuple(self) == tuple(other)
        return NotImplemented

    def __lt__(self, other: Any) -> bool:
        if self.is_comparable_with(other):
            return tuple(self) < tuple(other)
        return NotImplemented

    def __getitem__(self, idx: int | slice) -> tuple[Any, ...]:
        return self.members[idx]

    def __len__(self) -> int:
        return len(self.members)

    # Properties --------------------------------------------------------------

    @property
    def hashdata(self) -> tuple[Any, ...]:
        return (self.members,)

    # Methods -----------------------------------------------------------------

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, Sequence)

    def to_str(self, **kwds: Any) -> str:
        """Represent as string."""
        # pylint: disable=unused-argument
        return str(self.members)


@total_ordering
class ChainGroup(Group):
    """Chain of groups.

    Attributes
    ----------
    members
        Sequence of group objects.
    """
    __slots__ = ()

    def __init__(self, members: Iterable[Group] = ()) -> None:
        members = tuple(
            Group(m) if not isinstance(m, Group) else m
            for m in members
        )
        super().__init__(members)

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Sequence) -> bool:
        if self.is_comparable_with(other):
            return self.flat == tuple(other)
        return NotImplemented

    def __lt__(self, other: Sequence) -> bool:
        if self.is_comparable_with(other):
            return self.flat < tuple(other)
        return NotImplemented

    def __len__(self) -> int:
        return len(self.flat)

    def __getitem__(self, idx: int | slice) -> Any | tuple[Any, ...]:
        return self.flat[idx]

    # Properties --------------------------------------------------------------

    @property
    def groups(self) -> tuple[Group, ...]:
        return self.members

    @property
    def flat(self) -> tuple[Any, ...]:
        return tuple(m for g in self.members for m in g)

    @property
    def hashdata(self) -> tuple[Any, ...]:
        return self.members

    # Methods -----------------------------------------------------------------

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, Sequence)

    def to_str(self, *, color: bool = True, **kwds: Any) -> str:
        """Represent as string."""
        s = ", ".join(g.to_str(color=color, **kwds) for g in self.members)
        return f"({s})"


class Matcher:
    """Matcher class.

    Attributes
    ----------
    func
        Matching function.
        If ``None`` then fallback to :meth:`match` is attempted.
        The `match` method should be defined as a static method.
    """
    def __init__(self, func: Optional[Callable] = None) -> None:
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
