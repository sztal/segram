"""Enhanced :mod:`collections.abc` classes implementing
generic data filtering and transformation method.
"""
from typing import Self, Any, Iterable, Collection, Sequence, Callable
from abc import abstractmethod
from functools import total_ordering
from ..abc import SegramABC


class DataCollectionABC(Collection, SegramABC):
    """Data colelction class."""
    __slots__ = ("members",)

    @abstractmethod
    def __init__(self, members: Collection[Any]) -> None:
        """Initialization method."""
        if not isinstance(members, DataCollectionABC):
            members = tuple(members)
        self.members = members

    def __eq__(self, other: Any) -> bool:
        if not self.is_comparable_with(other):
            return NotImplemented
        if isinstance(other, DataCollectionABC):
            return self.members == other.members
        return self.members == other

    def __hash__(self) -> int:
        return super().__hash__()

    def __iter__(self) -> Iterable[Any]:
        yield from self.members

    def __len__(self) -> int:
        return len(self.members)

    def __contains__(self, other: Any) -> bool:
        return other in self.members

    def __repr__(self) -> str:
        return repr(self.members)

    # Properties --------------------------------------------------------------

    @property
    def hashdata(self) -> tuple[Any, ...]:
        return (self.members,)

    @property
    def flat(self) -> Self:
        return self.flatten()

    # Methods -----------------------------------------------------------------

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, Collection)

    def filter(
        self,
        func: str | Callable[[Any, ...], bool],
        *args: Any,
        **kwds: Any
    ) -> Self:
        """Filter members.

        Parameters
        ----------
        func
            Predicate function. If a string is passed then a method
            of the same name declared on items is used.
        *args, **kwds
            Passed to ``func``.
        """
        func = self._handle_string_func(func)
        return self.copy(members=self.members.__class__(
            m for m in self.members if func(m, *args, **kwds)
        ))

    def map(
        self,
        func: str | Callable[[Any, ...], Any],
        *args: Any,
        **kwds: Any
    ) -> Self:
        """Map function over members.

        Parameters
        ----------
        func
            Predicate function. If a string is passed then a method
            of the same name declared on items is used.
        *args, **kwds
            Passed to ``func``.
        """
        func = self._handle_string_func(func)
        return self.copy(members=self.members.__class__(
            func(m, *args, **kwds) for m in self.members
        ))

    def pipe(
        self,
        func: str | Callable[[Self, ...], Any],
        *args: Any,
        **kwds: Any
    ) -> Any:
        """Pipe self to a function."""
        func = self._handle_string_func(func)
        return func(self, *args, **kwds)

    def flatten(self) -> Self:
        """Flatten nested data."""
        return self.__class__(self.iter_flat())

    def iter_flat(self) -> Self:
        for member in self.members:
            if isinstance(member, DataCollectionABC):
                yield from member.iter_flat()
            else:
                yield member

    # Internals ---------------------------------------------------------------

    @staticmethod
    def _handle_string_func(func: str | Callable) -> Callable:
        if isinstance(func, str):
            def _func(o, *args, **kwds):
                return getattr(o, func)(*args, **kwds)
            return _func
        return func


@total_ordering
class DataSequence(DataCollectionABC, Sequence):
    """Data sequence class."""
    __slots__ = ()

    def __init__(self, members: Sequence[Any]) -> None:
        super().__init__(members)

    def __getitem__(self, idx: int | slice) -> Any:
        if isinstance(idx, slice):
            return self.copy(members=self.members[idx])
        return self.members[idx]

    def __lt__(self, other: Sequence) -> bool:
        if not self.is_comparable_with(other):
            return NotImplemented
        if isinstance(other, DataSequence):
            return self.members < other.members
        return self.members < other

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(self, Sequence)


class DataGrouped(DataSequence):
    """Data grouped sequence class."""
    __slots__ = ()

    def __init__(self, members: Sequence[Collection[Any]]) -> None:
        super().__init__(members)

    def __iter__(self) -> Iterable[Any]:
        yield from self.flat

    def __getitem__(self, idx: int | slice) -> Any:
        return self.flat[idx]

    def __len__(self) -> int:
        return sum(1 for _ in self.flat)

    # Properties --------------------------------------------------------------

    @property
    def groups(self) -> DataSequence:
        return self.members

    # Methods -----------------------------------------------------------------

    def flatten(self) -> DataSequence:
        return DataSequence(super().flatten().members)
