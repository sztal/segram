"""Enhanced :mod:`collections.abc` classes implementing
generic data filtering and transformation method.
"""
from typing import Self, Any, Iterable, Collection, Sequence, Callable
from types import MethodType
from abc import abstractmethod
from functools import total_ordering
from itertools import groupby
from ..abc import SegramABC


class DataCollectionABC(Collection, SegramABC):
    """Data colelction class."""
    __slots__ = ("members",)

    @abstractmethod
    def __init__(self, members: Collection[Any] = ()) -> None:
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

    def ifilter(
        self,
        func: str | Callable[[Any, ...], bool] | None,
        *args: Any,
        _drop_empty: bool = True,
        **kwds: Any
    ) -> Self:
        """Filter members iterator."""
        if func is None:
            func = bool
        else:
            func = self._handle_string_func(func)
        for member in self.members:
            if isinstance(member, DataCollectionABC):
                if (membs := member.filter(func, *args, **kwds)) \
                or not _drop_empty:
                    yield membs
            elif func(member, *args, **kwds):
                yield member

    def filter(
        self,
        func: str | Callable[[Any, ...], bool],
        *args: Any,
        _drop_empty: bool = True,
        **kwds: Any
    ) -> Self:
        """Filter members.

        Parameters
        ----------
        func
            Predicate function. If a string is passed then a method
            of the same name declared on items is used.
            If ``None`` then all falsy items are dropped.
        *args, **kwds
            Passed to ``func``.
        _drop_empty:
            Should nested containers that end up emtpy after
            filtering be dropped.
        """
        return self.copy(members=self.members.__class__(
            self.ifilter(func, *args, **kwds)
        ))

    def imap(
        self,
        func: str | Callable[[Any, ...], Any],
        *args: Any,
        **kwds: Any
    ) -> Self:
        """Map iterator."""
        func = self._handle_string_func(func)
        for member in self.members:
            if isinstance(member, DataCollectionABC):
                yield member.map(func, *args, **kwds)
            else:
                yield func(member, *args, **kwds)

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
        return self.copy(members=self.members.__class__(
            self.imap(func, *args, **kwds)
        ))

    def sort(
        self,
        key: str | Callable,
        *args: Any,
        reverse: bool = False,
        scores: bool = False,
        **kwds: Any
    ) -> Self:
        """Sort elements.

        It is typically best to first flatten the sequence
        in case it contains nested sequences.

        Parameters
        ----------
        by
            Name of an attribute or a method defined on items.
            Alternatively a callable.
        scores
            Should sorting scores be returned
            together with the objects (so 2-tuples are returned).
        *args, **kwds
            Passed to the sorting callable.
        """
        keyfunc = self._get_keyfunc(key, *args, **kwds)
        members = sorted(self.members, key=keyfunc, reverse=reverse)
        if scores:
            members = zip(sorted(self.map(keyfunc), reverse=reverse), members)
        return self.copy(members=members)

    def groupby(self, key: str | Callable, *args: Any, **kwds: Any) -> Self:
        """Group by key attribute or function/method.

        Importantly, the key function/values must be sortable.
        """
        members = []
        keyfunc = self._get_keyfunc(key, *args, **kwds)
        for _, group in self.sort(key, *args, **kwds).pipe(groupby, key=keyfunc):
            members.append(DataSequence(group))
        return DataChain(members)

    def get(self, attr: str) -> Self:
        """Extract attributes from members."""
        return self.map(lambda m: getattr(m, attr))

    def pipe(
        self,
        func: str | Callable[[Self, ...], Any],
        *args: Any,
        **kwds: Any
    ) -> Any:
        """Pipe self to a function."""
        func = self._handle_string_func(func)
        return func(self, *args, **kwds)

    def any(self) -> bool:
        return any(self)

    def all(self) -> bool:
        return all(self)

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

    @staticmethod
    def _get_keyfunc(func: str | Callable, *args: Any, **kwds: Any) -> Callable:
        def keyfunc(obj):
            nonlocal func
            key = func
            if isinstance(key, str):
                key = getattr(obj, key)
            if isinstance(key, MethodType):
                key = key(*args, **kwds)
            elif isinstance(key, Callable):
                key = key(obj, *args, **kwds)
            return key
        return keyfunc


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


class DataChain(DataSequence):
    """Chain of data sequences class."""
    __slots__ = ()

    def __init__(self, members: Sequence[Collection[Any]] = ()) -> None:
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
        return DataSequence(self.members)

    # Methods -----------------------------------------------------------------

    def flatten(self) -> DataSequence:
        return DataSequence(super().flatten().members)
