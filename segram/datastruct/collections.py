"""Enhanced :mod:`collections.abc` classes implementing
generic data filtering and transformation method.
"""
from typing import Self, Any, Callable, Sequence, Iterable
from types import MethodType
from abc import abstractmethod
from functools import total_ordering
from itertools import groupby, product


class DataIterableABC(Iterable):
    """Abstract base class for data iterables."""
    __slots__ = ()

    @abstractmethod
    def __iter__(self) -> Iterable:
        pass

    # Properties --------------------------------------------------------------

    @property
    def flat(self) -> Self:
        return self.__class__(self.iter_flat())

    # Methods -----------------------------------------------------------------

    def filter(
        self,
        func: str | Callable[[Any, ...], bool] | None,
        *args: Any,
        _drop_empty: bool = True,
        **kwds: Any
    ) -> Self:
        """Filter data iterator."""
        if func is None:
            func = bool
        else:
            func = self._handle_string_func(func)
        def _iter():
            for obj in self:
                if isinstance(obj, DataIterableABC):
                    if (subs := obj.filter(func, *args, **kwds)) \
                    or not _drop_empty:
                        yield subs
                elif func(obj, *args, **kwds):
                    yield obj
        return self.__class__(_iter())

    def map(self, func: str | Callable[[Any, ...], Any], *args: Any, **kwds: Any) -> Self:
        """Map data iterator."""
        func = self._handle_string_func(func)
        def _iter():
            for obj in self:
                if isinstance(obj, DataIterableABC):
                    yield obj.map(func, *args, **kwds)
                else:
                    yield func(obj, *args, **kwds)
        return self.__class__(_iter())

    def pipe(
        self,
        func: str | Callable[[Self, ...], Any],
        *args: Any,
        **kwds: Any
    ) -> Any:
        """Pipe self to a function."""
        func = self._handle_string_func(func)
        return func(self, *args, **kwds)

    def get(self, attr: str) -> Self:
        """Extract attributes from data items."""
        return self.map(lambda m: getattr(m, attr))

    def any(self) -> bool:
        return any(self)

    def all(self) -> bool:
        return all(self)

    def iter_flat(self) -> Iterable:
        for obj in self:
            if isinstance(obj, DataIterableABC | tuple | list):
                yield from obj
            else:
                yield obj

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


class DataIterable(DataIterableABC):
    """Data iterable class."""
    __slots__ = ("__data__",)

    def __init__(self, data: Iterable, /) -> None:
        self.__data__ = data

    def __iter__(self) -> Iterable:
        yield from self.__data__


@total_ordering
class DataSequenceABC(Sequence, DataIterableABC):
    """Data sequence class."""

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __getitem__(self, idx: int | slice) -> Any | Self:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass

    @abstractmethod
    def __lt__(self, other: Any) -> bool:
        pass

    def pairwise(self) -> Iterable[tuple[Any, Any]]:
        """Iterate over all pairs of data items."""
        return self.__class__(product(self, self))

    def sort(
        self,
        *args: Any,
        reverse: bool = False,
        show_keys: bool = False,
        **kwds: Any
    ) -> Self:
        """Sort elements.

        It is typically best to first flatten the sequence
        in case it contains nested sequences.

        Parameters
        ----------
        *args
            Name of an attribute or a method defined on items.
            Alternatively a callable. Further positional arguments
            are passed to the key function. If no positional arguments
            are used then standard data item sorting is used.
        show_keys
            Should sorting key values be returned
            together with the objects (so 2-tuples are returned).
        **kwds
            Passed to the sorting callable.
        """
        if args:
            key, *args = args
            keyfunc = self._get_keyfunc(key, *args, **kwds)
        else:
            keyfunc = None
        data = sorted(self, key=keyfunc, reverse=reverse)
        if keyfunc and show_keys:
            data = zip(sorted(self.map(keyfunc), reverse=reverse), data)
        return self.__class__(data)

    def groupby(self, *args: Any, **kwds: Any) -> Self:
        """Group by key attribute or function/method.

        Importantly, the key function/values must be sortable.

        Parameters
        ----------
        *args
            First argument is interpreted as key function (or its name).
            The rest is passed as actual ``*args`` to the function.
            No grouping is done if no function/name is passed.
        **kwds
            Passed to the function.
        """
        if not args:
            return self
        key, *args = args
        groups = []
        keyfunc = self._get_keyfunc(key, **kwds)
        for key, group in self.sort(key, **kwds).pipe(groupby, key=keyfunc):
            if not isinstance(group, DataChain):
                group = DataTuple(group)
            groups.append(group)
        return DataChain(groups)


class DataTuple(tuple, DataSequenceABC):
    """Data tuple class."""


class DataList(list, DataSequenceABC):
    """Data list class."""


class DataChain(DataSequenceABC):
    """Chain of data tuples class."""
    __slots__ = ("__data__",)

    def __init__(self, data: Iterable = ()) -> None:
        self.__data__ = DataTuple(
            x if isinstance(x, DataSequenceABC) else DataTuple(x)
            for x in data
        )

    def __repr__(self) -> str:
        return repr(self.__data__)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DataChain):
            return self.__data__ == other.__data__
        return False

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, DataChain):
            return self.__data__ < other.__data__
        return NotImplemented

    def __iter__(self) -> Iterable[Any]:
        yield from self.iter_flat()

    def __getitem__(self, idx: int | slice) -> Any:
        if isinstance(idx, slice):
            return DataTuple(self.flat[idx])
        return self.flat[idx]

    def __len__(self) -> int:
        return sum(1 for _ in self.flat)

    # Properties --------------------------------------------------------------

    @property
    def groups(self) -> DataTuple:
        return self.__data__

    @property
    def flat(self) -> DataTuple:
        return DataTuple(self.iter_flat())

    # Methods -----------------------------------------------------------------

    def iter_flat(self) -> Iterable:
        for obj in self.__data__:
            if isinstance(obj, DataIterableABC | tuple | list):
                yield from obj
            else:
                yield obj
