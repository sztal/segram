"""Enhanced :mod:`collections.abc` classes implementing
generic data filtering and transformation method.
"""
from typing import Self, Any, Iterable, Sequence, Callable
from typing import ClassVar, Mapping, MutableSequence
from types import MethodType
from abc import ABC, abstractmethod
from copy import copy
from functools import total_ordering
from itertools import groupby, product
from ..utils.meta import init_class_attrs


class DataABC(ABC):
    """Abstract base class for data collections."""
    __slots__ = ("__data__",)
    slot_names: ClassVar[tuple[str, ...]] = ()

    def __init__(self, data: Iterable = ()) -> None:
        self.__data__ = data

    def __repr__(self) -> str:
        return repr(self.__data__)

    def __init_subclass__(cls, interface: type | None = None) -> None:
        init_class_attrs(cls, {
            "__slots__": "slot_names"
        }, check_slots=True)

    # Methods -----------------------------------------------------------------

    def copy(self, **kwds: Any) -> Self:
        return self.__class__({ "data": copy(self.__data__), **kwds })

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


class DataIterable(Iterable, DataABC):
    """Abstract base class for data iterables."""

    def __iter__(self) -> Iterable:
        yield from self.__data__

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
                if isinstance(obj, DataIterable):
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
                if isinstance(obj, DataIterable):
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
        for obj in self.__data__:
            if isinstance(obj, Mapping) \
            or not isinstance(obj, DataIterable | tuple | list):
                yield obj
            elif isinstance(obj, DataIterable):
                yield from obj.iter_flat()
            else:
                yield from obj

@total_ordering
class DataSequence(Sequence, DataIterable):
    """Data sequence class."""

    def __getitem__(self, idx: int | slice) -> Any | Self:
        if isinstance(idx, int):
            return self.__data__[idx]
        return self.__class__(self.__data__[idx])

    def __len__(self) -> int:
        return len(self.__data__)

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        """Equality comparison."""

    @abstractmethod
    def __lt__(self, other: Any) -> bool:
        """Lower than comparison."""

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
        data = sorted(self.__data__, key=keyfunc, reverse=reverse)
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


class DataTuple(DataSequence):
    """Data tuple class."""
    def __init__(self, data: Iterable = ()) -> None:
        super().__init__(tuple(data))

    def __hash__(self) -> int:
        return hash(self.__data__)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DataTuple | tuple):
            return self.__data__ == tuple(other)
        return False

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, DataTuple | tuple):
            return NotImplemented
        return self.__data__ < tuple(other)


class DataList(MutableSequence, DataTuple):
    """Data list class."""
    def __init__(self, data: Iterable = ()) -> None:
        super().__init__(list(data))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DataList | list):
            return self.__data__ == list(other)
        return False

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, DataList | list):
            return NotImplemented
        return self.__data__ < list(other)

    def __setitem__(self, idx: int | slice, value: Any) -> None:
        self.__data__[idx] = value

    def __delitem__(self, idx: int | slice) -> None:
        del self.__data__[idx]

    def insert(self, index: int, value: Any) -> None:
        self.__data__.insert(index, value)


class DataChain(DataTuple):
    """Chain of data tuples class."""
    def __init__(self, data: Iterable = ()) -> None:
        super().__init__(tuple(
            x if isinstance(x, DataChain) else DataTuple(x)
            for x in data
        ))

    def __iter__(self) -> Iterable[Any]:
        yield from self.iter_flat()

    def __getitem__(self, idx: int | slice) -> Any:
        return DataTuple(tuple(self.flat[idx]))

    def __len__(self) -> int:
        return sum(1 for _ in self.flat)

    # Properties --------------------------------------------------------------

    @property
    def groups(self) -> DataTuple:
        return DataTuple(self.__data__)

    @property
    def flat(self) -> DataTuple:
        return DataTuple(self.iter_flat())
