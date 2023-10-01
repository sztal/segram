"""Enhanced :mod:`collections.abc` classes implementing
generic data filtering and transformation method.
"""
from typing import Self, Any, Callable, Literal
from typing import Iterable, Iterator, Sequence, Mapping
from types import MethodType
from abc import abstractmethod
from functools import total_ordering
from itertools import groupby, product, islice
from more_itertools import unique_everseen


class DataABC(Iterable):
    """Abstract base class for data classes."""
    __slots__ = ()

    @abstractmethod
    def __iter__(self) -> Iterable:
        pass

    def pipe(
        self,
        func: str | Callable[[Self, ...], Any],
        *args: Any,
        **kwds: Any
    ) -> Any:
        """Pipe self to a function."""
        func = self._handle_string_func(func)
        return func(self, *args, **kwds)

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


class DataIterable(DataABC):
    """Abstract base class for data iterables."""
    # pylint: disable=abstract-method

    # Properties --------------------------------------------------------------

    @property
    def flat(self) -> Self:
        return self.__class__(self.iter_flat())

    @property
    def list(self) -> "DataList":
        return self.pipe(DataList)

    @property
    def tuple(self) -> "DataTuple":
        return self.pipe(DataTuple)

    # Methods -----------------------------------------------------------------

    def get(self, attr: str) -> Self:
        """Extract attributes from data items."""
        return self.map(lambda m: getattr(m, attr))

    def any(self) -> bool:
        return any(self)

    def all(self) -> bool:
        return all(self)

    def map(self, func: str | Callable[[Any, ...], Any], *args: Any, **kwds: Any) -> Self:
        """Map data iterator."""
        func = self._handle_string_func(func)
        return self.__class__(func(x, *args, **kwds) for x in self)

    def filter(
        self,
        func: str | Callable[[Any, ...], bool] | None,
        *args: Any,
        **kwds: Any
    ) -> Self:
        """Filter data iterator."""
        if func is None:
            func = bool
        else:
            func = self._handle_string_func(func)
        return self.__class__(x for x in self if func(x, *args, **kwds))

    def unique(self, key: str | Callable[[Any, ...], Any] | None = None) -> Self:
        """Return unique values (only first unique occurences are returned)."""
        return self.__class__(self.pipe(unique_everseen, key=key))

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
        keyfunc = self._get_keyfunc(key, **kwds)
        groups = DataDict()
        data = DataTuple(self)
        for k, g in data.sort(key, **kwds).pipe(groupby, key=keyfunc):
            groups[k] = DataTuple(g)
        return groups

    def zip(self, iterable: Iterable) -> Self:
        """Zip with other iterable."""
        return self.__class__(zip(self, iterable))

    def iter_flat(self) -> Iterable:
        for obj in self:
            if isinstance(obj, DataIterable | tuple | list):
                yield from obj
            else:
                yield obj


class DataIterator(Iterator, DataIterable):
    """Data iterators class."""
    # pylint: disable=abstract-method
    __slots__ = ("__data__",)

    def __init__(self, data: Iterable, /) -> None:
        self.__data__ = iter(data)

    def __next__(self) -> Any:
        return next(self.__data__)

    def __getitem__(self, idx: int | slice) -> Any | Self:
        if isinstance(idx, slice):
            start = idx.start
            stop = idx.stop
            step = idx.step
            return self.__class__(islice(self, start, stop, step))
        return next(islice(self, idx, idx+1))


@total_ordering
class DataSequence(Sequence, DataIterable):
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
            Alternatively, an iterable of values used for sorting
            can be passed.
        show_keys
            Should sorting key values be returned
            together with the objects (so 2-tuples are returned).
        **kwds
            Passed to the sorting callable.
        """
        if args:
            key, *args = args
            if isinstance(key, str | Callable):
                keyfunc = self._get_keyfunc(key, *args, **kwds)
            else:
                keyfunc = key
        else:
            keyfunc = None
        if isinstance(keyfunc, Callable):
            data = sorted(self, key=keyfunc, reverse=reverse)
            if show_keys:
                data = zip(sorted(self.map(keyfunc), reverse=reverse), data)
        elif isinstance(keyfunc, Iterable):
            keyfunc = DataTuple(keyfunc)
            if len(self) != len(keyfunc):
                raise ValueError(
                    "sequence used for sorting must be of the same length as data"
                )
            data = sorted(zip(keyfunc, self), key=lambda x: x[0], reverse=reverse)
            if not show_keys:
                data = [ x[1] for x in data ]
        else:
            raise ValueError(
                "'key' must be a callable or its name or "
                "an iterable of sorting values"
            )
        return self.__class__(data)


class DataMapping(Mapping, DataABC):
    """Abstract base class for data mappings."""
    # pylint: disable=abstract-method
    _what_vals = ("items", "keys", "values")

    def keys(self) -> DataSequence:
        return DataTuple(super().keys())

    def values(self) -> DataSequence:
        return DataTuple(super().values())

    def items(self) -> DataSequence[tuple[Any, Any]]:
        return DataTuple(super().items())

    def map(self, _what: Literal[*_what_vals], *args: Any, **kwds: Any) -> Self:
        """Map over keys, values or items and return a transformed dictionary.

        Parameters
        ----------
        _what
            Part of dictionary to process.
        *args, **kwds
            Passed to :meth:`DataIteratorABC.map`.
        """
        if _what not in self._what_vals:
            raise ValueError(
                f"data dictionary can be mapped only over one of: {self._what_vals}"
            )
        if _what == "items":
            return self.__class__(self.items().map(*args, **kwds))
        keys = self.keys()
        vals = self.values()
        if _what == "keys":
            keys = keys.map(*args, **kwds)
        else:
            vals = vals.map(*args, **kwds)
        return self.__class__(zip(keys, vals))

    def filter(self, _what: Literal[*_what_vals], *args: Any, **kwds: Any) -> Self:
        """Filter dictonary by keys, values or items.

        Parameters
        ----------
        _what
            Part of dictionary to process.
        *args, **kwds
            Passed to :meth:`DataIteratorABC.map`.
        """
        if _what not in self._what_vals:
            raise ValueError(
                f"data dictionary can be filtered only over one of: {self._what_vals}"
            )
        if _what == "items":
            return self.__class__(self.items().filter(*args, **kwds))
        if args:
            func, *args = args
        if _what == "keys":
            flt = lambda item: func(item[0], *args, **kwds)
        else:
            flt = lambda item: func(item[1], *args, **kwds)
        return self.__class__(self.items().filter(flt))

    def sort(self, _what: Literal[*_what_vals], *args: Any, **kwds: Any) -> Self:
        """Sort dictionary.

        Parameters
        ----------
        _what
            Part of dictionary to process.
        *args, **kwds
            Passed to :meth:`DataSequenceABC.sort`,
            which is called on ``self.items()``.
        """
        return self._apply("sort", _what, *args, **kwds)

    # Internals ---------------------------------------------------------------

    def _apply(self, __method__: str, _what: str, *args: Any, **kwds: Any) -> Self:
        if _what not in self._what_vals:
            raise ValueError(
                f"data dictionary can be filtered only over one of: {self._what_vals}"
            )
        if args:
            func, *args = args
        else:
            func = None

        if _what == "keys":
            idx = 0
        elif _what == "values":
            idx = 1
        else:
            idx = None

        def flt(item, idx, func, *args, **kwds):
            if idx is not None:
                item = item[idx]
            if func is not None:
                item = func(item, *args, **kwds)
            return item

        method = getattr(self.items(), __method__)
        return self.__class__(method(flt, idx, func, *args, **kwds))


class DataTuple(tuple, DataSequence):
    """Data tuple class."""


class DataList(list, DataSequence):
    """Data list class."""


class DataDict(dict, DataMapping):
    """Data dict class."""
    keys = DataMapping.keys
    values = DataMapping.values
    items = DataMapping.items
