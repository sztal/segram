from typing import Any, ClassVar, Self, Callable
from abc import ABC, abstractmethod
from .utils.docstrings import inherit_docstring
from .utils.diff import iter_diffs, IDiffType
from .utils.meta import init_class_attrs, get_cname, get_ppath
from .utils.misc import stringify


def labelled(label: str) -> Callable:
    """Assign ``label`` as attribute ``attr`` to a function."""
    def decorator(func: Callable) -> Callable:
        if isinstance(func, property):
            target = func.fget
        else:
            target = func
        setattr(target, "__group_label__", label)
        return func
    return decorator


class SegramABC(ABC):
    """Abstract base class for specialized :mod:`segram` classes."""
    __slots__ = ("_hashdata",)
    slot_names: ClassVar[tuple[str, ...]] = ()
    differ: type["Differ"]

    def __init__(self) -> None:
        self._hashdata = None

    def __hash__(self) -> int:
        return hash(self.hashdata)

    def __init_subclass__(cls) -> None:
        if "__slots__" not in cls.__dict__:
            raise TypeError(f"'{cls.__name__}' does not define '__slots__'")
        cls.init_class_attrs({
            "__slots__": "slot_names",
        }, check_slots=True)
        if len(cls.slot_names) != len(set(cls.slot_names)):
            raise TypeError(f"'__slots__' are not unique: {cls.slot_names}")
        # Handle labelled methods
        for name, attr in vars(cls).items():
            target = attr.fget if isinstance(attr, property) else attr
            if (label := getattr(target, "__group_label__", None)):
                names_attr = f"{label}_names"
                names = getattr(cls, names_attr, ())
                setattr(cls, names_attr, (*names, name))
        inherit_docstring(cls)

    # Abstract methods --------------------------------------------------------

    @abstractmethod
    def is_comparable_with(self, other: Any) -> None:
        """Are ``self`` and ``other`` comparable."""
        raise NotImplementedError

    # Properties --------------------------------------------------------------

    @property
    def hashdata(self) -> tuple[Any, ...]:
        """Tuple with hashable objects used for calculating instance hash."""
        if self._hashdata is None:
            self._hashdata = self.get_hashdata()
        return self._hashdata

    @property
    def data(self) -> dict[str, Any]:
        """Dictionary mapping names and values for main slots."""
        return {
            n: getattr(self, n)
            for n in self.slot_names if not n.startswith("_")
        }

    # Methods -----------------------------------------------------------------

    def get_hashdata(self) -> None:
        """Get data used for generating object hash."""
        raise NotImplementedError(f"'{self.cname()}' is not hashable")

    @classmethod
    def cname(cls, obj: Any | None = None) -> str:
        """Get class name."""
        return get_cname(obj if obj is not None else cls)

    @classmethod
    def ppath(cls, obj: Any | None = None) -> str:
        """Get full python path of the class."""
        return get_ppath(obj if obj is not None else cls)

    @classmethod
    def init_class_attrs(cls, attrs: dict[str, str], **kwds: Any) -> None:
        """Initialize special class attributes if they are not
        already defined and set final values.

        See :func:`init_class_attrs` for details.
        """
        init_class_attrs(cls, attrs, **kwds)

    def copy(self, **kwds: Any) -> Self:
        """Copy self and modify attributes with ``**kwds``."""
        return self.__class__(**{ **self.data, **kwds })

    def check_comparable(self, other: Any) -> None:
        """Raise :class:`TypeError` if ``self`` and ``other``
        are not comparable.
        """
        if not self.is_comparable_with(other):
            raise TypeError(
                f"'{self.cname()}' and '{self.cname(other)}'"
                " objects are not comparable"
            )

    @staticmethod
    def are_equal(obj: Any, other: Any, *, strict: bool = True) -> bool:
        """Are ``obj`` and ``other`` equal.

        Parameters
        ----------
        strict
            Should exact match on class be required.
            It also means that NLP tokens can be equal
            only when they live in the same document.
        """
        return not any(iter_diffs(obj, other, strict=strict))

    @classmethod
    def stringify(cls, obj: Any, **kwds: Any) -> str:
        """Convert ``obj`` to string.

        If ``obj`` exposes ``to_str()`` then it is used
        with keyword arguments passed in ``**kwds``.
        Otherwise the plain ``__repr__()`` is used.
        """
        return stringify(obj, **kwds)

    def equal(self, other: Any, *, strict: bool = True) -> bool:
        """Are ``self`` and ``other`` equal.

        See :meth:`iter_diffs` for details.
        """
        return not any(self.iter_diffs(other, strict=strict))

    def iter_diffs(self, other: Any, *, strict: bool = True) -> IDiffType:
        """Iterate over differences between ``self`` and ``other``.

        Parameters
        ----------
        strict
            Should exact match on class be required.
            It also means that NLP tokens can be equal
            only when they live in the same document.
        """
        yield from iter_diffs(self, other, strict=strict)


# Register diffing methods on the differ class --------------------------------

@iter_diffs.register
def _(obj: SegramABC, other: Any, *, strict: bool = True) -> IDiffType:
    yield from iter_diffs(obj.to_data(), other.to_data())


class SegramWithDocABC(SegramABC):
    """Abstract base class for :mod:`segram` classes
    with NLP document objects.
    """
    __slots__ = ()

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Any) -> bool:
        if self.is_comparable_with(other):
            return id(self.doc) == id(other.doc)
        return NotImplemented

    def __contains__(self, other: Any) -> bool:
        raise TypeError(
            f"'{self.cname(other)}' objects cannot "
            f"be contained in '{self.cname()}' instances"
        )

    # Abstract methods --------------------------------------------------------

    @property
    @abstractmethod
    def doc(self) -> "Doc":
        raise NotImplementedError

    # Properties --------------------------------------------------------------

    @property
    def lang(self) -> str:
        """Language code of the document."""
        return self.doc.lang

    # Methods -----------------------------------------------------------------

    def get_hashdata(self) -> tuple[Any, ...]:
        return (hash(self.ppath()), id(self.doc))
