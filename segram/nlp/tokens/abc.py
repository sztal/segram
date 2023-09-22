"""Abstract base classes for generic tokens."""
# pylint: disable=import-self
from __future__ import annotations
from typing import Any, Optional, Iterable, Final, Callable
from functools import total_ordering
from abc import ABC, abstractmethod
from ...symbols import POS, Role
from ...utils.meta import init_class_attrs
from ...utils.diff import iter_diffs, equal, IDiffType
from ...utils.colors import color_role


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
    def doc(self) -> DocABC:
        """Parent document object (or ``self`` for documents)."""

    # Methods -----------------------------------------------------------------

    def equal(self, other: NLPToken, *, strict: bool = True) -> bool:
        return equal(self, other, strict=strict)

    def is_comparable_with(self, other: Any) -> bool:
        """Check if ``self`` defines the same abstract interface as ``other``."""
        if not isinstance(other, NLPToken):
            return NotImplemented
        if self.doc is not other.doc:
            raise ValueError("'self' and 'other' are based on different documents")
        return isinstance(other, self.__class__) or NotImplemented


@total_ordering
class TokenABC(NLPToken):
    """Token abstract base class."""
    __slots__ = ()

    def __repr__(self) -> str:
        return self.to_str(color=True)

    def __hash__(self) -> int:
        return hash((hash(self.doc), self.i))

    def __eq__(self, other: TokenABC) -> bool:
        if (res := super().__eq__(other)) is NotImplemented:
            return res
        return res and self.i == other.i

    def __lt__(self, other: TokenABC) -> bool:
        """Is ``self`` earlier in the document than ``other``."""
        if self.is_comparable_with(other):
            return self.i < other.i
        return NotImplemented

    # Attr-properties ---------------------------------------------------------

    @attr
    @property
    @abstractmethod
    def i(self) -> int:
        """Token index within the document sequence."""

    @attr
    @property
    @abstractmethod
    def whitespace(self) -> str:
        """Whitespace following the token."""

    @attr
    @property
    @abstractmethod
    def lemma(self) -> str:
        """Lemmatized token text."""

    @attr
    @property
    @abstractmethod
    def pos(self) -> POS:
        """Part-of-speech tag (UDEP)."""

    @attr
    @property
    @abstractmethod
    def role(self) -> Role:
        """Fixed token role."""

    @attr
    @property
    @abstractmethod
    def ent(self) -> Role:
        """Named entity type."""

    @attr
    @property
    @abstractmethod
    def corefs(self) -> Optional[tuple[TokenABC, ...]]:
        """Coreference tokens."""

    # Properties --------------------------------------------------------------

    @property
    @abstractmethod
    def text_with_ws(self) -> str:
        """Token text with following whitespace."""

    @property
    @abstractmethod
    def is_negation(self) -> bool:
        """Is negation token."""

    @property
    @abstractmethod
    def is_qmark(self) -> bool:
        """Is question mark token."""

    @property
    @abstractmethod
    def is_exclam(self) -> bool:
        """Is exclamation mark token."""

    @property
    @abstractmethod
    def is_intj(self) -> bool:
        """Is interjection token."""

    @property
    @abstractmethod
    def sent(self) -> SpanABC:
        """Sentence object containing the token."""

    # Properties --------------------------------------------------------------

    @property
    def lang(self) -> str:
        return self.doc.lang

    @property
    def coref(self) -> TokenABC:
        """Return main coreferred token or self."""
        if (refs := self.corefs):
            return refs[0]
        return self

    # Methods -----------------------------------------------------------------

    def to_str(
        self,
        *,
        color: bool = False,
        **kwds: Any
    ) -> str:
        """Represent as a string.

        Parameters
        ----------
        color
            Use colors.
        **kwds
            Passed to :func:`~segram.utils.colors.color_role`.
            They can be used to override the fixed token role
            with contextual roles using ``role`` keyword argument.
        """
        refs = self.corefs
        if refs:
            refs = ",".join(r.to_str(color=False) for r in refs)
            refs = f"[{refs}]"
            rrole = kwds.get("role")
            if rrole is Role.BG:
                refs = color_role(refs, **{ **kwds, "role": rrole })
        else:
            refs = ""
        kwds = { "role": self.role, **kwds }
        return f"{color_role(self.text, color=color, **kwds)}{refs}"


class SpanABC(NLPToken):
    """Span abstract base class."""
    __slots__ = ()

    def __hash__(self) -> int:
        return hash((hash(self.doc), self.start, self.end))

    def __eq__(self, other: SpanABC) -> bool:
        if (res := super().__eq__(other)) is NotImplemented:
            return res
        return res and (self.start, self.end) == (other.start, other.end)

    def __lt__(self, other: NLPToken) -> bool:
        """Is ``self`` earlier in the document than ``other``."""
        if self.is_comparable_with(other):
            return (self.start, self.end) < (other.start, other.end)
        return NotImplemented

    @abstractmethod
    def __iter__(self) -> Iterable[TokenABC]:
        """Iterate over tokens."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of tokens."""

    @abstractmethod
    def __getitem__(self, idx: int | slice) -> TokenABC:
        """Return specific token or (sub)span by indexing."""

    @abstractmethod
    def __contains__(self, tok: TokenABC) -> bool:
        """Check if ``self`` contains ``tok``."""

    @attr
    @property
    @abstractmethod
    def start(self) -> int:
        """Sentence start index."""

    @attr
    @property
    @abstractmethod
    def end(self) -> int:
        "Sentence stop index."

    # Properties --------------------------------------------------------------

    @property
    def lang(self) -> str:
        return self.doc.lang


class DocABC(NLPToken):
    """Document abstract base class."""
    __slots__ = ()

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: Any) -> bool:
        return self is other

    # Abstract methods --------------------------------------------------------

    @abstractmethod
    def __iter__(self) -> Iterable[TokenABC]:
        """Iterate over tokens."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of tokens."""

    @abstractmethod
    def __getitem__(self, idx: int | slice) -> TokenABC:
        """Return specific token or span by indexing."""

    @abstractmethod
    def __contains__(self, tok: TokenABC) -> bool:
        """Check if ``self`` contains ``tok``."""

    @attr
    @property
    @abstractmethod
    def lang(self) -> str:
        """Language code."""

    @property
    @abstractmethod
    def sents(self) -> Iterable[SpanABC]:
        """Iterate over sentences."""

    # Properties --------------------------------------------------------------

    @property
    def id(self) -> int:
        """Hash id of the document tokenization."""
        return hash(t.attrs for t in self)

    # Methods -----------------------------------------------------------------

    @abstractmethod
    def copy(self) -> DocABC:
        """Return copy of the self."""


@equal.register
def _(obj: DocABC, other: DocABC, *, strict: bool = True) -> bool:
    return ((strict and obj == other) or (not strict and obj.id == other.id))
@iter_diffs.register
def _(obj: DocABC, other: DocABC, *, strict: bool = True) -> IDiffType:
    if not equal(obj, other, strict=strict):
        msg = "DOCUMENT CONTENT"
        if obj.id == other.id:
            msg = "DOCUMENT TYPE"
        yield msg, obj, other

@equal.register
def _(obj: SpanABC, other: SpanABC, *, strict: bool = True) -> bool:
    return obj.doc.equal(other.doc, strict=strict) \
        and (obj.start, obj.end) == (other.start, other.end)
@iter_diffs.register
def _(obj: SpanABC, other: SpanABC, *, strict: bool = True) -> IDiffType:
    if not equal(obj, other, strict=strict):
        yield "SPAN", obj, other

@equal.register
def _(obj: TokenABC, other: TokenABC, *, strict: bool = True) -> bool:
    return equal(obj.doc, other.doc, strict=strict) \
        and (obj.i == other.i)
@iter_diffs.register
def _(obj: TokenABC, other: TokenABC, *, strict: bool = True) -> IDiffType:
    if not equal(obj, other, strict=strict):
        yield "TOKEN", obj, other
