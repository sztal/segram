"""Abstract base classes for generic tokens."""
# pylint: disable=import-self
from __future__ import annotations
from typing import Any, Optional, Iterable, ClassVar, Final
from functools import total_ordering
from abc import ABC, abstractmethod
from ...symbols import POS, Role
from ...utils.meta import init_class_attrs
from ...utils.diff import iter_diffs, equal, IDiffType
from ...utils.colors import color_role


class NLPTokenABC(ABC):
    """ABC for defning NLP tokens."""
    __slots__ = ()
    __attrs__: ClassVar[tuple[str, ...]] = ()

    def __hash__(self) -> int:
        return hash(self.doc)

    def __repr__(self) -> str:
        """String representation."""
        return self.text

    def __eq__(self, other: NLPTokenABC) -> bool:
        """Check equality with another token of the same type."""
        if self.is_comparable_with(other):
            return self.doc == other.doc
        return NotImplemented

    @property
    def attrs(self) -> tuple[Any, ...]:
        return tuple(getattr(self, attr) for attr in self.__attrs__)

    # Abstract methods --------------------------------------------------------

    @property
    @abstractmethod
    def text(self) -> str:
        """Raw text."""

    @property
    @abstractmethod
    def doc(self) -> DocABC:
        """Parent document object (or ``self`` for documents)."""

    @abstractmethod
    def is_comparable_with(self, other: Any) -> bool:
        """Check if ``self`` defines the same abstract interface as ``other``."""

    # Subclass hook -----------------------------------------------------------

    def __init_subclass__(cls, *args: Any, **kwds: Any) -> None:
        """Hook for customizing subclasses."""
        super().__init_subclass__(*args, **kwds)
        if "__slots__" not in cls.__dict__:
            raise TypeError(f"'{cls.__name__}' does not define '__slots__'")
        init_class_attrs(cls, {
            "__slots__": "slot_names"
        })
        try:
            total_ordering(cls)
        except ValueError:
            pass

    # Methods -----------------------------------------------------------------

    def equal(self, other: NLPTokenABC, *, strict: bool = True) -> bool:
        return self.is_comparable_with(other) and equal(self, other, strict=strict)


@total_ordering
class TokenABC(NLPTokenABC):
    """Token abstract base class."""
    __slots__ = ()
    __attrs__: Final[tuple[str, ...]] = \
        ("i", "text", "whitespace", "lemma", "pos", "role", "refs")

    def __repr__(self) -> str:
        return self.to_str(color=True)

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.i))

    def __eq__(self, other: TokenABC) -> bool:
        if (res := super().__eq__(other)) is NotImplemented:
            return res
        return res and self.i == other.i

    def __lt__(self, other: TokenABC) -> bool:
        """Is ``self`` earlier in the document than ``other``."""
        if self.is_comparable_with(other):
            return self.i < other.i
        return NotImplemented

    @property
    @abstractmethod
    def i(self) -> int:
        """Token index within the document sequence."""

    @property
    @abstractmethod
    def whitespace(self) -> str:
        """Whitespace following the token."""

    @property
    @abstractmethod
    def text_with_ws(self) -> str:
        """Token text with following whitespace."""

    @property
    @abstractmethod
    def lemma(self) -> str:
        """Lemmatized token text."""

    @property
    @abstractmethod
    def pos(self) -> POS:
        """Part-of-speech tag (UDEP)."""

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
    def role(self) -> Role:
        """Fixed token role."""

    @property
    @abstractmethod
    def ent_type(self) -> Role:
        """Named entity type."""

    @property
    @abstractmethod
    def refs(self) -> Optional[tuple[TokenABC, ...]]:
        """Coreference tokens."""

    @property
    @abstractmethod
    def sent(self) -> SpanABC:
        """Sentence object containing the token."""

    # Properties --------------------------------------------------------------

    @property
    def lang(self) -> str:
        return self.doc.lang

    @property
    def ref(self) -> TokenABC:
        """Return main coreferred token or self."""
        if (refs := self.refs):
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
        refs = self.refs
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

    def is_comparable_with(self, other: Any) -> bool:
        """Check if ``self`` defines the same abstract interface as ``other``."""
        return isinstance(other, TokenABC)


class SpanABC(NLPTokenABC):
    """Span abstract base class."""
    __slots__ = ()
    __attrs__: Final[tuple[str, ...]] = ("start", "end")

    def __hash__(self) -> int:
        return hash(super().__hash__(), self.start, self.end)

    def __eq__(self, other: SpanABC) -> bool:
        if (res := super().__eq__(other)) is NotImplemented:
            return res
        return res and (self.start, self.end) == (other.start, other.end)

    def __lt__(self, other: NLPTokenABC) -> bool:
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

    @property
    @abstractmethod
    def start(self) -> int:
        """Sentence start index."""

    @property
    @abstractmethod
    def end(self) -> int:
        "Sentence stop index."

    # Properties --------------------------------------------------------------

    @property
    def lang(self) -> str:
        return self.doc.lang

    # Methods -----------------------------------------------------------------

    def is_comparable_with(self, other: Any) -> bool:
        """Check if ``self`` defines the same abstract interface as ``other``."""
        return isinstance(other, SpanABC)


class DocABC(NLPTokenABC):
    """Document abstract base class."""
    __slots__ = ()
    __attrs__: Final[tuple[str, ...]] = ("lang",)

    # Abstract methods --------------------------------------------------------

    @abstractmethod
    def __hash__(self) -> int:
        """Document hash value."""

    @abstractmethod
    def __eq__(self, other: DocABC) -> None:
        """Is ``self`` equal to ``other``."""

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
        return hash(tuple((t.text, t.whitespace) for t in self))

    # Methods -----------------------------------------------------------------

    @abstractmethod
    def copy(self) -> DocABC:
        """Return copy of the self."""

    def is_comparable_with(self, other: Any) -> bool:
        """Check if ``self`` defines the same abstract interface as ``other``."""
        return isinstance(other, DocABC)


@equal.register
def _(obj: DocABC, other: DocABC, *, strict: bool = True) -> bool:
    return obj.is_comparable_with(other) \
        and ((strict and obj == other) or (not strict and obj.id == other.id))
@iter_diffs.register
def _(obj: DocABC, other: DocABC, *, strict: bool = True) -> IDiffType:
    if not equal(obj, other, strict=strict):
        msg = "DOCUMENT CONTENT"
        if obj.is_comparable_with(other) and obj.id == other.id:
            msg = "DOCUMENT TYPE"
        yield msg, obj, other

@equal.register
def _(obj: SpanABC, other: SpanABC, *, strict: bool = True) -> bool:
    return obj.is_comparable_with(other) \
        and (obj.start, obj.end) == (other.start, other.end) \
        and obj.doc.equal(other.doc, strict=strict)
@iter_diffs.register
def _(obj: SpanABC, other: SpanABC, *, strict: bool = True) -> IDiffType:
    if not equal(obj, other, strict=strict):
        yield "SPAN", obj, other

@equal.register
def _(obj: TokenABC, other: TokenABC, *, strict: bool = True) -> bool:
    return obj.is_comparable_with(other) \
        and (obj.i == other.i) \
        and equal(obj.doc, other.doc, strict=strict)
@iter_diffs.register
def _(obj: TokenABC, other: TokenABC, *, strict: bool = True) -> IDiffType:
    if not equal(obj, other, strict=strict):
        yield "TOKEN", obj, other
