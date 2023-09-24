"""Base classes from which ABCs of concret grammar classes are derived.

Grammar classes provide building blocks for representing complex
syntactical relationships within sentences which go beyond simple
syntax tree links and can be used to perform various tasks such as
component and phrase detection.
"""
from __future__ import annotations
from typing import Any, Optional, Iterable, MutableMapping, Self, Callable
from typing import ClassVar, Final
from abc import abstractmethod
import re
from functools import total_ordering
from catalogue import Registry
import numpy as np
from ..nlp.tokens import Doc, Span, Token
from ..utils.registries import grammars
from ..abc import SegramWithDocABC
from ..datastruct import Namespace


class GrammarNamespace(Namespace):
    Grammar: type["Grammar"]
    Component: type["Component"]
    Verb: type["Verb"]
    Noun: type["Noun"]
    Prep: type["Prep"]
    Desc: type["Desc"]
    Phrase: type["Phrase"]
    VP: type["VerbPhrase"]
    NP: type["NounPhrase"]
    DP: type["DescPhrase"]
    PP: type["PrepPhrase"]
    Sent: type["Sent"]


class Grammar(SegramWithDocABC):
    """Abstract base class for grammar classes.

    All grammar classes must be defined as **slots** classes.
    This is necessary for ensuring low-memory footprint
    and better computational efficiency. Even classes with no
    new slots need to declare ``__slots__ = ()``.
    This requirement is checked during class construction.
    Other class-specific requirements of this sort as well as
    their related validation checks may be implemented on specialized grammar
    classes using the standard ``__init_subclass__`` interface.
    This allows abstract base classes further down the inheritance chain
    to check for more complex requirements as well as apply dynamic class
    customizations.
    """
    __slots__ = ()
    alias: ClassVar[str] = "Grammar"
    types: ClassVar[GrammarNamespace] = GrammarNamespace()
    roles: ClassVar[MutableMapping] = Namespace()
    grammars: Final[Registry] = grammars

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Grammar) -> bool:
        if isinstance(other, Grammar):
            return id(self.doc) == id(other.doc)
        return NotImplemented

    def __contains__(self, other: Any) -> bool:
        raise TypeError(
            f"'{self.cname(other)}' objects cannot "
            f"be contained in '{self.cname()}' instances"
        )

    def __init_subclass__(cls, *, register: Optional[str] = None) -> None:
        super().__init_subclass__()
        if register:
            cls.types = GrammarNamespace()
            cls.roles = Namespace()
            cls.grammars.register(register, func=cls)

        # Add to the members list ---------------------------------------------
        if (alias := getattr(cls, "alias", None)):
            if (t := cls.types.get(alias)) \
            and (tpath := t.ppath()) != cls.ppath():
                raise TypeError(f"'{alias}' already defined by '{tpath}'")
            cls.types[alias] = cls

    # Methods -----------------------------------------------------------------

    @abstractmethod
    def to_data(self) -> dict[str, Any]:
        """Dump to data dictionary."""
        raise NotImplementedError


@total_ordering
class GrammarElement(Grammar):
    """Grammar element is any group of one or more tokens
    linked together by well-defined syntactic relationships.

    It is used to derive component classes.
    """
    __slots__ = ()
    alias: ClassVar[str] = "GElem"

    def __repr__(self) -> str:
        return self.to_str(color=True)

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: GrammarElement) -> bool:
        if self.is_comparable_with(other):
            return self.idx == other.idx
        return NotImplemented

    def __lt__(self, other: GrammarElement) -> bool:
        if self.is_comparable_with(other):
            return self.idx < other.idx
        return NotImplemented

    def __bool__(self) -> bool:
        return self.idx is not None

    # Abstract methods --------------------------------------------------------

    @abstractmethod
    def __iter__(self) -> Iterable[GrammarElement]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, i: int) -> GrammarElement:
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, other: Token | GrammarElement) -> bool:
        """Check whether ``self`` contains ``other``."""
        raise NotImplementedError

    # Properties --------------------------------------------------------------

    @property
    @abstractmethod
    def idx(self) -> int | tuple[int, ...]:
        """Element index."""
        raise NotImplementedError

    @property
    @abstractmethod
    def tokens(self) -> tuple[Token, ...]:
        """Tokens of the element."""

    @property
    @abstractmethod
    def sent(self) -> Span:
        """NLP sentence containing the element."""
        raise NotImplementedError

    @property
    def hashdata(self) -> tuple[Any, ...]:
        """Data tuple used for hashing."""
        return (*super().hashdata, self.idx)

    @property
    def text(self) -> str:
        """Raw text of element."""
        return self.to_str()

    @property
    def lemma(self) -> str:
        return "".join(t.lemma+t.whitespace for t in self.tokens).strip()

    @property
    def vector(self) -> np.ndarray[tuple[int], np.floating]:
        """Average token word vector."""
        toks = self.tokens
        return sum(tok.vector for tok in toks) / len(toks)

    # Methods -----------------------------------------------------------------

    @abstractmethod
    def to_str(self, **kwds: Any) -> str:
        """Represent as a string."""
        raise NotImplementedError

    def match(
        self,
        _pattern: Optional[str] = None,
        _flag: re.RegexFlag = re.NOFLAG,
        _ignore_missing: bool = False,
        **kwds: Any | Callable[[Any], bool]
    ) -> re.Pattern | None:
        """Match element text against a regex pattern
        using :func:`re.search` function.

        Parameters
        ----------
        _pattern
            Regular expression pattern used for matching.
            No matching is done when ``None``.
        _flag
            Regex flag.
        _ignore_missing_fields
            Should missing fields on ``self`` be ignored.
        **kwds
            Other keyword arguments can be used for testing
            values of different attributes on ``self``.
            If callables are passed as values then they are
            expected to be predicate functions returning
            boolean values.
        """
        is_match = True
        if _pattern is not None:
            is_match &= bool(re.search(_pattern, self.text, _flag))
        for field, test in kwds.items():
            try:
                attr = getattr(self, field)
            except AttributeError as exc:
                if _ignore_missing:
                    continue
                raise exc
            if isinstance(test, Callable):
                is_match &= bool(test(attr))
            else:
                is_match &= attr == test
        return is_match


class DocElement(GrammarElement):
    """Document element class."""
    __slots__ = ("_doc",)
    alias: ClassVar[str] = "DocElem"

    def __init__(self, doc: Doc) -> None:
        self._doc = doc

    # Abstract methods --------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_data(cls, doc: Doc, data: dict[str, Any]) -> Self:
        """Construct from sentence and a data dictionary."""
        # pylint: disable=arguments-renamed
        raise NotImplementedError

    # Properties --------------------------------------------------------------

    @property
    def doc(self) -> Doc:
        return self._doc

    # Methods -----------------------------------------------------------------

    def copy(self, **kwds: Any) -> Self:
        return self.__class__(**{ "doc": self.doc, **self.data, **kwds })


class SentElement(GrammarElement):
    """Sentence element class."""
    __slots__ = ("_sent",)
    alias: ClassVar[str] = "SentElem"

    def __init__(self, sent: "Sent") -> None:
        self._sent = sent

    # Abstract methods --------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_data(cls, sent: "Sent", data: dict[str, Any]) -> Grammar:
        """Construct from sentence and a data dictionary."""
        raise NotImplementedError

    # Properties --------------------------------------------------------------

    @property
    def sent(self) -> "Sent":
        return self._sent

    @property
    def doc(self) -> Doc:
        return self.sent.doc

    @property
    def hashdata(self) -> tuple[Any, ...]:
        return (self.ppath(), id(self.doc), id(self.sent), self.idx)

    # Methods -----------------------------------------------------------------

    def copy(self, **kwds: Any) -> SentElement:
        return self.__class__(**{ "sent": self.sent, **self.data, **kwds })
