"""Base classes from which ABCs of concret grammar classes are derived.

Grammar classes provide building blocks for representing complex
syntactical relationships within sentences which go beyond simple
syntax tree links and can be used to perform various tasks such as
component and phrase detection.
"""
from typing import Any, Self, Callable, ClassVar, Final
from typing import MutableMapping, Container, Sequence
from abc import abstractmethod
import re
from functools import total_ordering
from catalogue import Registry
import numpy as np
from ..nlp.tokens import Doc, Span, Token
from ..utils.registries import grammars
from ..abc import SegramWithDocABC
from ..datastruct import Namespace, DataTuple


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
    Doc: type["Doc"]


class Grammar(SegramWithDocABC, Container):
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

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, Grammar):
            return id(self.doc) == id(other.doc)
        return NotImplemented

    def __init_subclass__(cls, *, register: str | None = None) -> None:
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


class GrammarElement(Grammar, Sequence):
    """Abstract base class for grammar elements."""
    __slots__ = ()
    alias: ClassVar[str] = "GElem"

    def __repr__(self) -> str:
        return self.to_str(color=True)

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Self) -> bool:
        if self.is_comparable_with(other):
            return self.idx == other.idx
        return False

    def __bool__(self) -> bool:
        return self.idx is not None

    # Properties --------------------------------------------------------------

    @property
    @abstractmethod
    def idx(self) -> int | tuple[int, ...]:
        """Element index."""
        raise NotImplementedError

    @property
    def tokens(self) -> DataTuple[Token]:
        """Tokens sequence of the element."""
        return DataTuple(self)

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

    def get_hashdata(self) -> tuple[Any, ...]:
        return (*super().get_hashdata(), self.idx)

    def match(
        self,
        _pattern: str | None = None,
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


class DocElement(GrammarElement, Sequence):
    """Document element class."""
    __slots__ = ("doc",)
    alias: ClassVar[str] = "DocElem"

    def __init__(self, doc: Doc) -> None:
        super().__init__()
        self.doc = doc

    def __getitem__(self, idx: int | slice) -> Token | Span:
        return self.doc[idx]

    def __len__(self) -> int:
        return len(self.doc)

    # Properties --------------------------------------------------------------

    @property
    def idx(self) -> int:
        """Fast document id.

        It is stable for an instance, and allows for hashing,
        but is not stable for different objects with the same data,
        e.g. an element initialized from the same data twice may have
        differen ``.idx`` values each time.
        """
        return hash(self.doc)

    @property
    def id(self) -> int:
        """Slow persistent document id.

        It will be always the same for documents based
        on the same exact data.
        """
        return self.doc.id

    # Methods -----------------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_data(cls, data: dict[str, Any]) -> Self:
        """Construct data dictionary."""


@total_ordering
class SentElement(GrammarElement):
    """Grammar element based on a sentence span."""
    __slots__ = ("sent", "_doc")
    alias: ClassVar[str] = "SentElem"

    def __init__(self, sent: Span) -> None:
        super().__init__()
        if sent.root.sent is not sent:
            raise ValueError("'sent' has to be a proper sentence span object")
        self.sent = sent
        self._doc = None

    def __lt__(self, other: Self) -> bool:
        if self.is_comparable_with(other):
            return self.idx < other.idx
        return NotImplemented

    def __getitem__(self, idx: int | slice) -> Token | Span:
        return self.sent[idx]

    def __len__(self) -> int:
        return len(self.sent)

    # Properties --------------------------------------------------------------

    @property
    def doc(self) -> DocElement:
        if not self._doc:
            self._doc = self.sent.doc.grammar
        return self._doc

    @property
    def root(self) -> Token:
        return self.sent.root

    @property
    def start(self) -> int:
        return self.sent.start

    @property
    def end(self) -> int:
        return self.sent.end

    @property
    def idx(self) -> tuple[int, int]:
        """Sentence index equal to ``(self.start, self.end)``
        allowing for identification/hashing and sorting within
        the parent document.
        """
        return (self.start, self.end)

    # Methods -----------------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_data(cls, doc: Doc, data: dict[str, Any]) -> Self:
        """Construct from document and data dictionary."""


@total_ordering
class TokenElement(GrammarElement):
    """Grammar element based on a token."""
    __slots__ = ("tok", "_sent", "_doc")
    alias: ClassVar[str] = "TokElem"

    def __init__(self, tok: Token) -> None:
        super().__init__()
        self.tok = tok
        self._sent = None
        self._doc = None

    def __lt__(self, other: Self) -> bool:
        if self.is_comparable_with(other):
            return self.idx < other.idx
        return NotImplemented

    def __getitem__(self, idx: int | slice) -> Token | tuple[Token, ...]:
        return self.tokens[idx]

    def __len__(self) -> int:
        return len(self.tokens)

    # Properties --------------------------------------------------------------

    @property
    def doc(self) -> DocElement:
        if not self._doc:
            self._doc = self.tok.doc.grammar
        return self._doc

    @property
    def sent(self) -> SentElement:
        if not self._sent:
            self._sent = self.tok.sent.grammar
        return self._sent

    @property
    def idx(self) -> int:
        """Token index within the parent document."""
        return self.tok.i

    @property
    @abstractmethod
    def tokens(self) -> tuple[Token, ...]:
        pass

    # Methods -----------------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_data(cls, doc: Doc, data: dict[str, Any]) -> Self:
        """Construct from document and data dictionary."""
