"""Abstract base class for grammar components.

Grammar components are groups of associated tokens controlled
by a root token, e.g. a verb with its auxiliary verbs.
"""
from __future__ import annotations
from typing import Any, Iterable, Iterator, ClassVar, Self
from .abc import TokenElement
from .conjuncts import Conjuncts
from ..nlp.tokens import Token
from ..symbols import POS, Role, Tense, Modal, Mood, Symbol
from ..utils.misc import cosine_similarity
from ..datastruct import DataSequence
from ..nlp.tokens import Doc


class Component(TokenElement):
    """Abstract base class for grammar components.

    Components consists of a root token associated with (optional)
    additional subordinated tokens, e.g. a noun and its determiner.

    Default syntactic role assigned to the given component type
    can be defined using ``__role__`` class attribute.

    This is a base class used for implementing concrete
    components classes. Names of all possible controlled tokens
    (e.g. ``neg`` for a negation token) must be defined in
    ``__tokens__`` class attributes along the iheritance chain
    of concrete subclasses up to :class:`GrammarComponent``.
    Each controlled token name should be declared only once and
    all must be present also in ``__slots__``. Component classes
    not defining any new controlled token slots have to define
    ``__tokens__ = ()``. The same rules apply to defining component
    attributes through ``__attrs__`` class attributes.

    The above requirements are checked at runtime during class creation.

    Attributes
    ----------
    sent
        Sentence the component belongs to.
    tok
        Head token object.
    role
        Syntactic role of the component head token.
    sub
        Tokens dependent on the head
        and not included in other token categories.
        They are not printed.
    qmark
        Question mark token.
    exclam
        Exclamation mark token.
    intj
        Interjection token.
    neg
        Negation token(s).
    """
    __role__ = None
    __tokens__ = ("qmark", "exclam", "intj", "neg")
    __slots__ = ("_tid", "role", "sub", *__tokens__)
    alias: ClassVar[str] = "Component"
    token_names: ClassVar[tuple[str, ...]] = ()
    attr_names: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        tok: Token,
        *,
        role: Role | None = None,
        sub: Iterable[Token] = (),
        qmark: Token | None = None,
        exclam: Token | None = None,
        intj: Token | None = None,
        neg: Token | None = None
    ) -> None:
        super().__init__(tok)
        self._tid = None
        role = role or self.__role__
        self.role = Role.from_name(role) if isinstance(role, str) else role
        self.qmark = qmark
        self.exclam = exclam
        self.intj = intj
        self.neg = neg
        self.sub = DataSequence(sub)

    def __new__(cls, *args: Any, **kwds: Any) -> None:
        obj = super().__new__(cls)
        obj.__init__(*args, **kwds)
        if (cur := obj.sent.cmap.get(obj.idx)):
            cur_kws = { k: v for k, v in obj.data.items() if k in cur.slot_names }
            cur.__init__(obj.sent, **cur_kws)
            return cur
        obj.sent.cmap[obj.idx] = obj
        obj.sent.pmap[obj.idx] = obj.phrase
        return obj

    def __iter__(self) -> Iterator[Token]:
        yield from self.tokens

    def __len__(self) -> int:
        return len(self.tid)

    def __getitem__(self, idx: int) -> Token | tuple[Token, ...]:
        return self.tokens[idx]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.init_class_attrs({
            "__tokens__": "token_names",
            "__attrs__": "attr_names"
        })
        if "tok" in cls.__tokens__:
            raise TypeError("'tok' cannot be declared in '__tokens__'")
        tags = getattr(cls, "__tags__", None)
        # Build role map ------------------------------------------------------
        role = getattr(cls, "__role__", None)
        if role:
            if (t := cls.roles.get(role)) \
            and (tpath := t.ppath()) != cls.ppath():
                raise TypeError(f"'{role}' role already assigned to '{tpath}'")
            cls.roles[role] = cls
        if tags:
            tags = [ POS.from_name(tag) for tag in tags.name.split("|") ]
            for pos in tags:
                cur_type = cls.roles.get(pos)
                # pylint: disable=comparison-with-callable
                if cur_type and cur_type.ppath() != cls.ppath():
                    raise TypeError(
                        "cannot map multiple component "
                        f"classes to '{pos.name}' POS tag"
                    )
                cls.roles[pos] = cls

    # Properties --------------------------------------------------------------

    @property
    def idx(self) -> int:
        """Index of the component head token."""
        return self.tok.i

    @property
    def head(self) -> Token:
        """Component head token."""
        return self.tok

    @property
    def lead(self) -> Component:
        """Head component of the lead phrase."""
        return self.phrase.lead.head

    @property
    def is_lead(self) -> bool:
        """Is the controlling phrase of the component a lead phrase."""
        return self.phrase.is_lead

    @property
    def conjuncts(self) -> Conjuncts[Component]:
        return (conjs := self.phrase.conjuncts).copy(
            members=tuple(m.head for m in conjs.members)
        )

    @property
    def phrase(self) -> "Phrase":
        if (p := self.sent.pmap.get(self.idx)):
            return p
        return self.types.Phrase.from_component(self)

    @property
    def tid(self) -> tuple[int, ...]:
        if self._tid is None:
            self._tid = self.get_tid()
        return self._tid

    @property
    def tokens(self) -> tuple[Token, ...]:
        return tuple(self.doc[i] for i in self.tid)

    @property
    def subtokens(self) -> tuple[Token, ...]:
        return sorted((*self.tokens, *self.sub))

    @property
    def pos(self) -> POS:
        return self.tok.pos

    @property
    def attrs(self) -> dict[str, Any]:
        """Attributes dictionary."""
        dct = {}
        for name in self.attr_names:
            attr = getattr(self, name)
            if isinstance(attr, Symbol):
                attr = attr.name
            dct[name] = attr
        return dct

    # Methods -----------------------------------------------------------------

    @classmethod
    def from_data(cls, doc: Doc, data: dict[str, Any]) -> Self:
        """Construct from :class:`~segram.nlp.Doc` and a data dict."""
        data = data.copy()
        alias = data.pop("@class")
        typ = cls.types[alias]
        for name in ("tok", *typ.token_names, "sub"):
            if name not in data:
                continue
            idx = data[name]
            if isinstance(idx, int):
                data[name] =  doc[idx]
            else:
                data[name] = [ doc[i] for i in idx ]
        return typ(**data)

    def to_data(self) -> dict[str, Any]:
        """Dump to data dictionary."""
        slots = ("tok", "sub", *self.token_names)
        data = {}
        for name, tok in self.data.items():
            if name not in slots or not tok:
                continue
            if isinstance(tok, Token):
                data[name] = tok.i
            else:
                data[name] = [ t.i for t in tok ]
        return {
            "@class": self.alias,
            **data,
            **self.attrs,
        }

    @classmethod
    def get_comp_type(
        cls,
        role: Role = None,
        pos: POS | None = None
    ) -> type[Component]:
        """Get component type from role or POS tag."""
        return cls.roles.get(role, cls.roles.get(pos, cls))

    def to_str(self, *, color: bool = False, role: Role | None = None, **kwds: Any) -> str:
        """Represent as a string.

        Parameters
        ----------
        role
            Overrides head token role.
        """
        # pylint: disable=unused-argument
        return " ".join(
            t.to_str(color=color, role=r)
            for t, r in self.iter_token_roles(role=role)
        )

    def get_tid(self) -> tuple[int, ...]:
        """Get token tuple id."""
        def _iter():
            for name in ("tok", *self.token_names):
                if (value := getattr(self, name, None)):
                    if isinstance(value, Token):
                        yield value
                    else:
                        yield from value
        return tuple(sorted(t.i for t in _iter()))

    def iter_token_roles(
        self,
        *,
        role: Role | None = None,
        bg: bool = False
    ) -> Iterable[tuple[Token, Role | None]]:
        """Iterate over token-role pairs.

        Parameters
        ----------
        role
            Overrides head token role.
        bg
            Should tokens be marked as a background token
            (e.g. as a part of a subclause).
            This is used for graying out subclauses when printing.
        """
        role = role or self.role
        if bg:
            role = Role.BG
        for tok in self.tokens:
            yield tok, role if tok == self.tok else tok.role

    def is_comparable_with(self, other: Any) -> None:
        return isinstance(other, Component)

    def similarity(self, other: Component | Token) -> float:
        """Cosine similarity to other component."""
        return cosine_similarity(self.vector, other.vector)


# pylint: disable=abstract-method
class Verb(Component):
    """Abstract base class for verb components.

    Attributes
    ----------
    neg
        Negation token.

    Notes
    -----
    It defines also ``tense`` attribute.
    """
    __role__ =  Role.VERB
    __tags__ = POS.VERB | POS.AUX
    __attrs__ = ("tense", "modal", "mood")
    __slots__ = (*__attrs__,)
    alias: ClassVar[str] = "Verb"

    def __init__(
        self,
        *args: Any,
        tense: Tense = Tense.PRESENT,
        modal: Modal = Modal.NULL,
        mood: Mood = Mood.REAL,
        **kwds: Any
    ) -> None:
        super().__init__(*args, **kwds)
        self.tense = Tense.from_name(tense) if isinstance(tense, str) else tense
        self.modal = Modal.from_name(modal) if isinstance(modal, str) else modal
        self.mood = Mood.from_name(mood) if isinstance(mood, str) else mood


class Noun(Component):
    """Abstract base class for noun components.

    Attributes
    ----------
    mod
        Modifier tokens.
    """
    __role__ = Role.NOUN
    __tags__ = POS.NOUN | POS.PROPN | POS.PRON
    __slots__ = ()
    alias: ClassVar[str] = "Noun"


class Prep(Component):
    """Abstract base class for preposition components.

    Attributes
    ----------
    preps
        Chain of subsequent prepositions attached to the head token.
    """
    __role__ = Role.PREP
    __tags__ = POS.ADP
    __tokens__ = ("preps",)
    __slots__ = (*__tokens__,)
    alias: ClassVar[str] = "Prep"

    def __init__(
        self,
        *args: Any,
        preps: Iterable[Token] = (),
        **kwds: Any
    ) -> None:
        super().__init__(*args, **kwds)
        self.preps = tuple(preps)


class Desc(Component):
    """Abstract base class for description components.

    Attributes
    ----------
    mod
        Modifier tokens.
    """
    __role__ = Role.DESC
    __tags__ = POS.ADJ | POS.ADV
    __tokens__ = ("mod",)
    __slots__ = (*__tokens__,)
    alias: ClassVar[str] = "Desc"

    def __init__(
        self,
        *args: Any,
        mod: Iterable[Token] = (),
        **kwds: Any
    ) -> None:
        super().__init__(*args, **kwds)
        self.mod = tuple(mod)
