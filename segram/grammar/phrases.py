from __future__ import annotations
from typing import Any, Optional, ClassVar, Iterator, Self
from collections.abc import Sequence
from itertools import islice
from abc import abstractmethod
from .abc import SentElement
from .components import Component, Verb, Noun, Desc, Prep
from .conjuncts import Conjuncts
from ..nlp import TokenABC
from ..symbols import Role, Dep
from ..abc import labelled


controlled = labelled("part")


class Phrase(SentElement):
    """Sentence phrase class.

    Attributes
    ----------
    sent
        Sentence the phrase belongs to.
    head
        Phrase head component.
    dep
        Dependency relative to the (main) parent.
    sconj
        Subordinating conjunction token.
    lead
        Lead phrase, initialized from index.
    """
    # pylint: disable=too-many-public-methods
    __slots__ = ("head", "dep", "sconj", "_lead")
    alias: ClassVar[str] = "Phrase"
    part_names: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        sent: "Sent",
        head: Component,
        *,
        dep: Dep = Dep.misc,
        sconj: Optional[TokenABC] = None,
        lead: Optional[int] = None
    ) -> None:
        super().__init__(sent)
        self.head = head
        self.dep = dep
        self.sconj = sconj
        self._lead = lead

    def __new__(cls, *args: Any, **kwds: Any) -> None:
        obj = super().__new__(cls)
        obj.__init__(*args, **kwds)
        if (cur := obj.sent.pmap.get(obj.idx)):
            cur.__init__(obj.sent, **obj.data)
            return cur
        obj.sent.pmap[obj.idx] = obj
        return obj

    def __iter__(self) -> Iterator[Phrase]:
        yield from self.tokens

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int | slice) -> Phrase | list[Phrase]:
        return self.tokens[idx]

    def __contains__(self, other: Phrase | Component | TokenABC) -> bool:
        if self.is_comparable_with(other):
            return any(other == p for p in self.iter_subtree(skip=1))
        if isinstance(other, Component):
            return any(p.head == other for p in self.iter_subtree())
        if isinstance(other, TokenABC):
            return any(other in p.head for p in self.iter_subtree())
        return super().__contains__(other)

    # Abstract methods --------------------------------------------------------

    @classmethod
    @abstractmethod
    def governs(cls, comp: Component) -> bool:
        """Check whether given phrase class may govern ``comp``."""
        if isinstance(comp, Component):
            return True
        raise TypeError(
            f"'{cls.cname()}' cannot control "
            f"'{cls.cname(comp)}' objects"
        )

    # Properties --------------------------------------------------------------

    @property
    def idx(self) -> int:
        """Index of the head component."""
        return self.head.idx

    @property
    def lead(self) -> Phrase:
        """Lead phrase."""
        return self.sent.pmap[self._lead] if self._lead is not None else self

    @property
    def is_lead(self) -> Phrase:
        """Is the phrase a lead phrase."""
        return self.lead is self

    @property
    def tokens(self) -> tuple[TokenABC, ...]:
        return tuple(t for t, _ in self.iter_token_roles())

    @property
    def neg(self) -> TokenABC | None:
        return self.head.neg

    @property
    def data(self) -> dict[str, Any]:
        return {
            **super().data,
            "lead": self._lead
        }

    @property
    def children(self) -> tuple[Phrase, ...]:
        """Child phrases."""
        return self.sent.graph[self]

    @property
    def parents(self) -> tuple[Phrase, ...]:
        """Parent phrases."""
        return self.sent.graph.rev[self]

    @property
    def subtree(self) -> tuple[Phrase]:
        """Phrasal proper subtree."""
        return tuple(self.iter_subtree(skip=1))

    @property
    def suptree(self) -> tuple[Phrase]:
        """Phrasal proper supertree."""
        return tuple(self.iter_suptree(skip=1))

    @property
    def depth(self) -> int:
        """Depth of the phrase within the phrasal tree of the sentence."""
        if (parents := self.parents):
            return min(p.depth + 1 for p in parents)
        return 0

    @property
    def conjuncts(self) -> Conjuncts:
        """Conjoined phrases."""
        if (conjs := self.sent.conjs.get(self.lead)):
            return conjs.copy(members=[
                m for m in conjs.members if m is not self
            ])
        return Conjuncts()

    @property
    def group(self) -> Conjuncts:
        """Group of self and its conjoined phrases."""
        return self.sent.conjs.get(self.lead) \
            or Conjuncts([self])

    @property
    def verb(self) -> Optional[Phrase]:
        """Return ``self`` if VP or nothing otherwise."""
        return self if isinstance(self, VerbPhrase) else None

    @property
    @controlled
    def subj(self) -> Sequence[Phrase]:
        """Subject phrases."""
        subjects = []
        for c in self.children:
            if c.dep & Dep.subj:
                subjects.append(c)
            elif c.dep & Dep.agent:
                subjects.extend(c.subj)
        return Conjuncts.get_chain(subjects)

    @property
    @controlled
    def dobj(self) -> Sequence[Phrase]:
        """Direct object phrases."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.dobj
        )
    @property
    @controlled
    def iobj(self) -> Sequence[Phrase]:
        """Indirect object phrases."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.iobj
        )
    @property
    @controlled
    def desc(self) -> Sequence[Phrase]:
        """Description phrases."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & (Dep.desc | Dep.misc)
        )
    @property
    @controlled
    def cdesc(self) -> Sequence[Phrase]:
        """Clausal descriptions."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.cdesc
        )
    @property
    @controlled
    def adesc(self) -> Sequence[Phrase]:
        """Adjectival complement descriptions."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.adesc
        )
    @property
    @controlled
    def prep(self) -> Sequence[Phrase]:
        """Prepositions."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.prep
        )
    @property
    @controlled
    def pobj(self) -> Sequence[Phrase]:
        """Prepositional objects."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.pobj
        )
    @property
    @controlled
    def subcl(self) -> Sequence[Phrase]:
        """Subclauses."""
        return Conjuncts.get_chain(
            c for c in self.children
            if (c.dep & Dep.subcl) \
            or (isinstance(c, VerbPhrase) and (c.dep & Dep.acl))
        )
    @property
    @controlled
    def relcl(self) -> Sequence[Phrase]:
        """Relative clausses."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.relcl
        )
    @property
    @controlled
    def xcomp(self) -> Sequence[Phrase]:
        """Open clausal complements."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.xcomp
        )
    @property
    @controlled
    def appos(self) -> Sequence[Phrase]:
        """Appositional modifiers."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.appos
        )
    @property
    @controlled
    def nmod(self) -> Sequence[Phrase]:
        """Nominal modifiers."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.nmod
        )

    # Methods -----------------------------------------------------------------

    def match(
        self,
        text: Optional[str] = None,
        *,
        dep: Optional[Dep | str] = None,
        alias: Optional[str] = None,
        **kwds: Any
    ) -> bool:
        """Match element text against a regex pattern
        using :func:`re.search` function.

        Parameters
        ----------
        text
           String used for matching.
           No matching is done when ``None``.
        dep
            Dependency tag to match (partial overlap is considered a match).
            Can be provided as a string,
            including alternatives such ``"xcomp|relcl"``.
        alias
            Phrase type alias to match.
        **kwds
            Passed to :meth:`segram.grammar.abc.GrammarElement.match`.
        """
        matched = super().match(text, **kwds)
        if dep is not None:
            if isinstance(dep, str):
                dep = Dep.from_name(dep)
            matched &= bool(dep & self.dep)
        if alias is not None:
            matched &= self.alias == alias
        return matched

    def iter_subtree(self, *, skip: int = 0) -> Iterator[Phrase]:
        """Iterate over phrasal subtree and omit ``skip`` first items."""
        def _iter():
            yield self
            for child in self.children:
                yield from child.iter_subtree(skip=0)
        yield from islice(_iter(), skip, None)

    def iter_suptree(self, *, skip: int = 0) -> Iterator[Phrase]:
        """Iterate over phrasal supertree and omit ``skip`` first items."""
        def _iter():
            yield self
            for parent in self.parents:
                yield from parent.iter_suptree(skip=0)
        yield from islice(_iter(), skip, None)

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, Phrase)

    def to_str(self, *, color: bool = False, only_head: bool = False, **kwds: Any) -> str:
        """Represent as a string."""
        # pylint: disable=unused-argument
        if only_head:
            return self.head.to_str(color=color)
        return " ".join(
            t.to_str(color=color, role=r)
            for t, r in self.iter_token_roles()
        )

    def iter_token_roles(
        self,
        *,
        bg: bool = False
    ) -> Iterator[TokenABC, Role | None]:
        """Iterate over token-role pairs.

        Parameters
        ----------
        bg
            Should tokens be marked as a background token
            (e.g. as a part of a subclause).
            This is used for graying out subclauses when printing.
        """
        def _iter():
            role = self.dep.role if self.dep else None
            yield from self.head.iter_token_roles(role=role)
            if (sconj := self.sconj):
                yield sconj, sconj.role
            for child in self.children:
                if child.is_lead and (conjs := child.conjuncts):
                    if (pconj := conjs.preconj):
                        yield pconj, None
                    if (cconj := conjs.cconj):
                        yield cconj, None
                is_vp = isinstance(child, VerbPhrase)
                yield from child.iter_token_roles(bg=is_vp)
        toks = sorted(set(_iter()), key=lambda x: x[0])
        if bg:
            for tok, _ in toks:
                yield tok, Role.BG
        else:
            yield from toks

    @classmethod
    def from_component(cls, comp: Component, **kwds: Any) -> Self:
        """Construct from a grammar component."""
        for typ in cls.types.values():
            if not issubclass(typ, Phrase) \
            or getattr(typ, "__abstractmethods__", None):
                continue
            if typ.governs(comp):
                return typ(comp.sent, comp, **kwds)
        raise ValueError(f"no matching phrase type for '{cls.cname(comp)}'")

    def to_data(self) -> dict[str, Any]:
        """Serialize to a data dictionary."""
        return {
            "@class": self.alias,
            "head": self.head.idx,
            "dep": self.dep.name,
            "sconj": self.sconj.i if self.sconj else None,
            "lead": self._lead
        }

    @classmethod
    def from_data(cls, sent: "Sent", data: dict[str, Any]) -> Self:
        """Construct from sentence and data dictionary."""
        data = data.copy()
        doc = sent.doc
        typ = cls.types[data.pop("@class")]
        kwds = dict(
            head=sent.cmap[data["head"]],
            dep=Dep.from_name(data["dep"]),
            sconj=doc[i] if (i := data["sconj"]) is not None else None,
            lead=data["lead"]
        )
        return typ(sent, **kwds)


class VerbPhrase(Phrase):
    """Abstract base class for verb phrases."""
    __slots__ = ()
    alias: ClassVar[str] = "VP"

    @classmethod
    def governs(cls, comp: Component) -> bool:
        return super().governs(comp) \
            and isinstance(comp, Verb)

class NounPhrase(Phrase):

    """Abstract base class for noun phrases."""
    __slots__ = ()
    alias: ClassVar[str] = "NP"

    @classmethod
    def governs(cls, comp: Component) -> bool:
        return super().governs(comp) \
            and isinstance(comp, Noun)


class DescPhrase(Phrase):
    """Abstract base class for descriptive phrases."""
    __slots__ = ()
    alias: ClassVar[str] = "DP"

    @classmethod
    def governs(cls, comp: Component) -> bool:
        return super().governs(comp) \
            and isinstance(comp, Desc)


class PrepPhrase(Phrase):
    """Abstract base class for prepositional phrases."""
    __slots__ = ()
    alias: ClassVar[str] = "PP"

    @classmethod
    def governs(cls, comp: Component) -> bool:
        return super().governs(comp) \
            and isinstance(comp, Prep)
