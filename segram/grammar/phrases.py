from __future__ import annotations
from typing import Any, Optional, ClassVar, Iterator
from collections.abc import Sequence
from abc import abstractmethod
from .abc import SentElement
from .components import Component, Verb, Noun, Desc, Prep
from .conjuncts import Conjuncts
from ..nlp import TokenABC
from ..symbols import Role, Dep


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
        Lead phrase, initialized from index.i
    """
    # pylint: disable=too-many-public-methods
    __slots__ = ("head", "dep", "sconj", "_lead")
    alias: ClassVar[str] = "Phrase"

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
        yield from self.subtree

    def __len__(self) -> int:
        return sum(1 for _ in self.subtree)

    def __getitem__(self, idx: int | slice) -> Phrase | list[Phrase]:
        end = idx.stop if isinstance(idx, slice) else idx+1
        sub = []
        for i, p in enumerate(self.subtree):
            if not end or i < end:
                sub.append(p)
        return sub[idx]

    def __contains__(self, other: Phrase | Component | TokenABC) -> bool:
        if self.is_comparable_with(other):
            return any(other == p for p in self.subtree)
        if isinstance(other, Component):
            return any(p.head == other for p in self.subtree)
        if isinstance(other, TokenABC):
            return any(other in p.head for p in self.subtree)
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
        return self.head.idx

    @property
    def lead(self) -> Phrase:
        return self.sent.pmap[self._lead] if self._lead is not None else self

    @property
    def is_lead(self) -> Phrase:
        return self.lead is self

    @property
    def tokens(self) -> tuple[TokenABC, ...]:
        return tuple(t for t, _ in self.iter_token_roles())

    @property
    def data(self) -> dict[str, Any]:
        return {
            **super().data,
            "lead": self._lead
        }

    @property
    def children(self) -> tuple[Phrase, ...]:
        return self.sent.graph[self]

    @property
    def parents(self) -> tuple[Phrase, ...]:
        return self.sent.graph.rev[self]

    @property
    def subtree(self) -> Iterator[Phrase]:
        yield self
        for child in self.children:
            yield from child.subtree

    @property
    def suptree(self) -> Iterator[Phrase]:
        yield self
        for parent in self.parents:
            yield from parent.suptree

    @property
    def depth(self) -> int:
        if (parents := self.parents):
            return min(p.depth + 1 for p in parents)
        return 0

    @property
    def conjuncts(self) -> Conjuncts:
        if (conjs := self.sent.conjs.get(self.lead)):
            return conjs.copy(members=[
                m for m in conjs.members if m is not self
            ])
        return Conjuncts()

    @property
    def subj(self) -> Sequence[Phrase]:
        subjects = []
        for c in self.children:
            if c.dep & Dep.subj:
                subjects.append(c)
            elif c.dep & Dep.agent:
                subjects.extend(c.subj)
        return Conjuncts.get_chain(subjects)
    @property
    def dobj(self) -> Sequence[Phrase]:
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.dobj
        )
    @property
    def iobj(self) -> Sequence[Phrase]:
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.iobj
        )
    @property
    def desc(self) -> Sequence[Phrase]:
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.desc
        )
    @property
    def cdesc(self) -> Sequence[Phrase]:
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.cdesc
        )
    @property
    def adesc(self) -> Sequence[Phrase]:
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.adesc
        )
    @property
    def prep(self) -> Sequence[Phrase]:
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.prep
        )
    @property
    def pobj(self) -> Sequence[Phrase]:
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.pobj
        )
    @property
    def subcl(self) -> Sequence[Phrase]:
        return Conjuncts.get_chain(
            c for c in self.children
            if (c.dep & Dep.subcl) \
            or (isinstance(c, VerbPhrase) and (c.dep & Dep.acl))
        )
    @property
    def relcl(self) -> Sequence[Phrase]:
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.relcl
        )
    @property
    def xcomp(self) -> Sequence[Phrase]:
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.xcomp
        )
    @property
    def appos(self) -> Sequence[Phrase]:
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.appos
        )
    @property
    def nmod(self) -> Sequence[Phrase]:
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.nmod
        )

    # Methods -----------------------------------------------------------------

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

    def iter_token_roles(self) -> Iterator[TokenABC, Role | None]:
        """Iterate over token-role pairs."""
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
                yield from child.iter_token_roles()
        yield from sorted(set(_iter()), key=lambda x: x[0])

    @classmethod
    def from_component(cls, comp: Component, **kwds: Any) -> Phrase:
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
    def from_data(cls, sent: "Sent", data: dict[str, Any]) -> Phrase:
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
