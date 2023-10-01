# pylint: disable=no-name-in-module
from typing import Any, ClassVar, Self, Iterable
from abc import abstractmethod
from itertools import islice
from more_itertools import unique_everseen
import numpy as np
from .abc import TokenElement
from .components import Component, Verb, Noun, Desc, Prep
from .conjuncts import PhraseGroup, Conjuncts
from ..nlp.tokens import Doc, Token
from ..symbols import Role, Dep
from ..abc import labelled
from ..datastruct import DataIterator, DataTuple


controlled = labelled("controlled")
component = labelled("component")
PGType = PhraseGroup["Phrase"]


class Phrase(TokenElement):
    """Sentence phrase class.

    Attributes
    ----------
    dep
        Dependency relative to the (main) parent.
    sconj
        Subordinating conjunction token.
    """
    # pylint: disable=too-many-public-methods
    __slots__ = ("dep", "sconj", "_lead")
    alias: ClassVar[str] = "Phrase"
    controlled_names: ClassVar[tuple[str, ...]] = ()
    component_names: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        tok: Token,
        *,
        dep: Dep = Dep.misc,
        sconj: Token | None = None,
        lead: int | None = None
    ) -> None:
        """Initialization method.

        Parameters
        ----------
        lead
            Index of the lead phrase in a conjunct group.
        """
        super().__init__(tok)
        self.dep = dep
        self.sconj = sconj
        self._lead = lead

    def __new__(cls, *args: Any, **kwds: Any) -> None:
        obj = super().__new__(cls)
        obj.__init__(*args, **kwds)
        if (cur := obj.sent.pmap.get(obj.idx)):
            cur.__init__(**obj.data)
            return cur
        obj.sent.pmap[obj.idx] = obj
        return obj

    def __iter__(self) -> Iterable[Token]:
        for sub in self.iter_subdag():
            yield from sub.head

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __getitem__(self, idx: int | slice) -> tuple[Token, ...]:
        return tuple(self)[idx]

    # Properties --------------------------------------------------------------

    @property
    def idx(self) -> int:
        """Index of the head token."""
        return self.tok.i

    @property
    def head(self) -> Component:
        """Head component of the phrase."""
        return self.sent.cmap[self.idx]

    @property
    def lead(self) -> Self:
        """Lead phrase."""
        return self.sent.pmap[self._lead] if self._lead is not None else self

    @property
    def is_lead(self) -> Self:
        """Is the phrase a lead phrase."""
        return self.lead is self

    @property
    def tokens(self) -> tuple[Token, ...]:
        return tuple(self)

    @property
    def neg(self) -> Token | None:
        return self.head.neg

    @property
    def data(self) -> dict[str, Any]:
        return {
            **super().data,
            "lead": self._lead
        }

    @property
    def children(self) -> PGType:
        """Child phrases."""
        return PhraseGroup(self.sent.graph[self])

    @property
    def parents(self) -> PGType:
        """Parent phrases."""
        return PhraseGroup(self.sent.graph.rev[self])

    @property
    def subdag(self) -> PGType:
        """Phrasal proper subdag."""
        return PhraseGroup(self.iter_subdag(skip=1))

    @property
    def supdag(self) -> PGType:
        """Phrasal proper superdag."""
        return PhraseGroup(self.iter_supdag(skip=1))

    @property
    def depth(self) -> int:
        """Depth of the phrase within the phrasal tree of the sentence."""
        if (parents := self.parents):
            return min(p.depth + 1 for p in parents)
        return 0

    @property
    def conjuncts(self) -> Conjuncts:
        """Conjoined phrases."""
        if (conjs := self.sent.conjs.get(self._lead)):
            return conjs.copy(members=[
                m for m in conjs.members if m is not self
            ])
        return Conjuncts()

    @property
    def group(self) -> Conjuncts:
        """Group of self and its conjoined phrases."""
        return self.sent.conjs.get(self._lead) \
            or Conjuncts([self])

    @property
    @controlled
    def verb(self) -> PGType:
        """Return ``self`` if VP or nothing otherwise."""
        return PhraseGroup((self,)) \
            if isinstance(self, VerbPhrase) else PhraseGroup()
    @property
    @controlled
    def subj(self) -> PGType:
        """Subject phrases."""
        subjects = []
        for c in self.children:
            if c.dep & Dep.subj:
                subjects.append(c)
            elif c.dep & Dep.agent:
                subjects.extend(c.subj)
        return PhraseGroup(subjects)
    @property
    @controlled
    def dobj(self) -> PGType:
        """Direct object phrases."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.dobj
        )
    @property
    @controlled
    def iobj(self) -> PGType:
        """Indirect object phrases."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.iobj
        )
    @property
    @controlled
    def desc(self) -> PGType:
        """Description phrases."""
        return PhraseGroup(
            c for c in self.children if c.dep & (Dep.desc | Dep.misc)
        )
    @property
    @controlled
    def cdesc(self) -> PGType:
        """Clausal descriptions."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.cdesc
        )
    @property
    @controlled
    def adesc(self) -> PGType:
        """Adjectival complement descriptions."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.adesc
        )
    @property
    @controlled
    def prep(self) -> PGType:
        """Prepositions."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.prep
        )
    @property
    @controlled
    def pobj(self) -> PGType:
        """Prepositional objects."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.pobj
        )
    @property
    @controlled
    def subcl(self) -> PGType:
        """Subclauses."""
        return PhraseGroup(
            c for c in self.children
            if (c.dep & Dep.subcl) \
            or (isinstance(c, VerbPhrase) and (c.dep & Dep.acl))
        )
    @property
    @controlled
    def relcl(self) -> PGType:
        """Relative clausses."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.relcl
        )
    @property
    @controlled
    def xcomp(self) -> PGType:
        """Open clausal complements."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.xcomp
        )
    @property
    @controlled
    def appos(self) -> PGType:
        """Appositional modifiers."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.appos
        )
    @property
    @controlled
    def nmod(self) -> PGType:
        """Nominal modifiers."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.nmod
        )

    @property
    def components(self) -> DataTuple[Component]:
        return self.iter_subdag().get("head").tuple

    @component
    @property
    def verbs(self) -> DataTuple[Verb]:
        return self.components.filter(lambda c: isinstance(c, Verb)).tuple

    @component
    @property
    def nouns(self) -> DataTuple[Noun]:
        return self.components.filter(lambda c: isinstance(c, Noun)).tuple

    @component
    @property
    def preps(self) -> DataTuple[Verb]:
        return self.components.filter(lambda c: isinstance(c, Prep)).tuple

    @component
    @property
    def descs(self) -> DataTuple[Verb]:
        return self.components.filter(lambda c: isinstance(c, Desc)).tuple

    @property
    def vector(self) -> np.ndarray[tuple[int], np.floating]:
        comps = tuple(self.components)
        return sum(c.vector for c in comps) / len(comps)

    # Methods -----------------------------------------------------------------

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

    def iter_subdag(self, *, skip: int = 0) -> DataIterator[Self]:
        """Iterate over phrasal subtree and omit ``skip`` first items.

        Each phrase is emitted only when reached the first time
        during the depth-first search.
        """
        def _iter():
            yield self
            for child in self.children:
                yield from child.iter_subdag(skip=0)
        return DataIterator(islice(unique_everseen(_iter(), key=lambda p: p.idx), skip, None))

    def iter_supdag(self, *, skip: int = 0) -> DataIterator[Self]:
        """Iterate over phrasal supertree and omit ``skip`` first items.

        Each phrase is emitted only when reached the first time
        during the depth-first search.
        """
        def _iter():
            yield self
            for parent in self.parents:
                yield from parent.iter_supdag(skip=0)
        return DataIterator(islice(unique_everseen(_iter(), key=lambda p: p.idx), skip, None))

    def dfs(self, subdag: bool = True) -> DataTuple[DataTuple[Self]]:
        """Depth-first search.

        Parameters
        ----------
        subdag
            Should search be performed in the subgraph direction
            (i.e. through the children).
        """
        attr = "children" if subdag else "parents"
        def _dfs(phrase, chain=()):
            if (adjacent := getattr(phrase, attr)):
                for p in adjacent:
                    new_chain = list(chain)
                    new_chain.append(p)
                    yield from _dfs(p, chain=new_chain)
            else:
                yield DataTuple(chain)
        return DataIterator(_dfs(self))

    def similarity(self, *args: Any, **kwds: Any) -> float:
        """Structured similarity with respect to other phrase or sentence."""
        return self.Similarity(self, *args, **kwds).similarity

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
    ) -> Iterable[tuple[Token, Role | None]]:
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
                return typ(comp.tok, **kwds)
        raise ValueError(f"no matching phrase type for '{cls.cname(comp)}'")

    def to_data(self) -> dict[str, Any]:
        """Serialize to a data dictionary."""
        return {
            "@class": self.alias,
            "head": self.tok.i,
            "dep": self.dep.name,
            "sconj": self.sconj.i if self.sconj else None,
            "lead": self._lead
        }

    @classmethod
    def from_data(cls, doc: Doc, data: dict[str, Any]) -> Self:
        """Construct from sentence and data dictionary."""
        data = data.copy()
        typ = cls.types[data.pop("@class")]
        tok = doc[data["head"]]
        kwds = dict(
            dep=Dep.from_name(data["dep"]),
            sconj=doc[i] if (i := data["sconj"]) is not None else None,
            lead=data["lead"]
        )
        return typ(tok, **kwds)


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
