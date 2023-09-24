from __future__ import annotations
from typing import Any, Callable, ClassVar, Self, Iterable
from abc import abstractmethod
from itertools import islice, product
from more_itertools import unique_everseen
import numpy as np
from .abc import SentElement
from .components import Component, Verb, Noun, Desc, Prep
from .conjuncts import Conjuncts
from ..nlp.tokens import Token
from ..symbols import Role, Dep
from ..abc import labelled
from ..datastruct import DataSequence, DataChain
from ..utils.misc import cosine_similarity


controlled = labelled("part")
PGType = DataChain[Conjuncts["Phrase"]]
PVSpecType = dict[str, str | Iterable[str] | Callable[["Phrase"], bool]]


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
        sconj: Token | None = None,
        lead: int | None = None
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

    def __iter__(self) -> Iterable[Phrase]:
        yield from self.tokens

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int | slice) -> Phrase | list[Phrase]:
        return self.tokens[idx]

    def __contains__(self, other: Phrase | Component | Token) -> bool:
        if self.is_comparable_with(other):
            return any(other == p for p in self.iter_subdag(skip=1))
        if isinstance(other, Component):
            return any(p.head == other for p in self.iter_subdag())
        if isinstance(other, Token):
            return any(other in p.head for p in self.iter_subdag())
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
    def tokens(self) -> tuple[Token, ...]:
        return tuple(t for t, _ in self.iter_token_roles())

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
        return Conjuncts.get_chain(self.sent.graph[self])

    @property
    def parents(self) -> PGType:
        """Parent phrases."""
        return Conjuncts.get_chain(self.sent.graph.rev[self])

    @property
    def subdag(self) -> PGType:
        """Phrasal proper subdag."""
        return Conjuncts.get_chain(self.iter_subdag(skip=1))

    @property
    def supdag(self) -> PGType:
        """Phrasal proper superdag."""
        return Conjuncts.get_chain(self.iter_supdag(skip=1))

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
    @controlled
    def verb(self) -> PGType:
        """Return ``self`` if VP or nothing otherwise."""
        return Conjuncts.get_chain((self,)) \
            if isinstance(self, VerbPhrase) else DataChain()

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
        return Conjuncts.get_chain(subjects)

    @property
    @controlled
    def dobj(self) -> PGType:
        """Direct object phrases."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.dobj
        )
    @property
    @controlled
    def iobj(self) -> PGType:
        """Indirect object phrases."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.iobj
        )
    @property
    @controlled
    def desc(self) -> PGType:
        """Description phrases."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & (Dep.desc | Dep.misc)
        )
    @property
    @controlled
    def cdesc(self) -> PGType:
        """Clausal descriptions."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.cdesc
        )
    @property
    @controlled
    def adesc(self) -> PGType:
        """Adjectival complement descriptions."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.adesc
        )
    @property
    @controlled
    def prep(self) -> PGType:
        """Prepositions."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.prep
        )
    @property
    @controlled
    def pobj(self) -> PGType:
        """Prepositional objects."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.pobj
        )
    @property
    @controlled
    def subcl(self) -> PGType:
        """Subclauses."""
        return Conjuncts.get_chain(
            c for c in self.children
            if (c.dep & Dep.subcl) \
            or (isinstance(c, VerbPhrase) and (c.dep & Dep.acl))
        )
    @property
    @controlled
    def relcl(self) -> PGType:
        """Relative clausses."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.relcl
        )
    @property
    @controlled
    def xcomp(self) -> PGType:
        """Open clausal complements."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.xcomp
        )
    @property
    @controlled
    def appos(self) -> PGType:
        """Appositional modifiers."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.appos
        )
    @property
    @controlled
    def nmod(self) -> PGType:
        """Nominal modifiers."""
        return Conjuncts.get_chain(
            c for c in self.children if c.dep & Dep.nmod
        )

    @property
    def active_parts(self) -> tuple[str, ...]:
        return tuple(n for n in self.part_names if getattr(self, n))

    # Methods -----------------------------------------------------------------

    def iter_subdag(self, *, skip: int = 0) -> Iterable[Phrase]:
        """Iterate over phrasal subtree and omit ``skip`` first items.

        Each phrase is emitted only when reached the first time
        during the depth-first search.
        """
        def _iter():
            yield self
            for child in self.children:
                yield from child.iter_subdag(skip=0)
        yield from islice(unique_everseen(_iter(), key=lambda p: p.idx), skip, None)

    def iter_supdag(self, *, skip: int = 0) -> Iterable[Phrase]:
        """Iterate over phrasal supertree and omit ``skip`` first items.

        Each phrase is emitted only when reached the first time
        during the depth-first search.
        """
        def _iter():
            yield self
            for parent in self.parents:
                yield from parent.iter_supdag(skip=0)
        yield from islice(unique_everseen(_iter(), key=lambda p: p.idx), skip, None)

    def dfs(self, subdag: bool = True) -> DataSequence[DataSequence[Phrase]]:
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
                yield DataSequence(chain)
        return DataSequence(_dfs(self))

    def similarity(self, spec: Phrase | PVSpecType) -> float:
        """Similarity score with respect to specification.

        Parameters
        ----------
        spec
            Specification in the form of another phrase,
            template token(s) or a specification dictionary
            mapping template token(s) to different
            or a specification dictionary.
        """

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
    ) -> Iterable[Token, Role | None]:
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

    # Internals ---------------------------------------------------------------

    def _group_components(self) -> dict[str, list[Component]]:
        cdict = {}
        for part in self.part_names:
            comps = self.subdag.get(part).filter(None).flat.get("head")
            if comps:
                cdict[part] = comps
        return cdict

    def _get_component_vectors(
        self,
        *,
        average: bool = False
    ) -> dict[str, np.ndarray[tuple[int, int], np.floating]]:
        if not self.head.tok.has_vectors:
            raise AttributeError("word vectors not defined")
        vlen = self.head.tok.vocab.vectors_length
        dtype = self.head.tok.vocab.vectors.data.dtype
        cdict = self._group_components()
        for k, v in cdict.items():
            vecs = np.zeros((len(v), vlen), dtype=dtype)
            for i, c in enumerate(v):
                vecs[i] = c.vector
            cdict[k] = vecs
        return cdict

    def _sim_comps_outer(self, other: Phrase, **kwds: Any) -> float:
        # pylint: disable=protected-access
        sim = self.head.similarity(other.head)
        svecs = self._get_component_vectors()
        ovecs = other._get_component_vectors()
        n = 1
        for k, sv in svecs.items():
            if k not in ovecs:
                continue
            n += 1
            ov = ovecs[k]
            cos = cosine_similarity(sv, ov, **kwds)
            if isinstance(cos, np.ndarray):
                axis = 0 if cos.shape[0] >= cos.shape[1] else 1
                cos = cos.max(axis=axis).mean()
            sim += cos
        return float(sim) / n



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


# -----------------------------------------------------------------------------

class PhraseVectors:
    """Phrase vectors comparison.

    This is a helper class for implementing
    various types of comparisons between vectors
    based on word vector embeddings.
    """
    __slots__ = ("phrase", "spec", "what", "weights", "recursive", "outer", "only", "ignore")
    _what_type = str | Iterable[str] | Callable[[Phrase], Iterable[Phrase]]

    def __init__(
        self,
        phrase: Phrase,
        spec: Phrase | str | Iterable[str] | PVSpecType,
        what: str | dict[str, _what_type],
        *,
        weights: dict[str, float | int] | None = None,
        recursive: bool = False,
        outer: bool = False,
        only: str | Iterable[str] = (),
        ignore: str | Iterable[str] = ()
    ) -> None:
        self.phrase = phrase
        self.spec = spec
        self.what = what
        self.weights = weights or {}
        self.recursive = recursive
        self.outer = outer
        self.only = only
        self.ignore = ignore
        if self.only and self.ignore:
            raise ValueError("'only' and 'ignore' cannot be used at the same time")

    # Methods -----------------------------------------------------------------

    def similarity(self, **kwds: Any) -> float:
        """Structured similarity between ``self.phrase`` and ``self.spec``.

        Parameters
        ----------
        **kwds
            Passed to :func:`segram.utils.misc.cosine_similarity`.
        """

    # Internals ---------------------------------------------------------------

    def _sim_recursive(self, phrase: Phrase, other: Phrase, **kwds: Any) -> float:
        sim = 0
        total_weight = 0
        if self._is_name_ok((name := "head")):
            total_weight += self.weights.get(name, 1)
            sim += self._cos(phrase.head, other.head, outer=self.outer, **kwds) \
                * total_weight
        for name in set(phrase.active_parts).union(other.active_parts):
            if not self._is_name_ok(name):
                continue
            sps = getattr(phrase, name)

            if phrase in sps.flat:
                # This is to prevent infinite recursion
                # happening for verb phrases/clauses
                continue

            w = self.weights.get(name, 1)
            total_weight += w

            ops = getattr(other, name)
            if not sps or not ops:
                continue
            best_matches = self._best_matches(sps, ops, self._sim_recursive, **kwds)
            denom = max(len(ops), len(sps))
            sim += sum(x for x, *_ in best_matches) / denom * w
        if total_weight == 0:
            return 0
        return sim / total_weight

    def _cos(self, X, Y, outer=False, **kwds) -> float:
        if not isinstance(X, np.ndarray):
            X = X.vectors if outer else X.vector
            Y = Y.vectors if outer else Y.vector
        if not outer and X.ndim != 1:
            sim = cosine_similarity(X, Y, aligned=True, **kwds)
        else:
            sim = cosine_similarity(X, Y, **kwds)
            if not isinstance(sim, np.ndarray):
                return sim
            if sim.ndim == 2:
                axis = 0 if sim.shape[0] <= sim.shape[1] else 1
                sim = sim.max(axis=axis)
        return sim.mean()

    def _is_name_ok(self, name: str) -> bool:
        if self.ignore:
            return name not in self.ignore
        if self.only:
            return name in self.only
        return True

    def _best_matches(self, phrases, specs, func, **kwds):
        idx = 1 if len(phrases) <= len(specs) else 2
        yield from unique_everseen(
            sorted((
                (func(p, s, **kwds), p, s)
                for p, s in product(phrases, specs)
            ), key=lambda x: -x[0]
        ), key=lambda x: x[idx])
