# pylint: disable=no-name-in-module
from typing import Any, Union, Callable, ClassVar, Self, Iterable, Mapping
from typing import Literal, TypeAlias
from abc import abstractmethod
from itertools import islice
from more_itertools import unique_everseen
import numpy as np
from spacy.vocab import Vocab
from spacy.vectors import Vectors
from .abc import TokenElement
from .components import Component, Verb, Noun, Desc, Prep
from .conjuncts import PhraseGroup, Conjuncts
from ..nlp.tokens import Doc, Token
from ..symbols import Role, Dep
from ..abc import labelled
from ..datastruct import DataIterable, DataTuple, DataChain
from ..utils.misc import cosine_similarity, best_matches


part = labelled("part")
PGType: TypeAlias = PhraseGroup["Phrase"]
PVSpecType: TypeAlias = dict[str, Union[
    str, Iterable[str],
    Callable[[Union["Phrase", Component, Token]], float]
]]
_what_type: TypeAlias = \
    str | Callable[["Phrase"], DataTuple[Union["Phrase", Component]]]
_what_vals = ("phrases", "components")


class Phrase(TokenElement):
    """Sentence phrase class.

    Attributes
    ----------
    tok
        Head token object.
    dep
        Dependency relative to the (main) parent.
    sconj
        Subordinating conjunction token.
    lead
        Lead phrase, initialized from index.
    """
    # pylint: disable=too-many-public-methods
    __slots__ = ("dep", "sconj", "_lead")
    alias: ClassVar[str] = "Phrase"
    part_names: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        tok: Token,
        *,
        dep: Dep = Dep.misc,
        sconj: Token | None = None,
        lead: int | None = None
    ) -> None:
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
        if (conjs := self.sent.conjs.get(self.lead)):
            return conjs.copy(members=[
                m for m in conjs.members if m is not self
            ])
        return Conjuncts()

    @property
    def group(self) -> Conjuncts:
        """Group of self and its conjoined phrases."""
        return self.sent.conjs.get(self._lead) \
            or Conjuncts([self])

    @part
    @property
    def verb(self) -> PGType:
        """Return ``self`` if VP or nothing otherwise."""
        return PhraseGroup((self,)) \
            if isinstance(self, VerbPhrase) else DataChain()
    @part
    @property
    def subj(self) -> PGType:
        """Subject phrases."""
        subjects = []
        for c in self.children:
            if c.dep & Dep.subj:
                subjects.append(c)
            elif c.dep & Dep.agent:
                subjects.extend(c.subj)
        return PhraseGroup(subjects)
    @part
    @property
    def dobj(self) -> PGType:
        """Direct object phrases."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.dobj
        )
    @part
    @property
    def iobj(self) -> PGType:
        """Indirect object phrases."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.iobj
        )
    @part
    @property
    def desc(self) -> PGType:
        """Description phrases."""
        return PhraseGroup(
            c for c in self.children if c.dep & (Dep.desc | Dep.misc)
        )
    @part
    @property
    def cdesc(self) -> PGType:
        """Clausal descriptions."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.cdesc
        )
    @part
    @property
    def adesc(self) -> PGType:
        """Adjectival complement descriptions."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.adesc
        )
    @part
    @property
    def prep(self) -> PGType:
        """Prepositions."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.prep
        )
    @part
    @property
    def pobj(self) -> PGType:
        """Prepositional objects."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.pobj
        )
    @part
    @property
    def subcl(self) -> PGType:
        """Subclauses."""
        return PhraseGroup(
            c for c in self.children
            if (c.dep & Dep.subcl) \
            or (isinstance(c, VerbPhrase) and (c.dep & Dep.acl))
        )
    @part
    @property
    def relcl(self) -> PGType:
        """Relative clausses."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.relcl
        )
    @part
    @property
    def xcomp(self) -> PGType:
        """Open clausal complements."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.xcomp
        )
    @part
    @property
    def appos(self) -> PGType:
        """Appositional modifiers."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.appos
        )
    @part
    @property
    def nmod(self) -> PGType:
        """Nominal modifiers."""
        return PhraseGroup(
            c for c in self.children if c.dep & Dep.nmod
        )

    @property
    def active_parts(self) -> tuple[str, ...]:
        return tuple(n for n in self.part_names if getattr(self, n))

    @property
    def components(self) -> DataChain[DataTuple[Component]]:
        return self.iter_subdag().get("head")

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

    def iter_subdag(self, *, skip: int = 0) -> Iterable[Self]:
        """Iterate over phrasal subtree and omit ``skip`` first items.

        Each phrase is emitted only when reached the first time
        during the depth-first search.
        """
        def _iter():
            yield self
            for child in self.children:
                yield from child.iter_subdag(skip=0)
        return DataIterable(islice(unique_everseen(_iter(), key=lambda p: p.idx), skip, None))

    def iter_supdag(self, *, skip: int = 0) -> Iterable[Self]:
        """Iterate over phrasal supertree and omit ``skip`` first items.

        Each phrase is emitted only when reached the first time
        during the depth-first search.
        """
        def _iter():
            yield self
            for parent in self.parents:
                yield from parent.iter_supdag(skip=0)
        return DataIterable(islice(unique_everseen(_iter(), key=lambda p: p.idx), skip, None))

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
        return DataIterable(_dfs(self))

    def similarity(
        self,
        spec: Self | str | Iterable[str] | PVSpecType,
        what: Literal[*_what_vals] | dict[str, _what_type] = _what_vals[0],
        *,
        recursive: bool = False,
        comp_vectors: bool = True,
        **kwds: Any
    ) -> float:
        """Similarity score with respect to specification.

        Parameters
        ----------
        spec
            Specification against which the phrase is to be compared.
            Can be another phrase, a string or an iterable of strings,
            which should be single words.
            A single strings is splitted at whitespace and turned into
            multiple words. Finally, an averaged word vector for all words
            is computed. Alternatively, a specification can have a form
            of dictionary mapping names of phrase parts (see ``what`` argument)
            to either strings convertible to word vectors (as used here)
            or callables, which will be applied to the values of the fields
            specified by the dictionary keys and are expected to return
            floats representing structured similarity scores.
        what
            Specifies whether comparison with another phrase should be
            based on phrase parts such as subjects, direct objects etc.,
            or on a simple comparison based on components, i.e. verbs and nouns.
            Alternatively, custom parts may be defined by providing a dictionary
            mapping part names to either phrase attribute names or arbitrary
            callables accepting a single phrase and producing an instance of
            :class:`segram.datastruct.DataTuple` populated with phrase
            or component objects.
        weights
            Dictionary mapping phrase part or component names or custom names
            as defined by ``what`` to arbitrary weights (which must be positive).
            The weights do not have to be normalized and sum up to one.
            They are used for reweighting importance of different parts
            during scoring.
        recursive
            Should a more accurate recurisve algorithm be used
            instead of a structured average vector approach.
            The former may be somewhat slower than than the latter,
            especially in the case of phrases with complex syntactic structures.
        comp_vectors
            If ``True`` then word vectors are based only on component
            head tokens instead of all tokens belonging to a given
            phrase or component.
        only, ignore
            Lists of part names to selectively use or ignored
            even if ``what`` defines more fields. Cannot use both
            at the same time.

        Raises
        ------
        ValueError
            If word vectors are not available.
        """
        kwds = dict(recursive=recursive, comp_vectors=comp_vectors, **kwds)
        return PhraseVectors(self, spec, what, **kwds).similarity

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


# -----------------------------------------------------------------------------

class PhraseVectors:
    """Phrase vectors comparison.

    This is a helper class for implementing
    various types of comparisons between vectors
    based on word vector embeddings.

    Attributes
    ----------
    phrase
        Main phrase of interest.
    """
    __doc__ += "\n".join(
        s[4:] for s in Phrase.similarity.__doc__.split("\n")[4:-1]
    )
    __slots__ = (
        "phrase", "spec", "what", "weights",
        "comp_vectors", "recursive", "only", "ignore",
    )

    def __init__(
        self,
        phrase: Phrase,
        spec: Phrase | str | Iterable[str] | PVSpecType,
        what: Literal[*_what_vals] | dict[str, _what_type] = _what_vals[0],
        *,
        weights: dict[str, float | int] | None = None,
        recursive: bool = False,
        comp_vectors: bool = True,
        only: str | Iterable[str] = (),
        ignore: str | Iterable[str] = ()
    ) -> None:
        if not phrase.doc.has_vectors:
            raise ValueError("word vectors not available")
        if what not in _what_vals and not isinstance(what, Mapping):
            raise ValueError(f"'what' has to be a mapping or one of {_what_vals}")
        if only and ignore:
            raise ValueError("'only' and 'ignore' cannot be used at the same time")
        weights = weights or {}
        if any(v < 0 for v in weights.values()):
            raise ValueError("weights must be non-negative")
        self.phrase = phrase
        self.spec = spec
        self.what = what
        self.weights = weights
        self.recursive = recursive
        self.only = only
        self.ignore = ignore
        self.comp_vectors = comp_vectors

    # Properties --------------------------------------------------------------

    @property
    def vocab(self) -> Vocab:
        return self.phrase.doc.vocab

    @property
    def vectors(self) -> Vectors:
        return self.vocab.vectors

    @property
    def similarity(self) -> float:
        """Structured similarity between ``self.phrase`` and ``self.spec``."""
        if isinstance(self.spec, Phrase | Component):
            if self.recursive:
                return self._sim_recursive(self.phrase, self.spec)
            return self._sim_phrase(self.phrase, self.spec)
        if isinstance(self.spec, str | Iterable | Mapping):
            return self._sim_custom_spec(self.phrase, self.spec)
        pcn = Phrase.cname()
        raise ValueError(
            f"specification must be a '{pcn}' instance "
            f"or a 'dict', not '{self.spec.__class__.__name__}'"
        )

    # Internals ---------------------------------------------------------------

    def _sim_recursive(self, phrase: Phrase, other: Phrase) -> float:
        sim = 0
        total_weight = 0
        if self._is_name_ok((name := "head")):
            total_weight += self.weights.get(name, 1)
            sim += self._sim(phrase.head, other.head) * total_weight
        active_parts = set(phrase.active_parts).union(other.active_parts)
        for name in active_parts:
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
            best = best_matches(sps, ops, self._sim_recursive)
            denom = max(len(ops), len(sps))
            add_sim = sum(x for x, *_ in best)
            sim += add_sim * w / denom
        if total_weight == 0:
            return 0.
        return sim / total_weight

    def _sim_phrase(self, phrase: Phrase, other: Phrase) -> float:
        # pylint: disable=too-many-locals
        sdict = self._get_parts(phrase)
        odict = self._get_parts(other)
        shared = set(sdict).intersection(odict)
        denom = len(set(sdict).union(odict))
        num = len(shared)
        sdict = {
            k: self._get_vectors(v) for k, v in sdict.items()
            if k in shared and self._is_name_ok(k)
        }
        odict = {
            k: self._get_vectors(v) for k, v in odict.items()
            if k in shared and self._is_name_ok(k)
        }

        sim = 0
        total_weight = 0
        if self._is_name_ok((name := "head")):
            total_weight += self.weights.get(name, 1)
            sim += self._sim(phrase.head, other.head) * total_weight
            denom += 1
            num += 1

        vocab = phrase.doc.vocab
        dtype = vocab.vectors.data.dtype
        vlen = vocab.vectors_length
        weights = np.empty(len(sdict), dtype=dtype)
        svecs = np.empty((len(sdict), vlen), dtype=dtype)
        ovecs = np.empty_like(svecs)

        for i, kv in enumerate(sdict.items()):
            name, svec = kv
            svecs[i] = svec
            ovecs[i] = odict[name]
            weights[i] = self.weights.get(name, 1)

        total_weight += weights.sum()
        if total_weight == 0:
            return 0.
        cos = cosine_similarity(svecs, ovecs, aligned=True)
        return (sim + (cos * weights).sum()) / total_weight * (num / denom)

    def _sim_custom_spec(self, phrase: Phrase, spec: PVSpecType) -> float:
        if isinstance(spec, Mapping):
            pdict = self._get_parts(phrase)
            sim = 0
            denom = 0
            num = 0
            total_weight = 0
            for field, req in spec.items():
                denom += 1
                if not (val := pdict.get(field)):
                    continue
                w = self.weights.get(field, 1)
                total_weight += w
                num += 1
                if isinstance(req, Callable):
                    sim += req(val) * w
                else:
                    req = self._get_text_vector(req)
                    if isinstance(val, Phrase | Component | Token):
                        vector = val.vector
                    elif isinstance(val, Iterable):
                        vector = sum(map(lambda x: x.vector, val))
                    sim += cosine_similarity(vector, req) * w
            if total_weight == 0:
                return 0.
            return sim / total_weight * (num / denom)

        vector = self._get_text_vector(spec)
        return cosine_similarity(phrase.vector, vector)

    def _sim(
        self,
        X: np.ndarray[tuple[int] | tuple[int, int], np.floating],
        Y: np.ndarray[tuple[int] | tuple[int, int], np.floating]
    ) -> float:
        if not isinstance(X, np.ndarray):
            X = X.vector
            Y = Y.vector
        if X.ndim > 1:
            sim = cosine_similarity(X, Y, aligned=True)
        else:
            sim = cosine_similarity(X, Y)
            if not isinstance(sim, np.ndarray):
                return sim
            if sim.size <= 0:
                return 0.
            if sim.ndim == 2:
                axis = 1 if sim.shape[0] <= sim.shape[1] else 0
                sim = sim.max(axis=axis)
        return sim.mean()

    def _is_name_ok(self, name: str) -> bool:
        if self.ignore:
            return name not in self.ignore
        if self.only:
            return name in self.only
        return True

    def _get_parts(self, phrase: Phrase) -> dict[str, DataTuple[Phrase | Component]]:
        pdict = {}
        if isinstance(self.what, Mapping):
            for k, v in self.what.items():
                if isinstance(v, str):
                    pdict[k] = getattr(phrase, v)
                else:
                    pdict[k] = v(phrase)
        elif self.what == "components":
            for comp in phrase.components:
                pdict.setdefault(comp.alias.lower(), []).append(comp)
            pdict = { k: tuple(v) for k, v in pdict.items() }
        else:
            for name in phrase.part_names:
                if (value := getattr(phrase, name)):
                    pdict[name] = value
        return pdict

    def _get_vectors(self, seq: DataTuple):
        if self.what == "phrases" and self.comp_vectors:
            seq = [ c for p in seq for c in p.components ]
        if (vec := [ x.vector for x in seq ]):
            return sum(vec) / len(vec)
        return vec

    def _get_text_vector(
        self,
        toks: str | Iterable[str]
    ) -> np.ndarray[tuple[int], np.floating]:
        if isinstance(toks, str):
            toks = toks.strip().split()
        toks = tuple(toks)
        if not toks:
            raise ValueError("cannot fetch word vectors; empty token list")
        return sum(self._get_single_vec(tok) for tok in toks) / len(toks)

    def _get_single_vec(self, tok: str | int) -> np.ndarray[tuple[int], np.floating]:
        try:
            return self.vectors[tok]
        except KeyError:
            vlen = self.vocab.vectors_length
            dtype = self.vectors.data.dtype
            return np.zeros(vlen, dtype=dtype)
