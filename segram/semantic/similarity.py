# pylint: disable=no-name-in-module
from typing import Any, Literal, Iterable, Mapping
from abc import ABC, abstractmethod
from importlib import import_module
import re
import numpy as np
from spacy.vocab import Vocab
from spacy.vectors import Vectors
from ..grammar import Component, Phrase, Sent, Doc
from ..abc import init_class_attrs
from ..datastruct import DataTuple
from ..utils.misc import best_matches, cosine_similarity


SpecType = dict[str, str | Iterable[str] | Phrase | Sent | Doc]
FloatVec = np.ndarray[tuple[int], np.floating]
_sim_methods = ("components", "phrases", "recursive", "average")


class GrammarSimilarity(ABC):
    """Abstract base class for structured similarity scorers."""
    __slots__ = ("element", "spec", "np")
    slot_names: tuple[str, ...] = ()

    def __init__(self, element: "GrammarElement", spec: Any) -> None:
        self.element = element
        if not self.vocab.has_vector:
            raise RuntimeError("word vectors not available")
        self.spec = spec
        self.np = import_module(self.vocab.vectors.data.__class__.__module__)

    def __init_subclass__(cls, register_with: type["GrammarElement"]) -> None:
        init_class_attrs(cls, {
            "__slots__": "slot_names"
        }, check_slots=True)
        register_with.Similarity = cls
        ds = cls._get_docstring()
        ds = re.sub(r"(\n\s*)Attributes(\s*\n)", r"\1Parameters\2", ds)
        register_with.similarity.__doc__ += ds

    # Properties --------------------------------------------------------------

    @property
    def vocab(self) -> Vocab:
        return self.element.doc.vocab

    @property
    def vectors(self) -> Vectors:
        return self.vocab.vectors

    @property
    def similarity(self) -> float:
        sim = self.get_similarity(self.element, self.spec)
        return max(-1, min(sim, 1))

    @property
    @abstractmethod
    def config(self) -> dict[str, Any]:
        return {}

    # Methods -----------------------------------------------------------------

    @abstractmethod
    def get_similarity(self, element: "GrammarElement", spec: Any) -> float:
        """Get structured similarity between ``self.element`` and ``self.spec``."""

    # Internals ---------------------------------------------------------------

    @classmethod
    def _get_docstring(cls) -> str:
        return "\n"+"\n".join(cls.__doc__.split("\n")[1:-1])


class PhraseSimilarity(GrammarSimilarity, register_with=Phrase):
    r"""Structured similarity between phrases and sentences.

    All methods defined here are designed to ensure that:

    * Similarity of a phrase with respect to itself is ``1``.
    * Similarity ``x ~ y == y ~ x``.

    In some case the above may be true only approximately due to
    accumulation of floating point imprecision.

    Attributes
    ----------
    element
        Grammar phrase to compare.
    spec
        Specification against which the phrase is to be compared.
        Can be another phrase, a string or an iterable of strings,
        which should be single words. A single strings is splitted at
        whitespace and turned into multiple words.
        Finally, an averaged word vector for all words is computed.
        Alternatively, a specification can have a form
        of a dictionary mapping names of phrase parts or components
        (see :attr:`segram.grammar.phrases.Phrase.part_names`
        and :attr:`segram.grammar.phrase.Phrase.component_names`)
        to either strings or iterables of strings convertible to word
        vectors (as previously) or other phrases.
        Importantly, phrases can be also compared against
        :class:`segram.grammar.Sent` and :class:`segram.grammar.Doc`
        objects as long as they are comprised of a single sentence.
        See :class:`SentSimilarity` for details.
    method
        Method for calculating similarity between phrases:

        ``components``
            Components are grouped in buckets by type
            (verbs, nouns, prepositions and descriptions)
            and averaged vectors are compared between
            the same types. Finally, a weighted average
            (with weights defined by the ``weight`` parameter)
            is taken and rescaled with a factor ``shared / union``,
            where ``shared`` is the numebr of types present in
            both elements and ``union`` is the total number of unique
            types among both of them. Thus, the final result is akin
            to a fuzzy Jaccard similarity:

            .. math::

                J = \frac{|A \cap B|}{|A \cup B|}

        ``phrases``
            As above but based on phrase parts and phrase head compoents.
            See :attr:`segram.grammar.Phrase.part_names` for a full list.

        ``both``
            As above but components and phrases are used
            together.

        ``average``
            Simple average vectors calculated over all component
            head tokens are used. In this case weights are ignored.

        ``recursive``
            NOTE. Currently not implemented.
            First, head components are compared between two phrases,
            and then the same rule is applied recursively to all
            parts (subjects, direct objects etc.) where for each
            type elements of the two phrases are matched in pairs
            to maximize similarity. As previously, weights can be
            applied to different types and a Jaccard-like rescaling
            is applied. Additionaly, importance of nested phrases
            may be discounted using ``decay_rate`` parameter by
            rescaling each weight with a factor of ``decay_rate**depth``,
            where ``depth`` is calculated relative to the depth
            of the ``self.phrase``.

    weights
        Dictionary mapping phrase part or component names to arbitrary
        weights (which must be positive). The weights do not have to be
        normalized and sum up to one.
    decay_rate
        Additional parameter used when ``method="recursive"``,
        which controls the rate at which contributions coming
        from nested subphrases are discounted.
    only, ignore
        Lists of part or component names to selectively use or ignore.
        Both arguments cannot be used at the same time.

    Raises
    ------
    RuntimeError
        If word vectors are not available.
    """
    __slots__ = ("method", "weights", "decay_rate", "only", "ignore")

    def __init__(
        self,
        element: Phrase,
        spec: Phrase | str | Iterable[str] | SpecType,
        method: Literal[*_sim_methods] = _sim_methods[0],
        *,
        weights: dict[str, float | int] | None = None,
        decay_rate: float = 1,
        only: str | Iterable[str] = (),
        ignore: str | Iterable[str] = ()
    ) -> None:
        super().__init__(element, spec)
        if method not in _sim_methods:
            raise ValueError(f"'method' has to be one of {_sim_methods}")
        if only and ignore:
            raise ValueError("'only' and 'ignore' cannot be used at the same time")
        weights = weights or {}
        if any(v < 0 for v in weights.values()):
            raise ValueError("weights must be non-negative")
        if decay_rate <= 0:
            raise ValueError("'decay_rate' must be positive")
        self.method = method
        self.weights = weights
        self.decay_rate = decay_rate
        self.only = only
        self.ignore = ignore

    # Properties --------------------------------------------------------------

    @property
    def config(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "weights": self.weights,
            "decay_rate": self.decay_rate,
            "only": self.only,
            "ignore": self.ignore
        }

    # Methods -----------------------------------------------------------------

    def get_similarity(self, element: Phrase, spec: SpecType) -> float:
        r"""Structured similarity between ``self.phrase`` and ``self.spec``."""
        phrase = element
        if isinstance(spec, Doc):
            spec = self._make_sent(Doc)
        if isinstance(spec, Sent):
            proots = spec.proots
            return sum(self.get_similarity(phrase, p) for p in proots) \
                / len(proots)
        if isinstance(spec, Phrase):
            if self.method == "recursive":
                return self._sim_recursive(phrase, spec)
            if self.method == "average":
                return cosine_similarity(phrase.vector, spec.vector)
            return self._sim_parts(phrase, spec)
        if isinstance(self.spec, str | Iterable | Mapping):
            return self._sim_custom(phrase, spec)
        pcn = phrase.cname()
        raise ValueError(
            f"cannot compare '{pcn}' with '{self.__class__.__name__}'"
        )

    # Internals ---------------------------------------------------------------

    def _sim_recursive(self, phrase: Phrase, other: Phrase, depth: int = 0) -> float:
        raise NotImplementedError("'recursive' method is not yet implemented")
        # sim = 0
        # total_weight = 0
        # if self._is_name_ok((name := "head")):
        #     total_weight += self.weights.get(name, 1)
        #     sim += self._sim(phrase.head, other.head) * total_weight
        # active_parts = set(phrase.active_parts).union(other.active_parts)
        # for name in active_parts:
        #     if not self._is_name_ok(name):
        #         continue
        #     sps = getattr(phrase, name)

        #     if phrase in sps.flat:
        #         # This is to prevent infinite recursion
        #         # happening for verb phrases/clauses
        #         continue

        #     w = self.weights.get(name, 1) * self.decay_rate**(depth+1)
        #     total_weight += w

        #     ops = getattr(other, name)
        #     if not sps or not ops:
        #         continue
        #     # denom = max(len(ops), len(sps))
        #     best = best_matches(sps, ops, self._sim_recursive, depth=depth+1)
        #     add_sim = sum(x for x, *_ in best)
        #     # sim += add_sim * w / denom
        #     sim += add_sim * w
        # if not total_weight:
        #     return .0
        # return sim / total_weight

    def _sim_parts(self, phrase: Phrase, other: Phrase) -> float:
        sdict = self._get_parts(phrase)
        odict = self._get_parts(other)
        shared = set(sdict).intersection(odict)
        denom = sum(self.weights.get(k, 1) for k in set(sdict).union(odict))
        if not denom:
            return .0
        num = sum(self.weights.get(k, 1) for k in shared)
        sdict = {
            k: v for k, v in sdict.items()
            if k in shared and self._is_name_ok(k)
        }
        W = self.np.array([
            self.weights.get(k, 1) for k in shared
        ], dtype=self.vocab.vectors.data.dtype)
        w_total = W.sum()
        if not w_total:
            return .0
        odict = { k: odict[k] for k in sdict }
        svec = DataTuple(sdict.values()) \
            .map(lambda x: sum(c.vector for c in x)) \
            .pipe(self.np.vstack)
        ovec = DataTuple(odict.values()) \
            .map(lambda x: sum(c.vector for c in x)) \
            .pipe(self.np.vstack)
        cos = cosine_similarity(svec, ovec, aligned=True, nans_as_zeros=False)
        sim = self.np.nansum(cos * W) * (num / denom) / W.sum()
        return sim

    def _sim_custom(self, phrase: Phrase, spec: SpecType) -> float:
        if isinstance(spec, Mapping):
            invalid = set(spec) \
                - set(phrase.component_names) \
                - set(phrase.component_names) \
                - {"head"}
            if invalid:
                raise ValueError(f"incorrect specification fields: {invalid}")
            pdict = { k: getattr(phrase, k) for k in spec }
            sim = 0
            denom = 0
            num = 0
            total_weight = 0
            for key, _spec in spec.items():
                denom += 1
                if key not in pdict:
                    continue
                num += 1
                w = self.weights.get(key, 1)
                total_weight += 1
                parts = pdict[key]
                if not parts:
                    continue
                if isinstance(_spec, Doc):
                    _spec = self._make_sent(_spec)
                if isinstance(_spec, Phrase | Sent):
                    sim += max(self.get_similarity(p, _spec) for p in parts) \
                        * w
                elif isinstance(_spec, Iterable):
                    _spec = self._get_text_vector(_spec)
                    sim += max(cosine_similarity(p.vector, _spec) for p in parts) \
                        * w
                else:
                    raise ValueError(f"invalid specification '{_spec}' for key '{key}'")
            if not denom or not total_weight:
                return .0
            sim *= (num / denom) / total_weight
        else:
            spec = self._get_text_vector(spec)
            sim = cosine_similarity(phrase.vector, spec)
        return sim

    def _is_name_ok(self, name: str) -> bool:
        if self.ignore:
            return name not in self.ignore
        if self.only:
            return name in self.only
        return True

    def _get_parts(self, phrase: Phrase) -> dict[str, DataTuple[Phrase | Component]]:
        pdict = {}
        if self.method == "components":
            keys = phrase.component_names
        elif self.method == "phrases":
            keys = ("head", *phrase.controlled_names)
        else:
            raise ValueError(
                f"cannot calculate by parts comparison for method '{self.method}'"
            )
        if self.ignore:
            keys = [ k for k in keys if k not in self.ignore ]
        elif self.only:
            keys = [ k for k in keys if k in self.only]
        pdict = { k: v for k in keys if (v := getattr(phrase, k)) }
        return pdict

    def _get_text_vector(
        self,
        toks: str | Iterable[str]
    ) -> np.ndarray[tuple[int], np.floating]:
        if isinstance(toks, str):
            toks = toks.strip().split()
        toks = tuple(toks)
        if not toks:
            raise ValueError("cannot fetch word vectors; empty token list")
        vec = sum(self._get_single_vec(tok) for tok in toks) / len(toks)
        if vec.size == 0:
            raise ValueError("all provided tokens are out-of-vocabulary")
        return vec

    def _get_single_vec(self, tok: str | int) -> np.ndarray[tuple[int], np.floating]:
        try:
            return self.vectors[tok]
        except KeyError:
            vlen = self.vocab.vectors_length
            dtype = self.vectors.data.dtype
            return self.np.zeros(vlen, dtype=dtype)

    def _make_sent(self, doc: Doc) -> Phrase:
        if len(doc.sents[:2]) != 1:
            raise ValueError(
                "only documents with exactly one sentence "
                "can be compared with phrases"
            )
        return doc.sents[0]


class SentSimilarity(PhraseSimilarity, register_with=Sent):
    """Structured similarity between sentences and phrases."""
    # pylint: disable=protected-access
    __doc__ += PhraseSimilarity._get_docstring()

    @property
    def phrase(self) -> None:
        raise AttributeError(f"'{self.__class__.__name__}' object has not attribute 'phrase'")

    def get_similarity(self, element: Sent, spec: SpecType) -> float:
        """Structured similarity between ``self.phrase`` and ``self.spec``."""
        # pylint: disable=arguments-renamed
        sent = element
        if isinstance(spec, Doc):
            spec = self._make_sent(spec)
        if isinstance(spec, Sent):
            if self.method == "average":
                return cosine_similarity(sent.vector, spec.vector)
            if self.method == "components":
                return self._sim_parts(sent, spec)
            proots = sent.proots
            oroots = spec.proots
            return sum (score for score, *_ in best_matches(
                proots, oroots, lambda s, o: s.Similarity(s, o, **self.config) \
                    .similarity
            )) / max(len(proots), len(oroots))
        return max(
            p.Similarity(p, spec, **self.config).similarity
            for p in sent.proots
        )


class DocSimilarity(GrammarSimilarity, register_with=Doc):
    """Structured similarity between documents."""
    # pylint: disable=protected-access
    __doc__ += PhraseSimilarity._get_docstring()
