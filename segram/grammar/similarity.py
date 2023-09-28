# pylint: disable=no-name-in-module
from typing import Literal, Iterable, Mapping
import numpy as np
from . import Component, Phrase, Sent, Doc
from .abc import GrammarSimilarity
from ..datastruct import DataTuple
from ..utils.misc import best_matches, cosine_similarity


SpecType = dict[str, str | Iterable[str] | Phrase | Sent | Doc]
FloatVec = np.ndarray[tuple[int], np.floating]
_sim_methods = ("components", "phrases", "both", "recursive", "average")


class PhraseSimilarity(GrammarSimilarity, register_with=Phrase):
    """Structured similarity between phrases and sentences."""
    __slots__ = ("method", "weights", "decay_rate", "only", "ignore")

    def __init__(
        self,
        phrase: Phrase,
        spec: Phrase | str | Iterable[str] | SpecType,
        method: Literal[*_sim_methods] = _sim_methods[0],
        *,
        weights: dict[str, float | int] | None = None,
        decay_rate: float = 1,
        only: str | Iterable[str] = (),
        ignore: str | Iterable[str] = ()
    ) -> None:
        if not phrase.doc.has_vectors:
            raise RuntimeError("word vectors not available")
        if method not in _sim_methods:
            raise ValueError(f"'method' has to be one of {_sim_methods}")
        if only and ignore:
            raise ValueError("'only' and 'ignore' cannot be used at the same time")
        weights = weights or {}
        if any(v < 0 for v in weights.values()):
            raise ValueError("weights must be non-negative")
        if decay_rate <= 0:
            raise ValueError("'decay_rate' must be positive")
        self.phrase = phrase
        self.spec = spec
        self.method = method
        self.weights = weights
        self.decay_rate = decay_rate
        self.only = only
        self.ignore = ignore

    # Properties --------------------------------------------------------------

    @property
    def phrase(self) -> Phrase:
        return self.element

    # Methods -----------------------------------------------------------------

    def get_similarity(self, phrase: Phrase, spec: SpecType) -> float:
        r"""Structured similarity between ``self.phrase`` and ``self.spec``.

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
        if isinstance(spec, Doc):
            spec = self._make_sent(Doc)
        if isinstance(spec, Sent):
            proots = spec.proots
            return sum(self.get_similarity(self.phrase, p) for p in proots) \
                / len(proots)
        if isinstance(spec, Phrase):
            if self.method == "recursive":
                return self._sim_recursive(phrase, spec)
            if self.method == "average":
                return cosine_similarity(phrase.vector, spec.vector)
            return self._sim_parts(phrase, spec)
        if isinstance(self.spec, str | Iterable | Mapping):
            return self._sim_custom(phrase, spec)
        pcn = self.phrase.cname()
        raise ValueError(
            f"cannot compare '{pcn}' with '{self.__class__.__name__}'"
        )

    # Internals ---------------------------------------------------------------

    def _sim_recursive(self, phrase: Phrase, other: Phrase, depth: int = 0) -> float:
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

            w = self.weights.get(name, 1) * self.decay_rate**(depth+1)
            total_weight += w

            ops = getattr(other, name)
            if not sps or not ops:
                continue
            best = best_matches(sps, ops, self._sim_recursive, depth=depth+1)
            denom = max(len(ops), len(sps))
            add_sim = sum(x for x, *_ in best)
            sim += add_sim * w / denom
        if total_weight == 0:
            return 0.
        return sim / total_weight

    def _sim_parts(self, phrase: Phrase, other: Phrase) -> float:
        sdict = self._get_parts(phrase)
        odict = self._get_parts(other)
        shared = set(sdict).intersection(odict)
        denom = sum(self.weights.get(k, 1) for k in set(sdict).union(odict))
        num = sum(self.weights.get(k, 1) for k in shared)
        sdict = {
            k: v for k, v in sdict.items()
            if k in shared and self._is_name_ok(k)
        }
        W = self.np.array([
            self.weights.get(k, 1) for k in shared
        ], dtype=self.vocab.vectors.data.dtype)
        odict = { k: odict[k] for k in sdict }
        svec = DataTuple(sdict.values()) \
            .map(lambda x: sum(c.vector for c in x)) \
            .pipe(self.np.vstack)
        ovec = DataTuple(odict.values()) \
            .map(lambda x: sum(c.vector for c in x)) \
            .pipe(self.np.vstack)
        cos = cosine_similarity(svec, ovec, aligned=True)
        sim = (cos * W).sum() * (num / denom) / W.sum()
        return sim

    def _sim_custom(self, phrase: Phrase, spec: SpecType) -> float:
        if isinstance(spec, Mapping):
            pdict = self._get_parts(phrase)
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
            sim *= (num / denom) / total_weight
        else:
            spec = self._get_text_vector(spec)
            sim = cosine_similarity(self.phrase.vector, spec)
        return sim

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
        keys = []
        if self.method in ("components", "both"):
            keys.extend(phrase.component_names)
        if self.method in ("phrase", "both"):
            keys.extend(phrase.part_names)
        if self.ignore:
            keys = [ k for k in keys if k not in self.ignore ]
        elif self.only:
            keys = [ k for k in keys if k in self.only]
        pdict = { k: getattr(phrase, k) for k in keys }
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

    @property
    def sent(self) -> Sent:
        return self.element

    @property
    def phrase(self) -> None:
        raise AttributeError(f"'{self.__class__.__name__}' object has not attribute 'phrase'")

    def get_similarity(self, sent: Sent, spec: SpecType) -> float:
        """Structured similarity between ``self.phrase`` and ``self.spec``."""
        # pylint: disable=arguments-renamed
        if isinstance(spec, Doc):
            spec = self._make_sent(spec)
        if isinstance(spec, Sent):
            proots = sent.proots
            oroots = sent.spec
            return sum (score for score, *_ in best_matches(
                proots, oroots, lambda s, o: self.get_similarity(s, o)
            )) / max(len(proots), len(oroots))
        return max(super().get_similarity(p, spec) for p in sent.proots)


class DocSimilarity(GrammarSimilarity, register_with=Doc):
    """Structured similarity between documents."""
