# pylint: disable=no-name-in-module
from typing import Any, Optional, NamedTuple, Sequence, Iterable, Self
from collections import Counter
from spacy.vocab import Vocab
from spacy.vectors import Vectors
from spacy.language import Language
from .abc import DocABC


class TokenDistributions(NamedTuple):
    text: Counter = Counter()
    lemma: Counter = Counter()


class Corpus(Sequence):
    """Corpus class.

    Attributes
    ----------
    vocab
        Main vocabulary store of the corpus.
    vectors
        Word vector table. If then ``vocab.vectors`` are used
        as fallback if possible.
    resolve_coref
        If ``True`` then token coreferences are resolved when
        calculating token text and lemma frequency distributions.
    """
    def __init__(
        self,
        vocab: Vocab,
        *,
        vectors: Optional[Vectors] = None,
        resolve_coref: bool = True
    ) -> None:
        self._dmap = {}
        self.vocab = vocab
        self._vectors = vectors
        self.resolve_coref = resolve_coref

    def __getitem__(self, idx: int | slice) -> DocABC | tuple[DocABC, ...]:
        return self.docs[idx]

    def __len__(self) -> int:
        return len(self._dmap)

    def __contains__(self, doc: DocABC) -> bool:
        if isinstance(doc, DocABC):
            return hash(doc) in self._dmap
        return NotImplemented

    # Properties --------------------------------------------------------------

    @property
    def docs(self) -> tuple[DocABC]:
        return tuple(self._dmap.values())

    @property
    def vectors(self) -> Vectors:
        return self.vectors or self.vocab.vectors

    @property
    def has_vectors(self) -> bool:
        return bool(self.vectors.data.size)

    # Methods -----------------------------------------------------------------

    def add_doc(self, doc: DocABC) -> None:
        """Add document to the corpus."""
        if doc not in self:
            self._dmap[hash(doc)] = doc
            self.vocab.dist.text.update(t.coref.text for t in doc)
            self.vocab.dist.lemma.update(t.coref.lemma for t in doc)

    def add_docs(self, docs: Iterable[DocABC]) -> None:
        """Add documents to the corpus."""
        for doc in docs:
            self.add_doc(doc)

    @classmethod
    def from_texts(
        cls,
        nlp: Language,
        texts: Iterable[str],
        pipe_kws: Optional[dict[str, Any]] = None,
        **kwds: Any
    ) -> Self:
        """Construct from texts.

        Parameters
        ----------
        nlp
            Language model to use to parse texts.
        texts
            Texts to parse.
        pipe_kws
            Keyword arguments passed to :meth:`spacy.language.Language.pipe`.
        **kwds
            Passed :meth:`__init__`.
            Vocabulary is taken from the language model.
        """
        obj = cls(nlp.vocab, **kwds)
        obj.add_docs(nlp.pipe(texts, **(pipe_kws or {})))
        return obj
