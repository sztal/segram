# pylint: disable=no-name-in-module
from typing import Optional, NamedTuple, Sequence, Iterable
from collections import Counter
from spacy.vocab import Vocab
from spacy.vectors import Vectors
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
        Word vector table.
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
        self.vectors = vectors
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
