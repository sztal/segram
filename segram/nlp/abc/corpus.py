from typing import Sequence, Iterable
from .base import NLP
from .vocab import VocabABC
from .tokens import DocABC


class CorpusABC(NLP, Sequence):
    """Corpus abstract base class.

    Attributes
    ----------
    docs
        Sequence of documents.
        It can be extended after initialization.
    vocab
        Vocabulary.
    """
    __slots__ = ("_dmap", "_vocab")

    def __init__(self) -> None:
        self._dmap = {}
        self._vocab = None

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
    def docs(self) -> list[DocABC]:
        return tuple(self._dmap.values())

    @property
    def vocab(self) -> VocabABC:
        return self._vocab

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
