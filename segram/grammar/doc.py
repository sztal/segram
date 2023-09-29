# pylint: disable=no-name-in-module
from typing import Any, Self, Mapping
import numpy as np
from spacy.tokens import Doc as SpacyDoc, Token
from spacy.vocab import Vocab
from .abc import DocElement
from .sent import Sent
from .phrases import Phrase
from .components import Component
from ..nlp.tokens import Doc as DocNLP
from .. import __title__
from ..utils.misc import sort_map
from ..datastruct import DataIterator, DataTuple


class Doc(DocElement):
    """Grammar document class.

    This is grammar equivalent of NLP documents.

    Attributes
    ----------
    doc
        Underlying NLP document.
    smap
        Mapping from sentence ids to sentences.
    """
    __slots__ = ("smap",)
    alias = "Doc"

    def __init__(
        self,
        doc: DocNLP | SpacyDoc,
        smap: Mapping[tuple[int, int], Sent] | None = None
    ) -> None:
        alias = getattr(doc._, __title__+"_alias")
        if isinstance(doc, SpacyDoc):
            doc = getattr(doc._, alias+"_sns")
        setattr(doc._, alias+"_doc", self)
        super().__init__(doc)
        if smap is None:
            self.smap = {}  # Little trick to make 's.grammar' work
            smap = { (s.start, s.end): s.grammar for s in doc.sents }
        self.smap = sort_map(smap)

    # Properties --------------------------------------------------------------

    @property
    def sents(self) -> DataTuple[Sent]:
        """Sentences in the document."""
        return DataTuple(self.smap.values())

    @property
    def phrases(self) -> DataIterator[Phrase]:
        """Phrase in the document grouped by sentences and conjunct groups."""
        return DataIterator(s.phrases for s in self.sents).flat

    @property
    def components(self) -> DataIterator[Component]:
        """Unique components by sentences."""
        return DataIterator(s.components for s in self.sents).flat

    @property
    def tokens(self) -> DataTuple[Token]:
        return DataTuple(self.doc)

    @property
    def has_vectors(self) -> bool:
        """Check if document is equiped with word vectors."""
        return self.doc.has_vectors

    @property
    def vocab(self) -> Vocab:
        return self.doc.vocab

    @property
    def vector(self) -> np.ndarray[tuple[int], np.floating]:
        return self.doc.vector

    # Methods -----------------------------------------------------------------

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, Doc)

    def to_str(self, **kwds: Any) -> str:
        """Represent as string."""
        return " ".join(s.to_str(**kwds) for s in self.sents)

    def to_data(self, *, grammar: bool = True) -> dict[str, Any]:
        """Dump to data dictionary.

        Parameters
        ----------
        grammar
            Should grammar data be serialized too.
        """
        if grammar:
            key = f"{self.doc.alias}_data"
            smap = { idx: s.to_data() for idx, s in self.smap.items() }
            setattr(self.doc._, key, smap)
        return self.doc.to_data()

    def copy(self) -> Self:
        # pylint: disable=arguments-differ
        return self.from_data(self.to_data())

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> Self:
        """Construct from data dictionary
        as returned by :meth:`segram.nlp.tokens.Doc.to_data`.
        """
        doc = DocNLP.from_data(data)
        grammar = cls.from_doc(doc, smap={})
        smap = getattr(doc._, f"{doc.alias}_data")
        for idx, dct in smap.items():
            grammar.smap[idx] = grammar.types.Sent.from_data(doc, dct)
        return grammar

    @classmethod
    def from_doc(cls, doc: DocNLP, *args: Any, **kwds: Any) -> Self:
        """Construct from NLP document object."""
        if isinstance(doc, SpacyDoc):
            doc = getattr(doc._, doc.alias+"_sns")
        typ = doc.get_grammar_type()
        return typ.types.Doc(doc, *args, **kwds)
