from typing import Any, Self, Mapping
from spacy.tokens import Doc as SpacyDoc, Token
from .abc import DocElement
from .sent import Sent
from .phrases import Phrase
from .components import Component
from ..nlp.tokens import Doc as DocNLP
from .. import settings
from ..utils.misc import sort_map
from ..datastruct import DataSequence, DataChain


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
        alias = settings.spacy_alias
        if isinstance(doc, SpacyDoc):
            doc = getattr(doc._, alias)
        setattr(doc._, f"{alias}_grammar", self)
        super().__init__(doc)
        self.smap = sort_map(smap or {})

    # Properties --------------------------------------------------------------

    @property
    def sents(self) -> DataSequence[Sent]:
        """Sentences in the document."""
        return DataSequence(s.grammar for s in self.doc.sents)

    @property
    def phrases(self) -> DataChain[DataSequence[Phrase]]:
        """Phrase in the document grouped by sentences and conjunct groups."""
        return DataChain(s.phrases for s in self.sents)

    @property
    def components(self) -> DataChain[DataChain[DataSequence[Component]]]:
        """Unique components by sentences."""
        return DataChain(s.components for s in self.sents)

    @property
    def tokens(self) -> DataSequence[Token]:
        return DataSequence(self.doc)

    # Methods -----------------------------------------------------------------

    def to_data(self) -> dict[str, Any]:
        """Dump to data dictionary."""
        key = f"{settings.spacy_alias}_grammar_data"
        grammar_data = getattr(self.doc._, key)
        for sent in self.sents:
            if (idx := sent.idx) not in grammar_data:
                grammar_data[idx] = sent.to_data()
        return self.doc.to_data()

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> Self:
        """Construct from data dictionary
        as returned by :meth:`segram.nlp.tokens.Doc.to_data`.
        """
        return cls(DocNLP.from_data(data))

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, Doc)

    def to_str(self, **kwds: Any) -> str:
        """Represent as string."""
        return " ".join(s.to_str(**kwds) for s in self.sents)
