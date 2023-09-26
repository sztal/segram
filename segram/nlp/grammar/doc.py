from .grammar import GrammarNLP
from ...grammar import Doc
from ..tokens import Doc as _Doc


class DocNLP(GrammarNLP, Doc):
    """Abstract base class for document elements
    with NLP backend methods.
    """
    __slots__ = ()

    # Methods -----------------------------------------------------------------

    @classmethod
    def from_doc(cls, doc: _Doc) -> Doc:
        """Construct from a document object."""
        doc = cls(doc)
        doc.sents   # pylint: disable=pointless-statement
        return doc
