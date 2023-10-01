from .grammar import GrammarNLP
from ...grammar import Doc


class DocNLP(GrammarNLP, Doc):
    """Abstract base class for document elements
    with NLP backend methods.
    """
    __slots__ = ()
