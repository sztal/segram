from .grammar import GrammarNLP
from ...grammar import Phrase


class PhraseNLP(GrammarNLP, Phrase):
    """Abstract base class for phrase classes
    with NLP backend methods.
    """
    __slots__ = ()
