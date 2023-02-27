from ...grammar import Grammar


class GrammarNLP(Grammar, register="nlp"):
    """Abstract base class for grammar NLP backends."""
    __slots__ = ()
