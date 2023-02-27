from ...abc import Grammar

__lang__ = __name__.rsplit(".", maxsplit=2)[-2]


class EnglishGrammar(Grammar, register=__lang__):
    """Abstract base class for English grammar."""
    __slots__ = ()
