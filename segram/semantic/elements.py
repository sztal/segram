from .abc import SemanticElement
from ..grammar import Phrase, NounPhrase


class Actant(SemanticElement):
    """Actant semantic element."""

    # Methods -----------------------------------------------------------------

    @classmethod
    def is_anchor(cls, phrase: Phrase) -> bool:
        pass
