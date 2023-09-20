from .abc import SemanticElement
from ..grammar import Phrase, NounPhrase
from ..symbols import Dep


class Actant(SemanticElement):
    """Actant semantic element."""
    __slots__ = ()

    # Methods -----------------------------------------------------------------

    @classmethod
    def matches(cls, phrase: Phrase) -> bool:
        match phrase:
            case NounPhrase(dep=dep):
                return not dep & (Dep.nmod | Dep.desc | Dep.appos)
            case _:
                return False

    @classmethod
    def ends(cls, phrase: Phrase) -> bool:
        return phrase.dep & Dep.relcl

    @classmethod
    def extend(cls, phrase: Phrase) -> bool:
        yield from phrase.children
