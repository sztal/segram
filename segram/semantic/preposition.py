from __future__ import annotations
from typing import ClassVar, Any, Iterable
from .abc import SemanticElement, FrameABC
from ..grammar import Phrase, PrepPhrase, Conjuncts
from ..utils.types import ChainGroup, Group
from ..nlp.tokens import TokenABC
from ..symbols import Role


class Preposition(SemanticElement):
    """Semantic preposition class.

    Attributes
    ----------
    """
    alias: ClassVar[str] = "Prep"
    __parts__ = ("pobj",)
    __slots__ = (*__parts__,)

    def __init__(
        self,
        *args: Any,
        pobj: ChainGroup[Group[SemanticElement]] = (),
        **kwds: Any
    ) -> None:
        super().__init__(*args, **kwds)
        self.pobj = pobj

    # Methods -----------------------------------------------------------------

    @classmethod
    def iter_phrase_data(cls, phrase: Phrase) -> Iterable[dict[str, Any]]:
        yield { "pobj": Conjuncts.get_chain(phrase.pobj) }

    @classmethod
    def based_on(cls, phrase: Phrase) -> bool:
        return isinstance(phrase, PrepPhrase)
