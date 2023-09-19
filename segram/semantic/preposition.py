from __future__ import annotations
from typing import Self, Any, Iterable
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

    # Constructors ------------------------------------------------------------

    @classmethod
    def from_phrase(cls, phrase: Phrase) -> Iterable[Self]:
        if not isinstance(phrase, PrepPhrase):
            return
