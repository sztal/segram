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
    __parts__ = ()
    __slots__ = (*__parts__,)

    # Properties --------------------------------------------------------------

    @property
    def pobj(self) -> Iterable[SemanticElement]:
        """Preposition object child elements."""
        for phrase in self.phrase.pobj:
            yield from self.frame.make_element(phrase)

    # Methods -----------------------------------------------------------------

    @classmethod
    def based_on(cls, phrase: Phrase) -> bool:
        return isinstance(phrase, PrepPhrase)
