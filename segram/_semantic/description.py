from __future__ import annotations
from typing import ClassVar, Any, Iterable
from .abc import SemanticElement, FrameABC
from ..grammar import Phrase, DescPhrase, Conjuncts
from ..utils.types import ChainGroup, Group


class Description(SemanticElement):
    """Semantic preposition class.

    Attributes
    ----------
    """
    alias: ClassVar[str] = "Desc"
    __parts__ = ()
    __slots__ = (*__parts__,)

    # Properties --------------------------------------------------------------

    @property
    def cdesc(self) -> Iterable[SemanticElement]:
        """Description complement child elements."""
        for phrase in self.phrase.cdesc:
            yield from self.frame.make_element(phrase)

    # Methods -----------------------------------------------------------------

    @classmethod
    def based_on(cls, phrase: Phrase) -> bool:
        return isinstance(phrase, DescPhrase)
