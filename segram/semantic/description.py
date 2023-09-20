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
    __parts__ = ("cdesc",)
    __slots__ = (*__parts__,)

    def __init__(
        self,
        *args: Any,
        cdesc: ChainGroup[Group[SemanticElement]] = (),
        **kwds: Any
    ) -> None:
        super().__init__(*args, **kwds)
        self.cdesc = cdesc

    # Methods -----------------------------------------------------------------

    @classmethod
    def iter_phrase_data(cls, phrase: Phrase) -> Iterable[dict[str, Any]]:
        yield { "cdesc": Conjuncts.get_chain(phrase.cdesc) }

    @classmethod
    def based_on(cls, phrase: Phrase) -> bool:
        return isinstance(phrase, DescPhrase)
