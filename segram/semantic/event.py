from __future__ import annotations
from typing import Self, Any, Iterable
from .abc import SemanticElement, FrameABC
from ..grammar import Phrase, VerbPhrase, Conjuncts
from ..utils.types import ChainGroup, Group
from ..nlp.tokens import TokenABC
from ..symbols import Role


class Event(SemanticElement):
    """Semantic event class."""
    __parts__ = ("subj", "action", "dobj", "iobj", "event")
    __slots__ = (*__parts__,)

    def __init__(
        self,
        *args: Any,
        subj: ChainGroup[Group[SemanticElement]] = (),
        action: ChainGroup[Group[SemanticElement]] = (),
        dobj: ChainGroup[Group[SemanticElement]] = (),
        iobj: ChainGroup[Group[SemanticElement]] = (),
        event: ChainGroup[Group[SemanticElement]] = (),
        **kwds: Any
    ) -> None:
        super().__init__(*args, **kwds)
        self.subj = subj
        self.action = action
        self.dobj = dobj
        self.iobj = iobj
        self.event = event

    # Constructors ------------------------------------------------------------

    @classmethod
    def from_phrase(cls, phrase: Phrase) -> Iterable[Self]:
        if not isinstance(phrase, VerbPhrase):
            return

