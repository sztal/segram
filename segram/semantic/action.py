from __future__ import annotations
from typing import Self, Any, Iterable
from .abc import SemanticElement, FrameABC
from ..grammar import Phrase, VerbPhrase
from ..utils.types import ChainGroup, Group


class Action(SemanticElement):
    """Semantic action class."""
    __parts__ = ("xcomp",)
    __slots__ = (*__parts__,)

    def __init__(
        self,
        *args: Any,
        xcomp: ChainGroup[Group[Phrase]] = (),
        **kwds: Any
    ) -> None:
        super().__init__(*args, **kwds)
        self.xcomp = ChainGroup([xcomp])

    # Constructors ------------------------------------------------------------

    @classmethod
    def from_phrase(cls, phrase: Phrase) -> Iterable[Self]:
        if not isinstance(phrase, VerbPhrase):
            return
        def _iter_xcomps(phrase, chain=()):
            if not phrase.xcomp:
                yield tuple(chain)
            else:
                for xcomp in phrase.xcomp:
                    new_chain = list(chain)
                    new_chain.append(xcomp)
                    yield from _iter_xcomps(xcomp, chain=new_chain)

        kwds = dict(phrase=phrase, frame=None)
        chains = tuple(_iter_xcomps(phrase))
        if not chains:
            yield cls(**kwds)
        else:
            for chain in chains:
                yield cls(**kwds, xcomp=chain)
