from __future__ import annotations
from typing import Self, Any, Iterable, ClassVar
from .abc import SemanticElement, FrameABC
from ..grammar import Phrase, VerbPhrase
from ..utils.types import ChainGroup, Group
from ..nlp.tokens import TokenABC
from ..symbols import Role, Dep


class Action(SemanticElement):
    """Semantic action class.

    Attributes
    ----------
    xcomp
        Open clausal complements.
        These are not by themselves proper semantic elements
        and therefore they are represented as
        :class:`segram.grammar.Phrase` objects.
    """
    alias: ClassVar[str] = "Action"
    __parts__ = ("xcomp",)
    __slots__ = (*__parts__,)

    def __init__(
        self,
        *args: Any,
        xcomp: ChainGroup[Group[Phrase]] = (),
        **kwds: Any
    ) -> None:
        super().__init__(*args, **kwds)
        self.xcomp = xcomp

    # Constructors ------------------------------------------------------------

    @classmethod
    def from_phrase(cls, phrase: Phrase, frame: FrameABC) -> Iterable[Self]:
        def _iter_xcomps(phrase, chain=()):
            if not phrase.xcomp:
                yield tuple(chain)
            else:
                for xcomp in phrase.xcomp:
                    new_chain = list(chain)
                    new_chain.append(xcomp)
                    yield from _iter_xcomps(xcomp, chain=new_chain)

        kwds = dict(phrase=phrase, frame=frame)
        chains = tuple(_iter_xcomps(phrase))
        if not chains:
            yield cls(**kwds)
        else:
            for chain in chains:
                yield cls(**kwds, xcomp=ChainGroup([chain]))

    # Methods -----------------------------------------------------------------

    @classmethod
    def based_on(cls, phrase: Phrase) -> bool:
        return isinstance(phrase, VerbPhrase) \
            and phrase.head.dep & ~Dep.xcomp

    def iter_token_roles(self) -> tuple[TokenABC, Role | None]:
        """Iterate over token-role pairs."""
        yield from self._iter_token_roles(
            super().iter_token_roles(),
            self.head.iter_token_roles(),
            *(d.iter_token_roles() for d in self.phrase.desc)
        )
