from __future__ import annotations
from typing import Self, Any, Iterable
from .abc import SemanticElement, FrameABC
from ..grammar import Phrase, NounPhrase, Conjuncts
from ..utils.types import ChainGroup, Group
from ..nlp.tokens import TokenABC
from ..symbols import Role


class Actant(SemanticElement):
    """Semantic actant class.

    Attributes
    ----------
    relcl
        Relative clauses. These are proper semantic elements
        and therefore are represented as
        :class:`segram.semantic.Event`.
    """
    __parts__ = ("relcl",)
    __slots__ = (*__parts__,)

    def __init__(
        self,
        *args: Any,
        relcl: ChainGroup[Group[SemanticElement]] = (),
        **kwds: Any
    ) -> None:
        super().__init__(*args, **kwds)
        self.relcl = relcl

    # Constructors ------------------------------------------------------------

    @classmethod
    def from_phrase(cls, phrase: Phrase) -> Iterable[Self]:
        if not isinstance(phrase, NounPhrase):
            return
        kwds = dict(frame=None, relcl=Conjuncts.get_chain(phrase.relcl))
        yield cls(phrase, **kwds)

    # Methods -----------------------------------------------------------------

    def iter_token_roles(self) -> tuple[TokenABC, Role | None]:
        """Iterate over token-role pairs."""
        yield from self._iter_token_roles(
            super().iter_token_roles(),
            self.phrase.iter_token_roles()
        )
