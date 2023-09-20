from __future__ import annotations
from typing import Any, Iterable, ClassVar
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
    alias: ClassVar[str] = "Actant"
    __parts__ = ()
    __slots__ = (*__parts__,)

    # Properties --------------------------------------------------------------

    @property
    def relcl(self) -> Iterable[SemanticElement]:
        """Relative clause child elements."""
        for phrase in self.phrase.relcl:
            yield from self.frame.make_element(phrase)

    # Methods -----------------------------------------------------------------

    @classmethod
    def based_on(cls, phrase: Phrase) -> bool:
        return isinstance(phrase, NounPhrase) \
            and not any(isinstance(p, NounPhrase) for p in phrase.parents)

    def iter_token_roles(self) -> tuple[TokenABC, Role | None]:
        """Iterate over token-role pairs."""
        yield from self._iter_token_roles(
            super().iter_token_roles(),
            self.phrase.iter_token_roles()
        )
