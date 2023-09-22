from typing import Any, Optional
from .grammar import EnglishGrammar
from ... import Component
from ... import Verb, Noun
from ... import Prep, Desc
from ....nlp.abc import TokenABC


class EnglishComponent(EnglishGrammar, Component):
    """English grammar component."""
    __slots__ = ()


class EnglishVerb(EnglishComponent, Verb):
    """English verb components.

    Attributes
    ----------
    part
        Verb particle.
    aux
        Auxiliary verbs.
    expl
        Expletive token in expletive constructions
        (e.g. 'there is').
    """
    __tokens__ = ("part", "aux", "expl")
    __slots__ = (*__tokens__,)

    def __init__(
        self,
        *args: Any,
        part: Optional[TokenABC] = None,
        aux: tuple[TokenABC, ...] = (),
        expl: Optional[TokenABC] = None,
        **kwds: Any
    ) -> None:
        super().__init__(*args, **kwds)
        self.part = part
        self.aux = tuple(aux)
        self.expl = expl


class EnglishNoun(EnglishComponent, Noun):
    """Abstract base class for English noun components.

    Attributes
    ----------
    det
        Determiner token.
    """
    __tokens__ = ("det",)
    __slots__ = (*__tokens__,)

    def __init__(
        self,
        *args: Any,
        det: Optional[TokenABC] = None,
        **kwds: Any
    ) -> None:
        super().__init__(*args, **kwds)
        self.det = det


class EnglishPrep(EnglishComponent, Prep):
    """English preposition component."""
    __slots__ = ()


class EnglishDesc(EnglishComponent, Desc):
    """English description component."""
    __tokens__ = ("det",)
    __slots__ = (*__tokens__,)

    def __init__(
        self,
        *args: Any,
        det: Optional[TokenABC] = None,
        **kwds: Any
    ) -> None:
        super().__init__(*args, **kwds)
        self.det = det
