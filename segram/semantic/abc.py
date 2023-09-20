from typing import Iterable, Self
from abc import abstractmethod
from itertools import islice
from ..abc import SegramABC
from ..grammar import Phrase


class Semantic(SegramABC):
    """Abstract base class for semantic classes."""
    __slots__ = ()


class SemanticElement(Semantic):
    """Semantic element class."""

    # Constructors ------------------------------------------------------------

    @classmethod
    def from_phrase(cls, phrase: Phrase) -> Iterable[Self]:
        """Construct from phrase."""
        if cls.is_anchor(phrase):
            for another in cls.stream(phrase):
                if cls.is_terminus(another):
                    pass

    # Methods -----------------------------------------------------------------

    @classmethod
    @abstractmethod
    def is_anchor(cls, phrase: Phrase) -> bool:
        """Is phrase the anchor of an element."""

    @classmethod
    @abstractmethod
    def is_terminus(cls, phrase: Phrase) -> bool:
        """Is phrase the terminus of an element."""

    @classmethod
    @abstractmethod
    def stream(cls, phrase: Phrase) -> bool:
        """Stream intermediate phrases from an anchor."""

    @classmethod
    def iter_subtree(cls, phrase: Phrase) -> Iterable[Phrase]:
        """Iterate over the proper subtree of phrase."""
        yield from islice(phrase.subtree, 1, None)

    @classmethod
    def iter_suptree(cls, phrase: Phrase) -> Iterable[Phrase]:
        """Iterate over the proper supertree of phrase."""
        yield from islice(phrase.suptree, 1, None)
