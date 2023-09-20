from typing import Any, Self, Iterable
from .abc import Semantic
from ..grammar import Sent, Phrase


class Story(Semantic):
    """Semantic story class.

    Attributes
    ----------
    phrases
        List of phrases the story operates on.
    defs
        Dictionary of definitions of semantic elements in the story.
    emap
        Map from phrases to semantic elements.
    """
    __slots__ = ("phrases", "emap")

    def __init__(
        self,
        phrases: Iterable[Phrase] = ()
    ) -> None:
        self.phrases = list(phrases)
        self.emap = {}

    # Properties --------------------------------------------------------------

    @property
    def hashdata(self) -> tuple[Any, ...]:
        return (*super().hashdata, id(self))

    # Constructors ------------------------------------------------------------

    @classmethod
    def from_sents(cls, sents: Iterable[Sent], *args: Any, **kwds: Any) -> Self:
        """Construct from sentence objects."""
        phrases = [ p for s in sents for p in s.phrases ]
        return cls(phrases, *args, **kwds)

    # Methods -----------------------------------------------------------------

    def copy(self, **kwds: Any) -> Self:
        return self.__class__(**{ **self.data, **kwds })

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, Story)
