from typing import Any, Self, Optional, Mapping, Iterable
from .abc import Semantic, SemanticElement
from .elements import Actant
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
    __slots__ = ("phrases", "defs", "emap")

    def __init__(
        self,
        phrases: Iterable[Phrase] = (),
        defs: Optional[Mapping[str, SemanticElement]] = None
    ) -> None:
        self.phrases = list(phrases)
        self.defs = {
            "actants": Actant,
            **(defs or {})
        }
        self.emap = {}

    def __getattr__(self, attr: str) -> Any:
        if attr in self.defs:
            elem = self.defs[attr]
            return (e for p in self.phrases for e in elem.from_phrase(self, p))
        cn = self.__class__.__name__
        raise AttributeError(f"'{cn}' object has no attribute '{attr}'")

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
