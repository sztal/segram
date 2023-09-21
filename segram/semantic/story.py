from typing import Any, Optional, Self, Iterable, Mapping, Sequence, TypedDict, Callable
from .abc import Semantic, SemanticElement
from ..grammar import Sent, Phrase


class Story(Semantic):
    """Semantic story class.

    Attributes
    ----------
    phrases
        List of phrases the story operates on.
    emap
        Map from phrases to semantic elements.
    ctx
        Context dictionary for grouping phrases.
    """
    __slots__ = ("phrases", "emap", "ctx")

    class SelectorDict(TypedDict):
        elements: Optional[Sequence[str]]
        matcher: Optional[Callable[[SemanticElement], bool]]
        context: Optional[str]

    def __init__(
        self,
        phrases: Iterable[Phrase] = ()
    ) -> None:
        self.phrases = list(phrases)
        self.emap = {}
        self.ctx = {}

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

    def select(self, *elements, spec: Mapping[str, SelectorDict]) -> Self:
        """Make new story with selected phrases
        grouped into different contexts.

        Parameters
        ----------
        *elements
            Names of element types to consider.
        spec
            Selection specification with keys providing
            names of contexts to create.
        """
