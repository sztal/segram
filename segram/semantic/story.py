from typing import Self, Any, Iterable, MutableMapping
from .abc import Semantic
from .frames import Frame, Actants
from ..grammar import Sent, Phrase


class Story(Semantic, MutableMapping):
    """Semantic story class.

    Attributes
    ----------
    phrases
        Tuple of phrases the story operates on.
    """
    __slots__ = ("_phrases", "_frames")

    def __init__(
        self,
        phrases: Iterable[Phrase] = ()
    ) -> None:
        self._phrases = tuple(phrases)
        self._frames = {
            "actants": Actants(self)
        }

    def __getitem__(self, key: str) -> Frame:
        return self._frames[key]

    def __setitem__(self, key: str, value: Frame) -> None:
        self._frames[key] = value

    def __delitem__(self, key: str) -> None:
        del self._frames[key]

    def __iter__(self) -> Iterable[str]:
        yield from self._frames

    def __len__(self) -> int:
        return len(self._frames)

    # Properties --------------------------------------------------------------

    @property
    def phrases(self) -> tuple[Phrase, ...]:
        return self._phrases
    @phrases.setter
    def _(self, phrases: Iterable[Phrase]) -> None:
        self._phrases = tuple(phrases)
        for frame in self.frames:
            frame.clear()

    @property
    def frames(self) -> tuple[Frame, ...]:
        return tuple(self._frames.values())

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

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, Story)
