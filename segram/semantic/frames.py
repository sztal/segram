from __future__ import annotations
from typing import Self, Any, Sequence, Callable
from abc import abstractmethod
from .abc import Semantic
from ..grammar import Phrase, NounPhrase
from ..symbols import Dep
from ..utils.types import Matcher


class Frame(Semantic, Sequence):
    """Semantic frame class.

    Semantic frames groups phrases from a given :class:`segram.semantic.Story`
    instance according to some selection criteria.

    Attributes
    ----------
    story
        A Story the frame belongs to.
    matcher
        Arbitrary callable implementing filtering function.
    phrases
        Phrases matching the criteria.
    """
    __slots__ = ("_story", "matcher", "_phrases")

    def __init__(
        self,
        story: "Story"
    ) -> None:
        self._story = story
        self.matcher = Matcher(self.is_match)
        self._phrases = ()

    def __len__(self) -> int:
        return len(self.phrases)

    def __getitem__(self, idx: int | slice) -> Phrase | tuple[Phrase, ...]:
        return self.phrases[idx]

    def __and__(self, other: Frame) -> Frame:
        if isinstance(other, Frame | Callable):
            new = self.copy()
            func = other.matcher if isinstance(other, Frame) else other
            new.matcher &= func
            return new
        return NotImplemented

    def __rand__(self, other: Frame) -> Frame:
        return self & other

    def __or__(self, other: Frame) -> Frame:
        if isinstance(other, Frame | Callable):
            new = self.copy()
            func = other.matcher if isinstance(other, Frame) else other
            new.matcher |= func
            return new
        return NotImplemented

    def __ror__(self, other: Frame) -> Frame:
        return self | other

    # Properties --------------------------------------------------------------

    @property
    def story(self) -> "Story":
        return self._story

    @property
    def phrases(self) -> tuple[Phrase, ...]:
        if not self._phrases:
            self._phrases = \
                tuple(p for p in self.story.phrases if self.match(p))
        return self._phrases

    # Methods -----------------------------------------------------------------

    @abstractmethod
    def is_match(self, phrase: Phrase) -> bool:
        """Does ``phrase`` match the criteria of the frame."""

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, Frame)

    def match(self, phrase: Phrase) -> bool:
        """Match phrase against the selection criteria."""
        return self.matcher(phrase)

    def clear(self) -> None:
        """Clear phrase sequence."""
        self._phrases = ()

    def copy(self, **kwds: Any) -> Self:
        return self.__class__(self.story, **kwds)


class Actants(Frame):
    """Semantic frame of actants."""
    __slots__ = ()

    def is_match(self, phrase: Phrase) -> bool:
        match phrase:
            case NounPhrase(dep=dep):
                return not dep & (Dep.nmod | Dep.desc | Dep.appos)
            case _:
                return False
