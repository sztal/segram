from __future__ import annotations
from typing import Any, Optional, Callable, Sequence
from .abc import Semantic
from ..grammar import Phrase, NounPhrase
from ..symbols import Dep


class Frame(Semantic, Sequence):
    """Semantic frame class.

    Semantic frames groups phrases from a given :class:`segram.semantic.Story`
    instance according to some selection criteria.

    Attributes
    ----------
    name
        Name of the frame.
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
        story: "Story",
        matcher: Optional[Callable[[Any], bool]] = None
    ) -> None:
        self._story = story
        self.matcher = matcher
        self._phrases = ()

    def __len__(self) -> int:
        return len(self.phrases)

    def __getitem__(self, idx: int | slice) -> Phrase | tuple[Phrase, ...]:
        return self.phrases[idx]

    def __and__(self, other: Frame) -> Frame:
        if isinstance(other, Frame):
            return Frame(self.story, lambda p: self.match(p) and other.match(p))
        return NotImplemented

    def __rand__(self, other: Frame) -> Frame:
        return self & other

    def __or__(self, other: Frame) -> Frame:
        if isinstance(other, Frame):
            return Frame(self.story, lambda p: self.match(p) or other.match(p))
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

    def match(self, phrase: Phrase) -> bool:
        """Match phrase against the selection criteria."""
        if self.matcher is not None:
            return self.matcher(phrase)
        cn = self.__class__.__name__
        raise NotImplementedError(f"'{cn}' does not implement a default matching function")

    def clear(self) -> None:
        """Clear phrase sequence."""
        self._phrases = ()


class Actants(Matcher):
    """Semantic frame of actants."""

    @staticmethod
    def match(obj: Phrase) -> bool:
        """Check if phrase is an actant."""
        match obj:
            case NounPhrase(dep=dep):
                return not dep & (Dep.nmod | Dep.desc | Dep.appos)
            case _:
                return False
