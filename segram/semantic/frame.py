from __future__ import annotations
from typing import Any
from types import MappingProxyType
from functools import total_ordering
from .abc import StoryABC, FrameABC
from ..grammar import Sent, Conjuncts


@total_ordering
class Frame(FrameABC):
    """Semantic frame class.

    Attributes
    ----------
    sent
        Grammar sentence.
    """
    __slots__ = ("_story", "_sent")

    def __init__(
        self,
        story: StoryABC,
        sent: Sent
    ) -> None:
        self._story = story
        self._sent = sent
        self.story.pmap.maps.append(MappingProxyType(self.sent.pmap))

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Frame) -> bool:
        if self.is_comparable_with(other):
            return self.sent == other.sent
        return NotImplemented

    def __lt__(self, other: Frame) -> bool:
        if self.is_comparable_with(other):
            return self.sent < other.sent
        return NotImplemented

    # Properties --------------------------------------------------------------

    @property
    def sent(self) -> Sent:
        return self._sent

    @property
    def story(self) -> StoryABC:
        return self._story

    @property
    def hashdata(self) -> int:
        return (*super().hashdata, self.story, self.sent)

    # Methods -----------------------------------------------------------------

    def is_comparable_with(self, other: Any):
        return isinstance(other, Frame)

    # Internals ---------------------------------------------------------------

    def _find_roots(self) -> Conjuncts:
        """Find root elements.

        These are typically events, but sometimes they
        may be entities.
        """
        raise NotImplementedError
