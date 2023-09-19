from __future__ import annotations
from typing import Any
from types import MappingProxyType
from functools import total_ordering
from .abc import StoryABC, FrameABC, SemanticElement
from ..grammar import Sent, Phrase, Conjuncts


@total_ordering
class Frame(FrameABC):
    """Semantic frame class.

    Attributes
    ----------
    sent
        Grammar sentence.
    """
    __slots__ = ("story", "sent")

    def __init__(
        self,
        story: StoryABC,
        sent: Sent
    ) -> None:
        self.story = story
        self.sent = sent
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
    def hashdata(self) -> int:
        return (*super().hashdata, self.story, self.sent)

    # Methods -----------------------------------------------------------------

    def is_comparable_with(self, other: Any):
        return isinstance(other, Frame)

    def make_element(self, phrase: Phrase) -> SemanticElement:
        """Make semantic element from phrase."""


    # Internals ---------------------------------------------------------------

    def _find_roots(self) -> Conjuncts:
        """Find root elements.

        These are typically events, but sometimes they
        may be entities.
        """
        # group = self.sent.root.phrase.group
