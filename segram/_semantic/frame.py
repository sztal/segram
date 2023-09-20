from __future__ import annotations
from typing import Any, Iterable, ClassVar
from types import MappingProxyType
from functools import total_ordering
from .abc import FrameABC, SemanticElement
from .story import Story
from ..grammar import Sent, Phrase, Conjuncts
from ..nlp.tokens import DocABC


@total_ordering
class Frame(FrameABC):
    """Semantic frame class.

    Attributes
    ----------
    sent
        Grammar sentence.
    """
    alias: ClassVar[str] = "Frame"
    __slots__ = ("story", "sent")

    def __init__(
        self,
        story: Story,
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
    def doc(self) -> DocABC:
        return self.story.doc

    @property
    def hashdata(self) -> int:
        return (*super().hashdata, self.story, self.sent)

    # Methods -----------------------------------------------------------------

    def is_comparable_with(self, other: Any):
        return isinstance(other, Frame)

    def iter_elements(self, phrase: Phrase) -> Iterable[SemanticElement]:
        """Iterate over semantic element(s) from phrase."""
        for typ in self.types.values():
            if not issubclass(typ, SemanticElement) or typ.__abstractmethods__:
                continue
            if typ.based_on(phrase):
                yield from typ.from_phrase(phrase, self)

    # Internals ---------------------------------------------------------------

    def _find_roots(self) -> Conjuncts:
        """Find root elements.

        These are typically events, but sometimes they
        may be entities.
        """
        # group = self.sent.root.phrase.group
