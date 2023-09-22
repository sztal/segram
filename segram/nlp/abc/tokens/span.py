from __future__ import annotations
from typing import Iterable
from abc import abstractmethod
from ..base import NLPToken, attr
from .token import TokenABC
from ....utils.diff import iter_diffs, equal, IDiffType


class SpanABC(NLPToken):
    """Span abstract base class."""
    __slots__ = ()

    def __hash__(self) -> int:
        return hash((hash(self.doc), self.start, self.end))

    def __eq__(self, other: SpanABC) -> bool:
        if (res := super().__eq__(other)) is NotImplemented:
            return res
        return res and (self.start, self.end) == (other.start, other.end)

    def __lt__(self, other: NLPToken) -> bool:
        """Is ``self`` earlier in the document than ``other``."""
        if self.is_comparable_with(other):
            return (self.start, self.end) < (other.start, other.end)
        return NotImplemented

    @abstractmethod
    def __iter__(self) -> Iterable[TokenABC]:
        """Iterate over tokens."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of tokens."""

    @abstractmethod
    def __getitem__(self, idx: int | slice) -> TokenABC:
        """Return specific token or (sub)span by indexing."""

    @abstractmethod
    def __contains__(self, tok: TokenABC) -> bool:
        """Check if ``self`` contains ``tok``."""

    @attr
    @property
    @abstractmethod
    def start(self) -> int:
        """Sentence start index."""

    @attr
    @property
    @abstractmethod
    def end(self) -> int:
        "Sentence stop index."


# Comparison functions for testing --------------------------------------------

@equal.register
def _(obj: SpanABC, other: SpanABC, *, strict: bool = True) -> bool:
    return equal(obj.doc, other.doc, strict=strict) \
        and (obj.start, obj.end) == (other.start, other.end)
@iter_diffs.register
def _(obj: SpanABC, other: SpanABC, *, strict: bool = True) -> IDiffType:
    if not equal(obj, other, strict=strict):
        yield "SPAN", obj, other
