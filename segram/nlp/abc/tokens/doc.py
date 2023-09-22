from __future__ import annotations
from typing import Any, Iterable
from abc import abstractmethod
from ..base import NLPToken, attr
from .token import TokenABC
from .span import SpanABC
from ....utils.diff import iter_diffs, equal, IDiffType


class DocABC(NLPToken):
    """Document abstract base class."""
    __slots__ = ()

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: Any) -> bool:
        return self is other

    # Abstract methods --------------------------------------------------------

    @abstractmethod
    def __iter__(self) -> Iterable[TokenABC]:
        """Iterate over tokens."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of tokens."""

    @abstractmethod
    def __getitem__(self, idx: int | slice) -> TokenABC:
        """Return specific token or span by indexing."""

    @abstractmethod
    def __contains__(self, tok: TokenABC) -> bool:
        """Check if ``self`` contains ``tok``."""

    @attr
    @property
    @abstractmethod
    def lang(self) -> str:
        """Language code."""

    @property
    @abstractmethod
    def sents(self) -> Iterable[SpanABC]:
        """Iterate over sentences."""

    # Properties --------------------------------------------------------------

    @property
    def id(self) -> int:
        """Hash id of the document tokenization."""
        return hash(t.attrs for t in self)

    # Methods -----------------------------------------------------------------

    @abstractmethod
    def copy(self) -> DocABC:
        """Return copy of the self."""


# Comparison functions for testing --------------------------------------------

@equal.register
def _(obj: DocABC, other: DocABC, *, strict: bool = True) -> bool:
    return ((strict and obj == other) or (not strict and obj.id == other.id))
@iter_diffs.register
def _(obj: DocABC, other: DocABC, *, strict: bool = True) -> IDiffType:
    if not equal(obj, other, strict=strict):
        msg = "DOCUMENT CONTENT"
        if obj.id == other.id:
            msg = "DOCUMENT TYPE"
        yield msg, obj, other
