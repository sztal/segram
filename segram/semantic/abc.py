from __future__ import annotations
from typing import Any
from ..abc import SegramABC


class Semantic(SegramABC):
    """Abstract base class for semantic classes."""
    __slots__ = ()

    def __eq__(self, other: Any) -> bool:
        if not self.is_comparable_with(other):
            return NotImplemented
        return all(
            getattr(self, name) == getattr(other, name)
            for name in self.slot_names
        )
