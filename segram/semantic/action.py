from __future__ import annotations
from typing import Self, Any
from .abc import SemanticElement, FrameABC
from ..grammar import Phrase


class Action(SemanticElement):
    """Semantic action class."""
    __slots__ = ("desc", "xcomp")

    def __init__(
        self,
        *args: Any,
    )
