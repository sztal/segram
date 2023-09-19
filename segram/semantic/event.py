from __future__ import annotations
from typing import Self
from .abc import SemanticElement, FrameABC
from ..grammar import Phrase


class Event(SemanticElement):
    """Semantic event class."""
