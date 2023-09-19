from __future__ import annotations
from typing import Self, Any, Iterable
from .abc import SemanticElement, FrameABC
from ..grammar import Phrase, NounPhrase, Conjuncts
from ..utils.types import ChainGroup, Group
from ..nlp.tokens import TokenABC
from ..symbols import Role


class Event(SemanticElement):
    """Semantic event class."""
