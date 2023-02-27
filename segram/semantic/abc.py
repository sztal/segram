from __future__ import annotations
from typing import ClassVar, Type
from ..abc import SegramWithDocABC
from ..utils.types import Namespace


class SemanticNamespace(Namespace):
    Story: Type["Story"]
    Frame: Type["Frame"]
    FElem: Type["FrameElement"]
    Actor: Type["Actor"]
    Event: Type["Event"]
    Description: Type["Description"]


class Semantic(SegramWithDocABC):
    """Abstract base class for semantic classses."""
    __slots__ = ()
    types: ClassVar[SemanticNamespace] = SemanticNamespace()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.types[cls.__name__] = cls
