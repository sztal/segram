from __future__ import annotations
from typing import ClassVar, Type, Self
from abc import abstractmethod
from ..abc import SegramWithDocABC
from ..grammar import Phrase, Component, Conjuncts
from ..utils.types import Namespace


class SemanticNamespace(Namespace):
    Story: Type["Story"]
    Frame: Type["Frame"]
    FElem: Type["FrameElement"]
    Actor: Type["Actor"]
    Event: Type["Event"]
    Description: Type["Description"]
    Complement: Type["Complement"]


class Semantic(SegramWithDocABC):
    """Abstract base class for semantic classses."""
    __slots__ = ()
    types: ClassVar[SemanticNamespace] = SemanticNamespace()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.types[cls.__name__] = cls


class StoryABC(Semantic):
    """Abstract base class for semantic story."""


class FrameABC(Semantic):
    """Abstract base class for semantic frame."""


class SemanticElement(Semantic):
    """Abstract base class for semantic elements.

    Attributes
    ----------
    phrase
        Grammar phrase corresponding to the semantic element.
    frame
        Controlling semantic frame.
    parents
        Parent semantic elements.
    """
    __slots__ = ("phrase", "frame", "parents")

    def __init__(
        self,
        phrase: Phrase,
        frame: FrameABC,
        *,
        parents: Conjuncts[SemanticElement] = ()
    ) -> None:
        self.phrase = phrase
        self.frame = frame
        self.parents = Conjuncts(parents) \
            if not isinstance(parents, Conjuncts) else parents

    # Properties --------------------------------------------------------------

    @property
    def story(self) -> StoryABC:
        return self.frame.story

    @property
    def head(self) -> Component:
        """Head components of ``self.phrase``."""
        return self.phrase.head

    @property
    def depth(self) -> int:
        """Phrasal depth."""
        # TODO: rethink in terms of frame tree
        return self.phrase.depth

    @property
    def is_root(self) -> bool:
        """Check if root element with no parents."""
        return not self.parents

    # Constructors ------------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_phrase(cls, phrase: Phrase) -> Self:
        """Construct from phrase."""
