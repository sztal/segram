from __future__ import annotations
from typing import Any, ClassVar, Self, Iterable
from abc import abstractmethod
from ..abc import SegramWithDocABC
from ..grammar import Phrase, Component, Conjuncts
from ..utils.types import Namespace
from ..nlp.tokens import DocABC, TokenABC
from ..symbols import Role


class SemanticNamespace(Namespace):
    Story: type["Story"]
    Frame: type["Frame"]
    Elem: type["SemanticElement"]
    Actant: type["Actant"]
    Action: type["Action"]
    Prep: type["Preposition"]
    Desc: type["Description"]
    Event: type["Event"]


class Semantic(SegramWithDocABC):
    """Abstract base class for semantic classses."""
    __slots__ = ()
    types: ClassVar[SemanticNamespace] = SemanticNamespace()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if (alias := getattr(cls, "alias", None)):
            cls.types[alias] = cls


class FrameABC(Semantic):
    """Abstract base class for semantic frame."""
    __slots__ = ()


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
    alias: ClassVar[str] = "Elem"
    __parts__ = ()
    __slots__ = ("phrase", "frame", "parents", *__parts__)
    part_names: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        phrase: Phrase,
        frame: FrameABC,
        *,
        parents: Conjuncts[SemanticElement] = ()
    ) -> None:
        self.phrase = phrase
        self.frame = frame
        self.parents = parents  # TODO: implement this

    def __new__(cls, *args: Any, **kwds: Any) -> None:
        obj = super().__new__(cls)
        obj.__init__(*args, **kwds)
        if (cur := obj.story.emap.get(obj.idx)):
            cur.__init__(**obj.data)
            return cur
        obj.story.emap[obj.idx] = obj
        return obj

    def __repr__(self) -> str:
        return self.to_str(color=True)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.init_class_attrs({
            "__parts__": "part_names"
        }, check_slots=True)

    # Properties --------------------------------------------------------------

    @property
    def story(self) -> "Story":
        return self.frame.story

    @property
    def doc(self) -> DocABC:
        return self.story.doc

    @property
    def idx(self) -> int:
        return self.phrase.idx

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

    @property
    def is_lead(self) -> bool:
        """Check if lead element of a conjoined group."""
        return self.phrase.is_lead

    @property
    def group(self) -> Conjuncts[SemanticElement]:
        """Get conjoined group."""
        raise NotImplementedError

    # Constructors ------------------------------------------------------------

    @classmethod
    @abstractmethod
    def based_on(cls, phrase: Phrase) -> bool:
        """Check if element can be based on phrase."""

    @classmethod
    @abstractmethod
    def iter_phrase_data(cls, phrase: Phrase) -> Iterable[dict[str, Any]]:
        """Get phrase data needed for initializing element.

        Yields
        ------
        data
            One data dictionary per element originiating from ``phrase``.
        """

    # Methods -----------------------------------------------------------------

    @classmethod
    def from_phrase(cls, phrase: Phrase, frame: FrameABC) -> Iterable[Self] | None:
        """Construct from phrase.

        The method yields one or more elements from a single
        appropriate phrase. Nothing is returned if the phrase
        is not a basis of any semantic elements of the given type.
        """
        for data in cls.iter_phrase_data(phrase):
            yield cls(phrase, frame, **data)

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, SemanticElement)

    def iter_token_roles(self) -> tuple[TokenABC, Role | None]:
        """Iterate over token-role pairs."""
        def _iter():
            seen = set()
            for name in self.part_names:
                parts = getattr(self, name)
                for part in reversed(parts):
                    for tok, role in part.iter_token_roles():
                        if tok in seen:
                            continue
                        seen.add(tok)
                        yield tok, role
        yield from sorted(_iter(), key=lambda x: x[0])

    def to_str(self, *, color: bool = False, **kwds: Any) -> str:
        """Represent as string."""
        return " ".join(
            t.to_str(color=color, role=r, **kwds)
            for t, r in self.iter_token_roles()
        )

    # Internals ---------------------------------------------------------------

    def _iter_token_roles(self, *tokroles) -> tuple[TokenABC | None]:
        seen = set()
        show = []
        for tr in tokroles:
            for tok, role in tr:
                if tok in seen:
                    continue
                seen.add(tok)
                show.append((tok, role))
        yield from sorted(show, key=lambda x: x[0])
