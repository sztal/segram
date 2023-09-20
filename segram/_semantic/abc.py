from __future__ import annotations
from typing import Any, Optional, ClassVar, Self, Iterable
from abc import abstractmethod
from ..abc import SegramWithDocABC
from ..grammar import Phrase, Component, Conjuncts
from ..utils.types import Namespace, ChainGroup, Group
from ..nlp.tokens import DocABC, TokenABC
from ..symbols import Role, Dep


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
    """
    alias: ClassVar[str] = "Elem"
    __parts__ = ("prep", "desc")
    __slots__ = ("phrase", "frame")
    part_names: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        phrase: Phrase,
        frame: FrameABC,
    ) -> None:
        self.phrase = phrase
        self.frame = frame

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
        }, check_slots=False)

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
        return self.depth == 0

    @property
    def is_lead(self) -> bool:
        """Check if lead element of a conjoined group."""
        return self.phrase.is_lead

    @property
    def group(self) -> Conjuncts[SemanticElement]:
        """Get conjoined group."""
        raise NotImplementedError

    @property
    def children(self) -> ChainGroup[Group[SemanticElement]]:
        """Child elements."""
        return tuple(self.iter_children())

    @property
    def actant(self) -> tuple[SemanticElement]:
        """Actant child elements."""
        return tuple(self.iter_children("Actant"))

    @property
    def action(self) -> tuple[SemanticElement]:
        """Action child elements."""
        return tuple(self.iter_children("Action"))

    @property
    def prep(self) -> tuple[SemanticElement]:
        """Preposition child elements."""
        return tuple(self.iter_children("Prep"))

    @property
    def desc(self) -> tuple[SemanticElement]:
        """Description child elements."""
        return tuple(self.iter_children("Desc", dep=Dep.misc))

    # Constructors ------------------------------------------------------------

    @classmethod
    def from_phrase(cls, phrase: Phrase, frame: FrameABC) -> Iterable[Self] | None:
        """Construct from phrase.

        The method yields one or more elements from a single
        appropriate phrase. Nothing is returned if the phrase
        is not a basis of any semantic elements of the given type.
        """
        for data in cls.iter_phrase_data(phrase):
            base = cls._get_base_phrase_data(phrase)
            yield cls(phrase, frame, **base, **data)

    # Methods -----------------------------------------------------------------

    @classmethod
    @abstractmethod
    def based_on(cls, phrase: Phrase) -> bool:
        """Check if element can be based on phrase."""

    @classmethod
    def iter_phrase_data(cls, phrase: Phrase) -> Iterable[dict[str, Any]]:
        """Get phrase data needed for initializing element.

        Yields
        ------
        data
            One data dictionary per element originiating from ``phrase``.
        """
        yield { **cls._get_base_phrase_data(phrase) }

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, SemanticElement)

    def iter_token_roles(self) -> tuple[TokenABC, Role | None]:
        """Iterate over token-role pairs.

        Parameters
        ----------
        **kwds
            Passed to :meth:`_iter_token_roles`.
        """
        def _iter():
            for name in self.part_names:
                parts = getattr(self, name)
                for part in reversed(parts):
                    yield from part.iter_token_roles()
            yield from self.phrase.head.iter_token_roles()
        yield from self._iter_token_roles(_iter())

    def to_str(self, *, color: bool = False, **kwds: Any) -> str:
        """Represent as string."""
        return " ".join(
            t.to_str(color=color, role=r, **kwds)
            for t, r in self.iter_token_roles()
        )

    def iter_children(
        self,
        typ: Optional[str | type] = None,
        dep: Optional[Dep] = None
    ) -> Iterable[SemanticElement]:
        """Iterate over child elements.

        Parameters
        ----------
        typ
            Type of element to consider.
            Can be passed as ``type`` object or a string alias.
            Iterate over all children when ``None``.
        dep
            Alternaive selection condition based
            on phrasal dependency tag value.
        """
        # pylint: disable=too-many-boolean-expressions
        if typ and isinstance(typ, str):
            typ = self.frame.types[typ]
        for child in self.phrase.children:
            if (not typ and not dep) \
            or (typ and typ.based_on(child)) \
            or (dep and child.dep & dep):
                yield from self.frame.iter_elements(child)

    # Internals ---------------------------------------------------------------

    def _iter_token_roles(
        self,
        *tokroles,
        role: Optional[Role] = None
    ) -> tuple[TokenABC | None]:
        seen = set()
        show = []
        for tr in tokroles:
            for t, r in tr:
                if t in seen:
                    continue
                seen.add(t)
                show.append((t, r))
        tr = sorted(show, key=lambda x: x[0])
        if role is None:
            yield from tr
        else:
            for t, _ in tr:
                yield t, role

    @classmethod
    def _get_base_phrase_data(cls, phrase: Phrase) -> dict[str, Any]:
        # pylint: disable=unused-argument
        return {}
