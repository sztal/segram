from __future__ import annotations
from typing import Any, ClassVar, Type, Iterator
from functools import total_ordering
from itertools import product
from abc import abstractmethod
from collections.abc import Iterable
from .abc import Semantic
from .relations import Relation
from .relations import ActorRelation, ActionRelation
from .relations import DescriptionRelation, ComplementRelation
from ..grammar import Phrase, VerbPhrase, NounPhrase, DescPhrase, PrepPhrase
from ..grammar import Component, Conjuncts
from ..symbols import Dep, Role
from ..nlp import DocABC, TokenABC


@total_ordering
class FrameElement(Semantic):
    """Abstract base class for frame elements.

    Attributes
    ----------
    frame
        Semantic frame.
    phrase
        Grammar phrase.
    """
    __parts__ = ("descriptions", "complements")
    __slots__ = ("_frame", "phrase", *__parts__)
    part_names: ClassVar[tuple[str, ...]] = ()
    Relation: Type[Relation] = Relation


    def __init__(
        self,
        frame: "Frame",
        phrase: Phrase,
        *,
        descriptions: Iterable[Phrase] = (),
        complements: Iterable[Phrase] = ()
    ) -> None:
        if phrase not in frame.sent:
            raise ValueError("'phrase' does not belong to 'frame'")
        self._frame = frame
        self.phrase = phrase
        self.descriptions = Conjuncts.get_chain(descriptions)
        self.complements = Conjuncts.get_chain(complements)

    def __new__(cls, *args: Any, **kwds: Any) -> None:
        obj = super().__new__(cls)
        obj.__init__(*args, **kwds)
        if (cur := obj.story.emap.get(obj.idx)):
            cur.__init__(obj.frame, **obj.data)
            return cur
        obj.story.emap[obj.idx] = obj
        return obj

    def __repr__(self) -> str:
        return self.to_str(color=True)

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: FrameElement) -> bool:
        if self.is_comparable_with(other):
            return self.phrase == other.phrase
        return NotImplemented

    def __lt__(self, other: FrameElement) -> bool:
        if self.is_comparable_with(other):
            return self.phrase < other.phrase
        return NotImplemented

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.init_class_attrs({
            "__parts__": "part_names"
        }, check_slots=True)
        for name in cls.part_names:
            finder = f"find_{name}"
            if not hasattr(cls, finder):
                raise TypeError(
                    f"'{cls.__name__} does not define "
                    f"'{finder}' finder method for '{name}' part"
                )

    # Abstract methods --------------------------------------------------------

    @staticmethod
    @abstractmethod
    def starts_from(phrase: Phrase) -> bool:
        """Does an element starts from ``phrase``."""
        raise NotImplementedError

    # Properties --------------------------------------------------------------

    @property
    def doc(self) -> DocABC:
        return self.story.doc

    @property
    def story(self) -> "Story":
        return self.frame.story

    @property
    def frame(self) -> "Frame":
        return self._frame

    @property
    def idx(self) -> int:
        return self.phrase.idx

    @property
    def head(self) -> Component:
        return self.phrase.head

    @property
    def depth(self) -> int:
        return self.phrase.depth

    @property
    def lead(self) -> FrameElement:
        if (lead := self.story.emap.get(self.phrase.lead.idx)):
            return lead
        return FrameElement.from_phrase(self.frame, self.phrase.lead)

    @property
    def conjuncts(self) -> Conjuncts:
        if (conjs := self.phrase.conjuncts):
            conjs = conjs.copy(members=tuple(
                FrameElement.from_phrase(self.frame, m)
                for m in conjs.members
            ))
        return conjs

    @property
    def hashdata(self) -> int:
        return (*super().hashdata, self.frame, self.phrase)

    # Methods -----------------------------------------------------------------

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, FrameElement)

    def to_str(self, *, color: bool = False, **kwds: Any) -> str:
        """Represent as string."""
        return " ".join(
            t.to_str(color=color, role=r, **kwds)
            for t, r in self.iter_token_roles()
        )

    def iter_token_roles(self) -> Iterator[tuple[TokenABC, Role | None]]:
        """Iterate over token-role pairs."""
        def _iter():
            yield from self.phrase.iter_token_roles()
        yield from sorted(set(_iter()), key=lambda x: x[0])

    def iter_relations(self) -> Iterator[tuple[Any, ...]]:
        """Iterate over semantic relations."""
        for parts in product(*(
            getattr(self, name) or (None,)
            for name in self.part_names
        )):
            kwds = dict(zip(self.Relation.slot_names[1:], parts))
            yield self.Relation(self.head, **kwds)

    @staticmethod
    def find_descriptions(phrase: Phrase) -> Iterator[Phrase]:
        for child in phrase.children:
            if Description.starts_from(child) \
            or child.dep & (Dep.relcl | Dep.appos):
                yield child
        for parent in phrase.parents:
            if parent.dep & Dep.cdesc and phrase in parent.subj:
                yield parent

    @staticmethod
    def find_complements(phrase: Phrase) -> Iterator[Phrase]:
        for child in phrase.children:
            if Complement.starts_from(child) \
            or child.dep & Dep.subcl:
                yield child

    @classmethod
    def from_phrase(cls, frame: "Frame", phrase: Phrase) -> Phrase:
        """Construct from a grammar phrase."""
        # pylint: disable=unnecessary-dunder-call
        for typ in cls.types.values():
            if not issubclass(typ, FrameElement) \
            or getattr(typ, "__abstractmethods__", None):
                continue
            if typ.starts_from(phrase):
                obj = typ(frame, phrase)
                parts = {
                    name: getattr(typ, f"find_{name}")(phrase)
                    for name in typ.part_names
                }
                obj.__init__(obj.frame, **{ **obj.data, **parts })
                return obj
        raise ValueError(f"no matching frame element type for '{cls.cname(phrase)}'")


class Actor(FrameElement):
    """Actor semantic frame element."""
    __parts__ = ()
    __slots__ = (*__parts__,)
    Relation: ClassVar[Type[Relation]] = ActorRelation

    @staticmethod
    def starts_from(phrase: Phrase) -> bool:
        return isinstance(phrase, NounPhrase) \
            and not phrase.dep & (Dep.nmod | Dep.adesc | Dep.cdesc)


class Action(FrameElement):
    """Action semantic frame element."""
    __parts__ = ("subjects", "dobjects", "iobjects")
    __slots__ = (*__parts__,)
    Relation: ClassVar[Type[Relation]] = ActionRelation

    def __init__(
        self,
        frame: "Frame",
        phrase: Phrase,
        *,
        subjects: Iterable[Phrase] = (),
        dobjects: Iterable[Phrase] = (),
        iobjects: Iterable[Phrase] = (),
        **kwds: Any
    ) -> None:
        super().__init__(frame, phrase, **kwds)
        self.subjects = Conjuncts.get_chain(subjects)
        self.dobjects = Conjuncts.get_chain(dobjects)
        self.iobjects = Conjuncts.get_chain(iobjects)

    # Methods -----------------------------------------------------------------

    @staticmethod
    def find_subjects(phrase: Phrase) -> Iterator[Phrase]:
        yield from phrase.subj

    @staticmethod
    def find_dobjects(phrase: Phrase) -> Iterator[Phrase]:
        yield from phrase.dobj

    @staticmethod
    def find_iobjects(phrase: Phrase) -> Iterator[Phrase]:
        yield from phrase.iobj

    @staticmethod
    def starts_from(phrase: Phrase) -> bool:
        return isinstance(phrase, VerbPhrase) \
            and not phrase.dep & Dep.xcomp


class Description(FrameElement):
    """Description semantic frame element."""
    __parts__ = ("objects", "verbs")
    __slots__ = (*__parts__,)
    Relation: ClassVar[Type[Relation]] = DescriptionRelation

    def __init__(
        self,
        frame: "Frame",
        phrase: Phrase,
        *,
        objects: Iterable[Phrase] = (),
        verbs: Iterable[Component] = (),
        **kwds: Any
    ) -> None:
        super().__init__(frame, phrase, **kwds)
        self.objects = Conjuncts.get_chain(objects)
        self.verbs = Conjuncts.get_chain(verbs)

    # Methods -----------------------------------------------------------------

    @staticmethod
    def find_objects(phrase: Phrase) -> Iterator[Phrase]:
        if phrase.adesc:
            if phrase.dobj:
                yield from phrase.dobj
            else:
                yield from phrase.subj
        elif phrase.dep & Dep.cdesc and phrase.subj:
            yield from phrase.subj
        elif phrase.dep & Dep.adesc:
            seen = set()
            for parent in phrase.parents:
                for subj in parent.subj:
                    if subj not in seen:
                        yield subj
                        seen.add(subj)
        else:
            yield from phrase.parents

    @staticmethod
    def find_verbs(phrase: Phrase) -> Iterator[Phrase]:
        if phrase.dep & Dep.adesc:
            for parent in phrase.parents:
                if isinstance(parent, VerbPhrase):
                    yield parent


    @staticmethod
    def starts_from(phrase: Phrase) -> bool:
        return isinstance(phrase, DescPhrase) \
            or phrase.dep & (Dep.adesc | Dep.cdesc | Dep.nmod)


class Complement(FrameElement):
    """Complement semantic frame element."""
    __parts__ = ("objects",)
    __slots__ = (*__parts__,)
    Relation: ClassVar[Type[Relation]] = ComplementRelation

    def __init__(
        self,
        frame: "Frame",
        phrase: Phrase,
        *,
        objects: Iterable[Phrase] = (),
        **kwds: Any
    ) -> None:
        super().__init__(frame, phrase, **kwds)
        self.objects = Conjuncts.get_chain(objects)

    # Methods -----------------------------------------------------------------

    @staticmethod
    def find_objects(phrase: Phrase) -> Iterator[Phrase]:
        yield from phrase.parents


    @staticmethod
    def starts_from(phrase: Phrase) -> bool:
        return isinstance(phrase, PrepPhrase) \
            or phrase.dep & Dep.xcomp
