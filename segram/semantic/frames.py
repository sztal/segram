from typing import Self, Any, Sequence, Callable
from abc import abstractmethod
import re
from ..grammar import Sent, Phrase, NounPhrase, VerbPhrase
from ..symbols import Dep
from ..utils.matching import Matcher
from ..datastruct import DataIterator


class Frame(Sequence):
    """Semantic frame class.

    Semantic frames groups phrases from a given :class:`segram.semantic.Story`
    instance according to some selection criteria.

    Attributes
    ----------
    story
        A Story the frame belongs to.
    matcher
        Arbitrary callable implementing filtering function.
    phrases
        Phrases matching the criteria.
    """
    def __init__(self, story: "Story") -> None:
        self.story = story
        self.matcher = Matcher(self.is_match)

    def __len__(self) -> int:
        return len(self.phrases)

    def __getitem__(self, idx: int | slice) -> Phrase | tuple[Phrase, ...]:
        return self.phrases[idx]

    def __and__(self, other: Self) -> Self:
        if isinstance(other, Frame | Callable):
            new = self.copy()
            func = other.matcher if isinstance(other, Frame) else other
            new.matcher &= func
            return new
        return NotImplemented

    def __rand__(self, other: Self) -> Self:
        return self & other

    def __or__(self, other: Self) -> Self:
        if isinstance(other, Frame | Callable):
            new = self.copy()
            func = other.matcher if isinstance(other, Frame) else other
            new.matcher |= func
            return new
        return NotImplemented

    def __ror__(self, other: Self) -> Self:
        return self | other

    def __call__(self, phrase: Phrase) -> bool:
        return self.match(phrase)

    # Properties --------------------------------------------------------------

    @property
    def phrases(self) -> DataIterator[Phrase]:
        return self.story.phrases.filter(self.match)

    @property
    def sents(self) -> DataIterator[Sent]:
        return self.phrases.unique(lambda p: (p.doc.idx, p.idx)).get("sent")

    # Methods -----------------------------------------------------------------

    @abstractmethod
    def is_match(self, phrase: Phrase) -> bool:
        """Does ``phrase`` match the criteria of the frame."""

    def match(self, phrase: Phrase) -> bool:
        """Match phrase against the selection criteria."""
        return self.matcher(phrase)

    def copy(self, **kwds: Any) -> Self:
        return self.__class__(**{ "story": self.story, **kwds })

    @classmethod
    def subclass(cls, is_match: Callable[[Phrase], bool]) -> type[Self]:
        """Make subclass from a callable.

        Callable is injected into a class as a staticmethod,
        so it should not use the ``self`` parameter.
        """
        name = re.sub(r"\W", r"", is_match.__name__)
        return type(f"{name}{id(is_match)}", (cls,), {
            "__slots__": (),
            "is_match": staticmethod(is_match)
        })


class Actants(Frame):
    """Semantic frame of actants."""
    __slots__ = ()

    def is_match(self, phrase: Phrase) -> bool:
        match phrase:
            case NounPhrase(dep=dep):
                return not dep & (Dep.nmod | Dep.desc | Dep.appos)
            case _:
                return False


class Events(Frame):
    """Semantic frame of events."""
    __slots__ = ()

    def is_match(self, phrase: Phrase) -> bool:
        match phrase:
            case VerbPhrase(dep=dep):
                return not dep & Dep.xcomp
            case _:
                return False
