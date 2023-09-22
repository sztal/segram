from __future__ import annotations
from typing import Self, Any, Iterable, MutableMapping, Callable
from more_itertools import unique_everseen
from .abc import Semantic
from .frames import Frame, Actants, Events
from ..grammar import Sent, Phrase, Conjuncts
from ..nlp import Corpus
from ..nlp.tokens import Doc


class Story(Semantic, MutableMapping):
    """Semantic story class.

    Attributes
    ----------
    phrases
        Tuple of phrases the story operates on.
    """
    __slots__ = ("_phrases", "_frames")

    def __init__(
        self,
        phrases: Iterable[Phrase] = ()
    ) -> None:
        self._phrases = Conjuncts.get_chain(phrases)
        self._frames = {
            "actants": Actants(self),
            "events": Events(self)
        }

    def __getitem__(self, key: str) -> Frame:
        return self._frames[key]

    def __setitem__(self, key: str, value: Frame) -> None:
        if isinstance(value, Callable):
            value = Frame.subclass(value)(self)
        self._frames[key] = value

    def __delitem__(self, key: str) -> None:
        del self._frames[key]

    def __iter__(self) -> Iterable[str]:
        yield from self._frames

    def __len__(self) -> int:
        return len(self._frames)

    # Properties --------------------------------------------------------------

    @property
    def phrases(self) -> tuple[Phrase, ...]:
        return self._phrases
    @phrases.setter
    def _(self, phrases: Iterable[Phrase]) -> None:
        self._phrases = Conjuncts.get_chain(phrases)
        for frame in self.frames:
            frame.clear()

    @property
    def frames(self) -> tuple[Frame, ...]:
        return tuple(self._frames.values())

    @property
    def sents(self) -> Iterable[Sent]:
        yield from unique_everseen(p.sent for p in self.phrases)

    # Constructors ------------------------------------------------------------

    @classmethod
    def from_sents(cls, *sents: Sent) -> Self:
        """Construct from sentences."""
        phrases = [ p for s in sents for p in s.phrases ]
        return cls(phrases)

    @classmethod
    def from_docs(cls, *docs: Doc, **kwds: Any) -> Self:
        """Construct from documents.

        ``**kwds`` are passed to :meth:`segram.nlp.tokens.span.Span.grammar`.
        """
        sents = [ sent for doc in docs for sent in doc.iter_grammar(**kwds) ]
        return cls.from_sents(*sents)

    @classmethod
    def from_corpus(cls, corpus: Corpus, **kwds: Any) -> Self:
        """Construct from a corpus.

        ``**kwds`` are passed to :meth:`segram.nlp.tokens.span.Span.grammar`.
        """
        return cls.from_docs(*corpus.docs, **kwds)

    # Methods -----------------------------------------------------------------

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, Story)

    def copy(self, **kwds: Any) -> Self:
        # pylint: disable=protected-access
        new = self.__class__(phrases=self.phrases, **kwds)
        new._frames = self._frames.copy()
        return new
