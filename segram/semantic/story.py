from typing import Self, Any, Iterable, MutableMapping, Callable
from more_itertools import unique_everseen
from .abc import Semantic
from .frames import Frame, Actants, Events
from ..grammar import Sent, Phrase, Conjuncts
from ..nlp import Corpus, Doc
from ..datastruct import DataSequence


class Story(Semantic, MutableMapping):
    """Semantic story class.

    Attributes
    ----------
    phrases
        Tuple of phrases the story operates on.
    """
    __slots__ = ("corpus", "_phrases", "frames")

    def __init__(
        self,
        corpus: Corpus | None = None,
        **kwds: Frame
    ) -> None:
        self.corpus = corpus
        self.frames = {
            "actants": Actants(self),
            "events": Events(self),
            **kwds
        }

    def __getitem__(self, key: str) -> Frame:
        return self.frames[key]

    def __setitem__(self, key: str, value: Frame) -> None:
        if isinstance(value, Callable) \
        and not isinstance(value, Frame):
            value = Frame.subclass(value)(self)
        self.frames[key] = value

    def __delitem__(self, key: str) -> None:
        del self.frames[key]

    def __iter__(self) -> Iterable[str]:
        yield from self.frames

    def __len__(self) -> int:
        return len(self.frames)

    # Properties --------------------------------------------------------------

    @property
    def phrases(self) -> DataSequence[Phrase]:
        return self._phrases \
            .groupby(lambda p: p.lead.sent.idx) \
            .groupby(lambda g: hash(g[0].doc))
    @phrases.setter
    def _(self, phrases: Iterable[Phrase]) -> None:
        self._phrases = Conjuncts.get_chain(phrases)
        for frame in self.frames:
            frame.clear()

    @property
    def sents(self) -> Iterable[Sent]:
        return DataSequence(unique_everseen(
            (p.sent for p in self.phrases),
            key=lambda s: (hash(s.doc), s.idx)
        )).groupby(lambda s: hash(s.doc))

    # Constructors ------------------------------------------------------------

    @classmethod
    def from_sents(cls, *sents: Sent) -> Self:
        """Construct from sentences."""
        phrases = [ p for s in sents for p in s.phrases ]
        return cls(phrases)

    @classmethod
    def from_docs(cls, *docs: Doc) -> Self:
        """Construct from documents."""
        sents = [ sent.grammar for doc in docs for sent in doc.sents ]
        return cls.from_sents(*sents)

    @classmethod
    def from_corpus(cls, corpus: Corpus) -> Self:
        """Construct from a corpus."""
        return cls.from_docs(*corpus.docs)

    # Methods -----------------------------------------------------------------

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, Story)

    def copy(self, **kwds: Any) -> Self:
        # pylint: disable=protected-access
        new = self.__class__(phrases=self.phrases, **kwds)
        new.frames = self.frames.copy()
        return new
