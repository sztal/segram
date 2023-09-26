from typing import Self, Any, Callable
from spacy.language import Language
from .frames import Frame, Actants, Events
from ..grammar import Doc, Sent, Phrase
from ..nlp import Corpus
from ..datastruct import DataIterable


class Story:
    """Story class.

    Attributes
    ----------
    phrases
        Tuple of phrases the story operates on.
    """
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

    # Properties --------------------------------------------------------------

    @property
    def docs(self) -> DataIterable[Doc]:
        """Grammar documents in the story."""
        return DataIterable(doc.grammar for doc in self.corpus.docs)

    @property
    def sents(self) -> DataIterable[Sent]:
        """Grammar sentences in the story."""
        return DataIterable(doc.sents for doc in self.docs).flat

    @property
    def phrases(self) -> DataIterable[Phrase]:
        """Phrase in the story."""
        return DataIterable(s.phrases for s in self.sents).flat

    # Methods -----------------------------------------------------------------

    def copy(self) -> Self:
        obj = self.__class__(self.corpus.copy())
        obj.frames = { n: f.copy(story=obj) for n, f in self.frames.copy() }
        return obj

    @classmethod
    def from_texts(cls, nlp: Language, *texts: str, **kwds: Any) -> Self:
        """Construct from texts.

        All arguments are passed to :meth:`segram.nlp.Corpus`.
        """
        corpus = Corpus.from_texts(nlp, *texts, **kwds)
        return cls(corpus)
