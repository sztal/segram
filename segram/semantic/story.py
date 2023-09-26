from typing import Self, Any, Iterable, MutableMapping, Callable
from spacy.language import Language
from .frames import Frame, Actants, Events
from ..grammar import Sent, Phrase, Conjuncts
from ..nlp import Corpus
from ..grammar import Doc, Sent, Phrase
from ..datastruct import DataTuple, DataChain


class Story(MutableMapping):
    """Story class.

    Attributes
    ----------
    phrases
        Tuple of phrases the story operates on.
    """
    __slots__ = ("corpus", "frames")

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
    def docs(self) -> DataTuple[Doc]:
        """Enhanced grammar documents in the story."""

    @property
    def sents(self) -> DataChain[DataTuple[Sent]]:
        """Enhanced grammar sentences in the story."""
        return DataChain(DataTuple(doc.sents) for doc in self.corpus)

    # Methods -----------------------------------------------------------------

    @classmethod
    def from_texts(cls, nlp: Language, *texts: str, **kwds: Any) -> Self:
        """Construct from texts.

        All arguments are passed to :meth:`segram.nlp.Corpus`.
        """
        corpus = Corpus.from_texts(nlp, *texts, **kwds)
        return cls(corpus)
