from typing import Self, Any, Callable
from types import ModuleType
import os
import pickle
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

    def to_data(self) -> dict[str, Any]:
        """Dump to data dictionary."""
        return {
            "corpus": self.corpus.to_data(),
            "frames": self.frames.copy()
        }

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> Self:
        """Construct from data dictionary."""
        data = data.copy()
        data["corpus"] = Corpus.from_data(data["corpus"])
        return cls(**data)

    def to_disk(
        self,
        path: str | bytes | os.PathLike,
        compression: ModuleType | type | None = None
    ) -> None:
        """Save to disk.

        Anything exposing :func:`open` function/method
        can be passed as ``compression`` argument.
        """
        _open = compression.open if compression else open
        with _open(path, "wb") as fh:
            pickle.dump(self.to_data(), fh)

    @classmethod
    def from_disk(
        cls,
        path: str | bytes | os.PathLike,
        compression: ModuleType | type | None = None
    ) -> Self:
        """Construct from disk.

        Anything exposing :func:`open` function/method
        can be passed as ``compression`` argument.
        """
        _open = compression.open if compression else open
        with _open(path, "rb") as fh:
            return cls.from_data(pickle.load(fh))
