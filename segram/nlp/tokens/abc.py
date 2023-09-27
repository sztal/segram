"""Abstract base class for :mod:`segram`-enhanced :mod:`spacy` tokens."""
# pylint: disable=no-name-in-module
from typing import Any, Self
from abc import ABC
from functools import total_ordering
import numpy as np
from spacy.vocab import Vocab
from spacy.tokens import Doc, Span, Token
from spacy.tokens.underscore import Underscore
from ... import settings
from ...utils.meta import init_class_attrs
from ...utils.misc import cosine_similarity


class NLP(ABC):
    """Abstract base class for NLP tokens.

    Attributes
    ----------
    tok
        Base :mod:`spacy` token object.
    """
    __slots__ = ("tok",)

    def __init__(self, tok: Doc | Span | Token) -> None:
        self.tok = tok

    def __repr__(self) -> str:
        """String representation."""
        return self.text

    def __hash__(self) -> int:
        return hash((0, self.tok))

    def __eq__(self, other: Self) -> bool:
        """Check equality with another token of the same type."""
        if self.is_comparable_with(other) is True:
            return self.doc == other.doc
        return NotImplemented

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "__slots__" not in cls.__dict__:
            raise TypeError(f"'{cls.__name__}' does not define '__slots__'")
        init_class_attrs(cls, {
            "__slots__": "slot_names"
        })
        try:
            total_ordering(cls)
        except ValueError:
            pass

    # Properties --------------------------------------------------------------

    @property
    def text(self) -> str:
        return self.tok.text

    @property
    def doc(self) -> "Doc":
        return self.sns(self.tok.doc)

    @property
    def lang(self) -> str:
        return self.doc.lang

    @property
    def vocab(self) -> Vocab:
        return self.tok.vocab

    @property
    def vector(self) -> np.ndarray[tuple[int], np.floating]:
        return self.tok.vector

    @property
    def has_vectors(self) -> bool:
        return self.vocab.vectors_length > 0

    # Methods -----------------------------------------------------------------

    def is_comparable_with(self, other: Any) -> bool:
        """Check if ``self`` defines the same abstract interface as ``other``."""
        if not isinstance(other, NLP):
            return NotImplemented
        if self.doc is not other.doc:
            raise ValueError("'self' and 'other' are based on different documents")
        return isinstance(other, self.__class__) or NotImplemented

    # Properties --------------------------------------------------------------

    @property
    def _(self) -> Underscore:
        return self.tok._

    # Methods -----------------------------------------------------------------

    @classmethod
    def sns(cls, tok: Doc | Span | Token) -> Self:
        """Get :mod:`segram` namespace from :mod:`spacy` token."""
        return getattr(tok._, settings.spacy_alias+"_sns")

    def similarity(self, other: Doc | Span | Token) -> float:
        return cosine_similarity(self.vector, other.vector)
