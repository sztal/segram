"""Abstract base class for :mod:`segram`-enhanced :mod:`spacy` tokens."""
from __future__ import annotations
from spacy.tokens import Doc, Span, Token
from spacy.tokens.underscore import Underscore
from ...abc import NLPToken
from ...abc.tokens import DocABC
from .... import settings


class SpacyNLPToken(NLPToken):
    """Abstract base class for enhanced :mod:`spacy` tokens.

    Attributes
    ----------
    tok
        Token object.
    """
    __slots__ = ("tok",)

    def __init__(self, tok: Doc | Span | Token) -> None:
        self.tok = tok

    # Properties --------------------------------------------------------------

    @property
    def _(self) -> Underscore:
        return self.tok._

    @property
    def text(self) -> str:
        return self.tok.text

    @property
    def doc(self) -> DocABC:
        return self.sns(self.tok.doc)

    # Methods -----------------------------------------------------------------

    @classmethod
    def sns(cls, tok: Doc | Span | Token) -> SpacyNLPToken:
        """Get :mod:`segram` namespace from :mod:`spacy` token."""
        return getattr(tok._, settings.spacy_alias)
