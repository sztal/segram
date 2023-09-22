"""_Segram_ generic class for NLP token."""
from __future__ import annotations
from typing import Optional, NamedTuple
from ..abc.tokens import TokenABC, SpanABC, DocABC
from ...symbols import POS, Role


# Token data container --------------------------------------------------------

class TokenData(NamedTuple):
    """Token data container."""
    text: str
    pos: POS
    whitespace: str = ""
    lemma: Optional[str] = None
    ent: Optional[str] = None
    role: Optional[Role] = None
    corefs: Optional[tuple[int, ...]] = None
    is_negation: bool = False
    is_qmark: bool = False
    is_exclam: bool = False
    is_intj: bool = False
    sent_start: Optional[int] = None
    sent_end: Optional[int] = None


# Token -----------------------------------------------------------------------


class Token(TokenABC):
    """Minimal token class allowing interoperability
    with grammar classes.

    Attributes
    ----------
    doc
        Document object controlling the token.
    i
        Index within the document.
    """
    __slots__ = ("doc", "_i")

    def __init__(self, doc: DocABC, i: int) -> None:
        self.doc = doc
        self._i = i

    # Properties --------------------------------------------------------------

    @property
    def i(self) -> int:
        return self._i

    @property
    def data(self) -> TokenData:
        return self.doc.data[self.i]

    @property
    def text(self) -> str:
        return self.data.text

    @property
    def text_with_ws(self) -> str:
        return self.text+self.whitespace

    @property
    def whitespace(self) -> str:
        return self.data.whitespace

    @property
    def lemma(self) -> str:
        return self.data.lemma or self.text

    @property
    def pos(self) -> POS:
        return self.data.pos

    @property
    def role(self) -> Role:
        return self.data.role

    @property
    def ent(self) -> str:
        return self.data.ent

    @property
    def corefs(self) -> Token:
        if (refs := self.data.corefs):
            return tuple(self.doc[ref] for ref in refs)
        return ()

    @property
    def is_negation(self) -> bool:
        return self.data.is_negation

    @property
    def is_qmark(self) -> bool:
        return self.data.is_qmark

    @property
    def is_exclam(self) -> bool:
        return self.data.is_exclam

    @property
    def is_intj(self) -> bool:
        return self.data.is_intj

    @property
    def sent(self) -> SpanABC:
        data = self.data
        return self.doc[data.sent_start:data.sent_end]
