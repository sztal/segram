"""_Segram_ generic class for NLP document."""
from __future__ import annotations
from typing import Iterable
from .abc import DocABC, SpanABC, TokenABC
from .token import TokenData, Token
from .span import Span


class Doc(DocABC):
    """Minimal document class allowing interoperability
    with grammar classes.

    Attributes
    ----------
    lang
        Document language.
    data
        Raw token data.
    sent_spans
        Start and end indices of sentence spans.
    """
    __slots__ = ("_lang", "data", "sent_spans")

    def __init__(
        self,
        lang: str,
        data: Iterable[TokenData],
        sent_spans: Iterable[tuple[int, int]]
    ) -> None:
        super().__init__()
        self._lang = lang
        self.data = tuple(data)
        self.sent_spans = tuple(sent_spans)

    def __iter__(self) -> Iterable[TokenABC]:
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int | slice) -> TokenABC | SpanABC:
        # pylint: disable=abstract-class-instantiated
        if isinstance(idx, slice):
            if idx.step is not None:
                raise ValueError("stepped slices not supported")
            start = idx.start or 0
            end = idx.stop or len(self)
            return Span(self, start, end)
        if isinstance(idx, int):
            return Token(self, idx)
        raise TypeError("'idx' has to be 'int' or 'slice'")

    def __contains__(self, tok: Token) -> bool:
        return tok.data in self.data

    # Properties --------------------------------------------------------------

    @property
    def doc(self) -> Doc:
        return self

    @property
    def lang(self) -> str:
        return self._lang

    @property
    def sents(self) -> Iterable[Span]:
        for start, end in self.sent_spans:
            yield self[start:end]

    @property
    def text(self) -> str:
        return "".join(t.text_with_ws for t in self)

    # Methods -----------------------------------------------------------------

    def copy(self) -> Doc:
        return self.__class__(
            data=tuple(TokenData(*tok) for tok in self.data),
            sent_spans=tuple((*span,) for span in self.sent_spans),
            lang=self.lang
        )
