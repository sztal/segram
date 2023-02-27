"""_Segram_ generic class for NLP span."""
from __future__ import annotations
from typing import Iterable
from .abc import DocABC, SpanABC, TokenABC


class Span(SpanABC):
    """Minimal span class allowing interoperability
    with grammar classes.

    Attributes
    ----------
    doc
        Document object.
    start
        Start index.
    end
        End index + 1  (Python indexing convention).
    """
    __slots__ = ("doc", "_start", "_end")

    def __init__(self, doc: DocABC, start: int, end: int) -> None:
        self.doc = doc
        self._start = start
        self._end = end

    def __iter__(self) -> Iterable[TokenABC]:
        for i in range(self.start, self.end):
            yield self.doc[i]

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int | slice) -> TokenABC:
        if isinstance(idx, slice):
            start = (idx.start or 0) + self.start
            end = (idx.end or len(self)) + self.end
            idx = slice(start, end, idx.step)
        else:
            idx = idx + self.start
        return self.doc[idx]

    def __contains__(self, tok: TokenABC) -> bool:
        if not isinstance(tok, TokenABC):
            raise NotImplementedError
        return self.start <= tok.i <= self.end

    @property
    def start(self) -> int:
        return self._start

    @property
    def end(self) -> int:
        return self._end

    @property
    def text(self) -> str:
        return "".join(t.text_with_ws for t in self)
