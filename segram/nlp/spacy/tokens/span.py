from __future__ import annotations
from typing import Any, Optional, Iterable
from spacy import displacy
from spacy.tokens import Span, Token
from .abc import SpacyNLPTokenABC
from .token import SpacyTokenABC
from ...tokens import DocABC, SpanABC
from .... import settings
from ....grammar import Sent


class SpacySpan(SpacyNLPTokenABC, SpanABC):
    """Enhanced span class."""
    __slots__ = ()

    def __iter__(self) -> Iterable[SpacySpan]:
        for tok in self.span:
            yield self.sns(tok)

    def __len__(self) -> int:
        return len(self.span)

    def __getitem__(self, idx: int | slice) -> SpacySpan | SpacyTokenABC:
        return self.sns(self.span[idx])

    def __contains__(self, other: SpacyTokenABC | Token) -> bool:
        if isinstance(other, SpacyTokenABC):
            return other.tok in self.span
        return other in self.span

    # Properties --------------------------------------------------------------

    @property
    def span(self) -> Span:
        return self.tok

    @property
    def sent(self) -> SpacySpan:
        return self.sns(self.span.sent)

    @property
    def sents(self) -> Iterable[SpacySpan]:
        for sent in self.span.sents:
            yield self.sns(sent)

    @property
    def doc(self) -> DocABC:
        return self.sns(self.span.doc)

    @property
    def start(self) -> int:
        return self.span.start

    @property
    def end(self) -> int:
        return self.span.end

    @property
    def root(self) -> SpacyTokenABC:
        return self.sns(self.span.root)

    @property
    def conjuncts(self) -> tuple[SpacyTokenABC, ...]:
        return self.root.conjuncts

    @property
    def noun_chunks(self) -> Iterable[SpacySpan]:
        for chunk in self.span.noun_chunks:
            yield self.sns(chunk)

    @property
    def lefts(self) -> Iterable[SpacyTokenABC]:
        for tok in self.span.lefts:
            yield self.sns(tok)

    @property
    def rights(self) -> Iterable[SpacyTokenABC]:
        for tok in self.span.rights:
            yield self.sns(tok)

    @property
    def subtree(self) -> Iterable[SpacyTokenABC]:
        for tok in self.span.subtree:
            yield self.sns(tok)

    # Methods -----------------------------------------------------------------

    def grammar(self, *, use_data: Optional[bool] = None) -> Sent:
        """Get grammar sentence object.

        Parameters
        ----------
        use_data
            Should precomputed data stored in the underlying
            document object be used instead of parsing of the
            sentence. If ``None`` then the data is used if it
            is available.
        """
        typ = self.doc.get_grammar()
        alias = settings.spacy_alias
        if use_data is None or use_data:
            attr = f"{alias}_grammar_data"
            data = getattr(self.doc._, attr, None)
            use_data = bool(data)
        if not use_data:
            return typ.types.Sent.from_sent(self)
        data = data[(self.start, self.end)]
        return typ.types.Sent.from_data(self.doc, data)

    def char_span(self, *args: Any, **kwds: Any) -> Optional[SpacySpan]:
        out = self.span.char_span(*args, **kwds)
        if out is not None:
            out = self.sns(out)
        return out

    def as_doc(self, *args: Any, **kwds: Any) -> DocABC:
        doc = self.span.as_doc(*args, **kwds)
        doc._.segram_extensions = self.doc._.segram_extensions
        return self.sns(doc)

    def display(self, *args: Any, **kwds: Any) -> None:
        """Visualize syntactic dependency structure
        using :func:`spacy.displacy.serve`
        """
        displacy.serve(self.span, *args, **kwds)

    def render(self, *args: Any, **kwds: Any) -> str:
        """Get SVG string representing syntactic
        dependency structure using :func:`spacy.displacy.render`.
        """
        return displacy.render(self.span, *args, **kwds)
