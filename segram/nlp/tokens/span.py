from typing import Any, Iterable, Self
from spacy import displacy
from spacy.tokens import Span as SpacySpan
from .abc import NLP
from .token import Token
from ...utils.diff import iter_diffs, equal, IDiffType


class Span(NLP):
    """Span wrapper class."""
    __slots__ = ()

    def __iter__(self) -> Iterable[Token]:
        for tok in self.span:
            yield self.sns(tok)

    def __len__(self) -> int:
        return len(self.span)

    def __getitem__(self, idx: int | slice) -> Self | Token:
        return self.sns(self.span[idx])

    def __contains__(self, other: Token | Token) -> bool:
        if isinstance(other, Token):
            return other.tok in self.span
        return other in self.span

    # Properties --------------------------------------------------------------

    @property
    def span(self) -> Self:
        return self.tok

    @property
    def sent(self) -> SpacySpan:
        return self.sns(self.span.sent)

    @property
    def sents(self) -> Iterable[SpacySpan]:
        for sent in self.span.sents:
            yield self.sns(sent)

    @property
    def doc(self) -> "Doc":
        return self.sns(self.span.doc)

    @property
    def start(self) -> int:
        return self.span.start

    @property
    def end(self) -> int:
        return self.span.end

    @property
    def root(self) -> Token:
        return self.sns(self.span.root)

    @property
    def conjuncts(self) -> tuple[Token, ...]:
        return self.root.conjuncts

    @property
    def noun_chunks(self) -> Iterable[SpacySpan]:
        for chunk in self.span.noun_chunks:
            yield self.sns(chunk)

    @property
    def lefts(self) -> Iterable[Token]:
        for tok in self.span.lefts:
            yield self.sns(tok)

    @property
    def rights(self) -> Iterable[Token]:
        for tok in self.span.rights:
            yield self.sns(tok)

    @property
    def subtree(self) -> Iterable[Token]:
        for tok in self.span.subtree:
            yield self.sns(tok)

    @property
    def grammar(self) -> "Sent":
        """Grammar sentence object associated with the NLP sentence."""
        if (obj := self.doc.grammar.smap.get((self.start, self.end))):
            return obj
        typ = self.doc.get_grammar_type()
        return typ.types.Sent.from_sent(self)

    # Methods -----------------------------------------------------------------

    def char_span(self, *args: Any, **kwds: Any) -> SpacySpan | None:
        out = self.span.char_span(*args, **kwds)
        if out is not None:
            out = self.sns(out)
        return out

    def as_doc(self, *args: Any, **kwds: Any) -> "Doc":
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


# Register comparison functions for testing -----------------------------------

@equal.register
def _(obj: Span, other: Span, *, strict: bool = True) -> bool:
    return equal(obj.doc, other.doc, strict=strict) \
        and (obj.start, obj.end) == (other.start, other.end)
@iter_diffs.register
def _(obj: Span, other: Span, *, strict: bool = True) -> IDiffType:
    if not equal(obj, other, strict=strict):
        yield "SPAN", obj, other
