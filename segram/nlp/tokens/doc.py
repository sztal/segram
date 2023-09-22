from __future__ import annotations
from typing import Any, Optional, Iterable, Self
from spacy.tokens import Doc as SpacyDoc, Token as SpacyToken
from .abc import NLP
from .token import Token
from .span import Span
from ... import settings, __title__
from ...utils.registries import grammars
from ...utils.diff import iter_diffs, equal, IDiffType


class Doc(NLP):
    """Enhanced document class."""
    __slots__ = ()

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, Doc):
            return self.tok == other.tok
        return NotImplemented

    def __iter__(self) -> Iterable[Token]:
        for tok in self.tok:
            yield self.sns(tok)

    def __len__(self) -> int:
        return len(self.tok)

    def __getitem__(self, idx: int | slice) -> Token | Span:
        return self.sns(self.tok[idx])

    def __contains__(self, other: Token | SpacyToken) -> bool:
        if isinstance(other, Token):
            return other.tok in self.tok
        return other in self.tok

    # Properties --------------------------------------------------------------

    @property
    def doc(self) -> Self:
        return self

    @property
    def lang(self) -> str:
        return self.tok.lang_

    @property
    def id(self) -> int:
        """Hash id of the document tokenization."""
        return hash(t.attrs for t in self)

    @property
    def noun_chunks(self) -> Iterable[Span]:
        for chunk in self.tok.noun_chunks:
            yield self.sns(chunk)

    @property
    def sents(self) -> Iterable[Span]:
        for sent in self.tok.sents:
            yield self.sns(sent)

    # @property
    # def simple(self) -> SimpleDoc:
    #     """Generic :mod:`segram` document object."""
    #     # pylint: disable=protected-access
    #     data = []
    #     sent_spans = []
    #     for sent in self.doc.sents:
    #         start, end = sent.start, sent.end
    #         sent_spans.append((start, end))
    #         for tok in sent:
    #             gtok = TokenData(
    #                 text=tok.text,
    #                 pos=tok.pos,
    #                 whitespace=tok.whitespace,
    #                 lemma=tok.lemma,
    #                 ent=tok.ent,
    #                 role=tok.role,
    #                 corefs=getattr(tok._, f"{settings.spacy_alias}_corefs"),
    #                 is_negation=tok.is_negation,
    #                 is_qmark=tok.is_qmark,
    #                 is_exclam=tok.is_exclam,
    #                 sent_start=start,
    #                 sent_end=end
    #             )
    #             data.append(gtok)
    #     return SimpleDoc(self.lang, tuple(data), tuple(sent_spans))

    # Methods -----------------------------------------------------------------

    def char_span(self, *args: Any, **kwds: Any) -> Optional[Span]:
        res = self.tok.char_span(*args, **kwds)
        return res if res is None else self.sns(res)

    @classmethod
    def from_docs(cls, *args: Any, **kwds: Any) -> Optional[Doc]:
        res = Doc.from_docs(*args, **kwds)
        return res if res is None else cls.sns(res)

    def copy(self) -> SpacyDoc:
        return self.sns(self.tok.copy())

    def get_grammar(self):
        key = getattr(self._, f"{settings.spacy_alias}_meta")[f"{__title__}_grammar"]
        return grammars.get(key)


# Register comparison functions for testing -----------------------------------

@equal.register
def _(obj: Doc, other: Doc, *, strict: bool = True) -> bool:
    return ((strict and obj == other) or (not strict and obj.id == other.id))
@iter_diffs.register
def _(obj: Doc, other: Doc, *, strict: bool = True) -> IDiffType:
    if not equal(obj, other, strict=strict):
        msg = "DOCUMENT CONTENT"
        if obj.id == other.id:
            msg = "DOCUMENT TYPE"
        yield msg, obj, other
