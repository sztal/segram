from __future__ import annotations
from typing import Any, Optional, Iterable
from spacy.tokens import Doc, Token
from .abc import SpacyNLPTokenABC
from .token import SpacyTokenABC
from .span import SpacySpan
from ...tokens import DocABC
from ...tokens import Doc as SimpleDoc, TokenData
from .... import settings, __title__
from ....utils.registries import grammars


class SpacyDoc(SpacyNLPTokenABC, DocABC):
    """Enhanced document class."""
    __slots__ = ()

    def __hash__(self) -> int:
        return hash((0, hash(self.tok)))

    def __eq__(self, other: SpacyDoc) -> bool:
        if isinstance(other, SpacyDoc):
            return self.tok == other.tok
        return NotImplemented

    def __iter__(self) -> Iterable[SpacyTokenABC]:
        for tok in self.tok:
            yield self.sns(tok)

    def __len__(self) -> int:
        return len(self.tok)

    def __getitem__(self, idx: int | slice):
        return self.sns(self.tok[idx])

    def __contains__(self, other: SpacyTokenABC | Token) -> bool:
        if isinstance(other, SpacyTokenABC):
            return other.tok in self.tok
        return other in self.tok

    # Properties --------------------------------------------------------------

    @property
    def doc(self) -> SpacyDoc:
        return self

    @property
    def lang(self) -> str:
        return self.tok.lang_

    @property
    def noun_chunks(self) -> Iterable[SpacySpan]:
        for chunk in self.tok.noun_chunks:
            yield self.sns(chunk)

    @property
    def sents(self) -> Iterable[SpacySpan]:
        for sent in self.tok.sents:
            yield self.sns(sent)

    @property
    def simple(self) -> SimpleDoc:
        """Generic :mod:`segram` document object."""
        # pylint: disable=protected-access
        data = []
        sent_spans = []
        for sent in self.doc.sents:
            start, end = sent.start, sent.end
            sent_spans.append((start, end))
            for tok in sent:
                gtok = TokenData(
                    text=tok.text,
                    pos=tok.pos,
                    whitespace=tok.whitespace,
                    lemma=tok.lemma,
                    ent_type=tok.ent_type,
                    role=tok.role,
                    refs=getattr(tok._, f"{settings.spacy_alias}_refs"),
                    is_negation=tok.is_negation,
                    is_qmark=tok.is_qmark,
                    is_exclam=tok.is_exclam,
                    sent_start=start,
                    sent_end=end
                )
                data.append(gtok)
        return SimpleDoc(self.lang, tuple(data), tuple(sent_spans))

    # Methods -----------------------------------------------------------------

    def char_span(self, *args: Any, **kwds: Any) -> Optional[SpacySpan]:
        res = self.tok.char_span(*args, **kwds)
        return res if res is None else self.sns(res)

    @classmethod
    def from_docs(cls, *args: Any, **kwds: Any) -> Optional[SpacyDoc]:
        res = Doc.from_docs(*args, **kwds)
        return res if res is None else cls.sns(res)

    def copy(self) -> SpacyDoc:
        return self.sns(self.tok.copy())

    def get_grammar(self):
        key = getattr(self._, f"{settings.spacy_alias}_meta")[f"{__title__}_grammar"]
        return grammars.get(key)
