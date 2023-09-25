from __future__ import annotations
from typing import Any, Iterable, Self
from spacy.tokens import Doc as SpacyDoc, Token as SpacyToken
from .abc import NLP
from .token import Token
from .span import Span
from ... import settings
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
        return hash(tuple(k, tuple(v)) for k, v in self.data.items())

    @property
    def noun_chunks(self) -> Iterable[Span]:
        for chunk in self.tok.noun_chunks:
            yield self.sns(chunk)

    @property
    def sents(self) -> Iterable[Span]:
        for sent in self.tok.sents:
            yield self.sns(sent)

    @property
    def data(self) -> dict[str, Any]:
        return self.to_data()

    @property
    def cache(self) -> dict[str, dict[int | tuple[int, int], Any]]:
        return getattr(self._, f"{settings.spacy_alias}_cache")

    # Methods -----------------------------------------------------------------

    def to_data(self) -> dict[str, Any]:
        """Dump to data dictionary sufficient to recreate simple document
        without any language model data.
        """
        user_data = self.tok.user_data.copy()
        cachekey = ("._.", f"{settings.spacy_alias}_cache", None, None)
        user_data[cachekey] = {}
        data = {
            "vocab": self.vocab,
            "words": [ t.text for t in self ],
            "spaces": [ t.whitespace for t in self ],
            "user_data": user_data,
            "tags": [ t.tag_ for t in self.tok ],
            "pos": [ t.pos_ for t in self.tok ],
            "morphs": [ str(t.morph) for t in self.tok ],
            "lemmas": [ t.lemma_ for t in self.tok ],
            "heads": [ t.head.i for t in self.tok ],
            "deps": [ t.dep_ for t in self.tok ],
            "ents": [ f"{t.ent_tag}" for t in self ]
        }
        return data

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> Self:
        """Construct from data dictionary produced by :meth:`to_data`."""
        return getattr(SpacyDoc(**data)._, settings.spacy_alias)

    def char_span(self, *args: Any, **kwds: Any) -> Span | None:
        res = self.tok.char_span(*args, **kwds)
        return res if res is None else self.sns(res)

    @classmethod
    def from_docs(cls, *args: Any, **kwds: Any) -> Doc | None:
        res = Doc.from_docs(*args, **kwds)
        return res if res is None else cls.sns(res)

    def copy(self) -> SpacyDoc:
        return self.sns(self.tok.copy())

    def get_grammar_type(self):
        key = getattr(self._, f"{settings.spacy_alias}_meta")[f"{settings.spacy_alias}_grammar"]
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
