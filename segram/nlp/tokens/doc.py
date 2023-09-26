from typing import Any, Iterable, Self
import json
from murmurhash import hash_unicode
from spacy.tokens import Doc as SpacyDoc
from spacy.tokens import Span as SpacySpan
from spacy.tokens import Token as SpacyToken
from .abc import NLP
from .token import Token
from .span import Span
from ... import settings
from ...utils.registries import grammars
from ...utils.diff import iter_diffs, equal, IDiffType


class Doc(NLP):
    """Enhanced document class."""
    __slots__ = ("_id",)

    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        self._id = None

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

    def __contains__(self, other: Token | SpacyToken | Span | SpacySpan) -> bool:
        if isinstance(other, SpacyToken | SpacySpan):
            return other in self.tok
        if isinstance(other, Token | Span):
            return other.tok in self.tok
        ocn = other.__class__.__name__
        scn = self.__class__.__name__
        raise TypeError(f"'{scn}' cannot contain '{ocn}' objects")

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
        if self._id is None:
            string = json.dumps(
                self.coredata, check_circular=False, indent=None,
                separators=(",", ":"), sort_keys=True
            )
            self._id = hash_unicode(string)
        return self._id

    @property
    def coredata(self) -> dict[str, Any]:
        fields = (
            "words", "spaces", "tags", "pos",
            "morphs", "lemmas", "heads", "deps", "ents"
        )
        meta = getattr(self._, f"{settings.spacy_alias}_meta")
        data = { k: v for k, v in self.data.items() if k in fields }
        return { "meta": meta, "data": data }

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

    @property
    def grammar(self) -> "Doc":
        return getattr(self._, f"{settings.spacy_alias}_doc")

    # Methods -----------------------------------------------------------------

    def to_data(self) -> dict[str, Any]:
        """Dump to data dictionary sufficient to recreate simple document
        without any language model data.
        """
        user_data = self.tok.user_data.copy()
        alias = settings.spacy_alias
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
        user_data[("._.", f"{alias}_cache", None, None)] = {}
        user_data[("._.", f"{alias}_doc", None, None)] = None
        return data

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> Self:
        """Construct from data dictionary produced by :meth:`to_data`."""
        return getattr(SpacyDoc(**data)._, settings.spacy_alias)

    def char_span(self, *args: Any, **kwds: Any) -> Span | None:
        res = self.tok.char_span(*args, **kwds)
        return res if res is None else self.sns(res)

    @classmethod
    def from_docs(cls, *args: Any, **kwds: Any) -> Self | None:
        res = Doc.from_docs(*args, **kwds)
        return res if res is None else cls.sns(res)

    def copy(self) -> SpacyDoc:
        return self.sns(self.tok.copy())

    def get_grammar_type(self):
        alias = settings.spacy_alias
        key = getattr(self._, f"{alias}_meta")[f"{alias}_doc"]
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
