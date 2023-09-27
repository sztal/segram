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
    def grammar(self) -> "Doc":
        if (doc := getattr(self._, f"{settings.spacy_alias}_doc")):
            return doc
        typ = self.get_grammar_type()
        return typ.types.Doc(self)

    # Methods -----------------------------------------------------------------

    @staticmethod
    def clear_user_data(user_data: dict):
        """Clear user data from cached :mod:`segram` objects."""
        alias = settings.spacy_alias
        _alias = "_"+alias
        for k, v in user_data.items():
            user_data[k] = v if _alias not in k else None
        user_data[("._.", f"{alias}_doc", None, None)] = None
        return user_data


    def to_data(self) -> dict[str, Any]:
        """Dump to data dictionary sufficient to recreate simple document
        without any language model data.
        """
        data = {
            "vocab": self.vocab,
            "words": [ t.text for t in self ],
            "spaces": [ t.whitespace for t in self ],
            "tags": [ t.tag_ for t in self.tok ],
            "pos": [ t.pos_ for t in self.tok ],
            "morphs": [ str(t.morph) for t in self.tok ],
            "lemmas": [ t.lemma_ for t in self.tok ],
            "heads": [ t.head.i for t in self.tok ],
            "deps": [ t.dep_ for t in self.tok ],
            "ents": [ f"{t.ent_tag}" for t in self ]
        }
        data["user_data"] = self.clear_user_data(self.tok.user_data.copy())
        return data

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> Self:
        """Construct from data dictionary produced by :meth:`to_data`."""
        return getattr(SpacyDoc(**data)._, settings.spacy_alias+"_sns")

    def char_span(self, *args: Any, **kwds: Any) -> Span | None:
        res = self.tok.char_span(*args, **kwds)
        return res if res is None else self.sns(res)

    @classmethod
    def from_docs(cls, *args: Any, **kwds: Any) -> Self | None:
        res = Doc.from_docs(*args, **kwds)
        return res if res is None else cls.sns(res)

    def get_grammar_type(self):
        alias = settings.spacy_alias
        key = getattr(self._, f"{alias}_meta")[f"{alias}_doc"]
        return grammars.get(key)

    def copy(self) -> Self:
        return self.from_data(self.to_data())


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
