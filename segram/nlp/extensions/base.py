"""Default :mod:`spacy` extension backend."""
# pylint: disable=protected-access
from typing import ClassVar, Mapping, Union
from types import MappingProxyType
from functools import partial
from spacy.tokens import Doc as SpacyDoc, Span as SpacySpan, Token as SpacyToken
from ..tokens import Doc, Span, Token
from ... import __title__


class SpacyExtensions:
    """Backend providing implementations
    of base custom :mod:`spacy` extensions attributes.

    Attributes
    ----------
    doc
        Enhanced document type.
    span
        Enhanced span type.
    token
        Enhanced token type.
    attributes
        Specification of extension attributes to register.
    alias
        :mod:`segram` alias.
    """
    __spacy_token_types__: ClassVar[Mapping[str, type]] = MappingProxyType({
        "token": SpacyToken,
        "span": SpacySpan,
        "doc": SpacyDoc
    })
    __attributes__: ClassVar[dict[str, dict]] = {
        "token": {
            "corefs": { "default": None },
        },
        "doc": {
            "meta": { "default": None },   # Segram metadata dictionary
            "doc": { "default": None },    # Segram grammar document pointer
            "data": { "default": None },   # Serialized Segram grammar data
            "numpy": { "default": None },  # Numpy/Cupy pointer
        }
    }

    def __init__(
        self,
        doc: type[Doc],
        span: type[Span],
        token: type[Token],
        alias: str
    ) -> None:
        self.doc = doc
        self.span = span
        self.token = token
        self.alias = alias

    # Methods -----------------------------------------------------------------

    def register(self) -> None:
        """Initialize extensions."""
        alias = self.alias
        tok_types = self.__class__.__spacy_token_types__
        for typ, attrs in self.__attributes__.items():
            for attr, kwds in attrs.items():
                if attr.startswith("_"):
                    name = f"_{alias}{attr[1:]}"
                else:
                    name = f"{alias}_{attr}"
                tok_types[typ].set_extension(name, **kwds)
        # Register SNS getters and keys
        tok_types["doc"].set_extension(__title__+"_alias", default=None)
        tok_types["doc"].set_extension(alias, getter=self.grammar)
        tok_types["span"].set_extension(alias, getter=self.grammar)
        alias += "_sns"
        for attr, spacy in tok_types.items():
            segram = getattr(self, attr)
            spacy.set_extension("_"+alias, default=None)
            spacy.set_extension(alias, getter=partial(self.sns_get, typ=segram))

    # Doc extension attributes ------------------------------------------------

    @staticmethod
    def sns_get(
        tok: SpacyDoc | SpacySpan | SpacyToken,
        typ: type[Token]
    ) -> Token:
        alias = "_"+getattr(tok.doc._, __title__+"_alias")+"_sns"
        if (obj := getattr(tok._, alias)):
            return obj
        obj = typ(tok)
        setattr(tok._, alias, obj)
        return obj

    @staticmethod
    def grammar(tok: SpacyDoc | SpacySpan) -> Union["Doc", "Span"]:
        alias = getattr(tok.doc._, __title__+"_alias")
        return getattr(tok._, alias+"_sns").grammar
