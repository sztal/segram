"""Default :mod:`spacy` extension backend."""
# pylint: disable=protected-access
from typing import ClassVar, Mapping, Union
from types import MappingProxyType
from functools import partial
from spacy.tokens import Doc as SpacyDoc, Span as SpacySpan, Token as SpacyToken
from ..tokens import Doc, Span, Token
from ... import settings


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
        }
    }

    def __init__(
        self,
        doc: type[Doc],
        span: type[Span],
        token: type[Token]
    ) -> None:
        self.doc = doc
        self.span = span
        self.token = token

    # Methods -----------------------------------------------------------------

    def register(self) -> None:
        """Initialize extensions."""
        alias = settings.spacy_alias
        tok_types = self.__class__.__spacy_token_types__
        for typ, attrs in self.__attributes__.items():
            for attr, kwds in attrs.items():
                if attr.startswith("_"):
                    name = f"_{alias}{attr[1:]}"
                else:
                    name = f"{alias}_{attr}"
                tok_types[typ].set_extension(name, **kwds)
        # Register SNS getters and keys
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
        alias = "_"+settings.spacy_alias+"_sns"
        if (obj := getattr(tok._, alias)):
            return obj
        obj = typ(tok)
        setattr(tok._, alias, obj)
        return obj

    @staticmethod
    def grammar(tok: SpacyDoc | SpacySpan) -> Union["Doc", "Span"]:
        return getattr(tok._, settings.spacy_alias+"_sns").grammar
