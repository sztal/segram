"""Default :mod:`spacy` extension backend."""
# pylint: disable=protected-access
from typing import ClassVar, Mapping
from types import MappingProxyType
from functools import partial
from spacy.tokens import Doc as SpacyDoc, Span as SpacySpan, Token as SpacyToken
from ..tokens.abc import NLP
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
    __extension_types__: ClassVar[tuple[str, ...]] = \
        ("method", "getter", "getter_cached")
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
            "meta": { "default": None },
            "cache": { "default": None },
            "doc": { "default": None },
            "data": { "default": None },
            "model": { "default": None }
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
        SpacyDoc.set_extension(alias, getter=partial(self.sns_getter, typ=self.doc))
        SpacySpan.set_extension(alias, getter=partial(self.sns_getter, typ=self.span))
        SpacyToken.set_extension(alias, getter=partial(self.sns_getter, typ=self.token))

    # Doc extension attributes ------------------------------------------------

    @staticmethod
    def sns_getter(
        tok: Doc | Span | Token,
        typ: type[Doc] | type[Span] | type[Token]
    ) -> NLP:
        cache = getattr(tok.doc._, f"{settings.spacy_alias}_cache")
        if isinstance(tok, SpacyToken):
            key = tok.i
        elif isinstance(tok, SpacySpan):
            key = (tok.start, tok.end)
        else:
            key = -1
        sns = cache.get(key)
        if sns is None:
            sns = typ(tok)
            cache[key] = sns
        return sns
