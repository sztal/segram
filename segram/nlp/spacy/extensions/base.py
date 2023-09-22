"""Default :mod:`spacy` extension backend."""
# pylint: disable=protected-access
from __future__ import annotations
from typing import Type, ClassVar, Mapping
from types import MappingProxyType
from functools import partial
from spacy.tokens import Doc, Span, Token
from ..tokens import SpacyNLPToken
from ..tokens import SpacyDoc, SpacySpan, SpacyTokenABC
from .... import settings


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
        "token": Token,
        "span": Span,
        "doc": Doc
    })
    __attributes__: ClassVar[dict[str, dict]] = {
        "token": {
            "corefs": { "default": None },
        },
        "doc": {
            "meta": { "default": None },
            "cache": { "default": None },
            "grammar_data": { "default": None }
        }
    }

    def __init__(
        self,
        doc: Type[SpacyDoc],
        span: Type[SpacySpan],
        token: Type[SpacyTokenABC]
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
        Doc.set_extension(alias, getter=partial(self.sns_getter, typ=self.doc))
        Span.set_extension(alias, getter=partial(self.sns_getter, typ=self.span))
        Token.set_extension(alias, getter=partial(self.sns_getter, typ=self.token))

    # Doc extension attributes ------------------------------------------------

    @staticmethod
    def sns_getter(
        tok: Doc | Span | Token,
        typ: Type[SpacyNLPToken]
    ) -> SpacyNLPToken:
        cache = getattr(tok.doc._, f"{settings.spacy_alias}_cache")
        if isinstance(tok, Token):
            key = tok.i
        elif isinstance(tok, Span):
            key = (tok.start, tok.end)
        else:
            key = -1
        sns = cache.get(key)
        if sns is None:
            sns = typ(tok)
            cache[key] = sns
        return sns
