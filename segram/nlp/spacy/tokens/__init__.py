"""Wrapper classes for injecting :mod:`segram`
functionalities into :mod:`spacy` tokens.

They implemented using composition by wrapping
around a :mod:`spacy` token object.
"""
from .abc import SpacyNLPToken
from .doc import SpacyDoc
from .span import SpacySpan
from .token import SpacyTokenABC
