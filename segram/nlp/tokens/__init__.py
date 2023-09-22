"""Wrapper classes for injecting :mod:`segram`
functionalities into :mod:`spacy` tokens.

They implemented using composition by wrapping
around a :mod:`spacy` token object.
"""
from .doc import Doc
from .span import Span
from .token import Token
