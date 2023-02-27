"""Generic NLP backend.

Generic classes providing minimal implementations
of third-party classes used by NLP backends required
for interoperability with :mod:`segram.nlp.grammar`.

All other backend need to provide token, span and doc
classes comaptibile with the minimal generic interface
defined here.
"""
from .abc import DocABC, SpanABC, TokenABC
from .doc import Doc
from .span import Span
from .token import Token, TokenData
