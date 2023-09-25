"""NLP backends.

NLP backends allows decoupling of logic specific for different
languages and NLP engines (e.g. :mod:`spacy`) from the rest of
the package concerned with processing the output produced by
a backend.
"""
from .corpus import Corpus
from .tokens import Token, Span, Doc
