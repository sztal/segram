from typing import NamedTuple
from collections import Counter
from .base import NLP


class TokenDistributions(NamedTuple):
    text: Counter = Counter()
    lemma: Counter = Counter()


class VocabABC(NLP):
    """Vocabulary abstract base class.

    Attributes
    ----------
    corpus
        Corpus the vocabulary is based on.
    lexeme
        Lexical store mapping strings to lexemes.
    dist
        Named tuple with ``text`` frequency distribution
        of raw token texts and ``lemma`` distribution
        of token lemmas.
    """
    __slots__ = ("corpus", "lexeme", "dist")

    def __init__(self, corpus: "CorpusABC") -> None:
        self.corpus = corpus
        self.lexeme = {}
        self.dist = TokenDistributions()
