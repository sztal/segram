from .grammar import EnglishGrammar
from ... import Phrase, VerbPhrase, NounPhrase, DescPhrase, PrepPhrase


class EnglishPhrase(EnglishGrammar, Phrase):
    """English phrase."""
    __slots__ = ()


class EnglishVerbPhrase(EnglishGrammar, VerbPhrase):
    """English verb phrase."""
    __slots__ = ()


class EnglishNounPhrase(EnglishGrammar, NounPhrase):
    """English noun phrase."""
    __slots__ = ()


class EnglishDescPhrase(EnglishGrammar, DescPhrase):
    """English descriptive phrase."""
    __slots__ = ()


class EnglishPrepPhrase(EnglishGrammar, PrepPhrase):
    """English prepositional phrase."""
    __slots__ = ()
