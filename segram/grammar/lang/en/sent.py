from .grammar import EnglishGrammar
from ... import Sent


class EnglishSent(EnglishGrammar, Sent):
    """English sentence element."""
    __slots__ = ()
