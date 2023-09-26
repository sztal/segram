from .grammar import EnglishGrammar
from ... import Doc


class EnglishDoc(EnglishGrammar, Doc):
    """English document element."""
    __slots__ = ()
