from .grammar import RulebasedEnglishGrammar
from ......grammar import PhraseNLP
from .......grammar.lang.en import EnglishPhrase
from .......grammar.lang.en import EnglishVerbPhrase, EnglishNounPhrase
from .......grammar.lang.en import EnglishDescPhrase, EnglishPrepPhrase



class RulebasedEnglishPhrase(
    RulebasedEnglishGrammar,
    EnglishPhrase, PhraseNLP
):
    """Rule-based English :mod:`spacy` phrase."""
    __slots__ = ()


class RulebasedEnglishVerbPhrase(
    RulebasedEnglishPhrase, EnglishVerbPhrase
):
    """Rule-based English :mod:`spacy` verb phrase."""
    __slots__ = ()


class RulebasedEnglishNounPhrase(
    RulebasedEnglishPhrase, EnglishNounPhrase
):
    """Rule-based English :mod:`spacy` noun phrase."""
    __slots__ = ()


class RulebasedEnglishDescPhrase(
    RulebasedEnglishPhrase, EnglishDescPhrase
):
    """Rule-based English :mod:`spacy` desc phrase."""
    __slots__ = ()


class RulebasedEnglishPrepPhrase(
    RulebasedEnglishPhrase, EnglishPrepPhrase
):
    """Rule-based English :mod:`spacy` prep phrase."""
    __slots__ = ()
