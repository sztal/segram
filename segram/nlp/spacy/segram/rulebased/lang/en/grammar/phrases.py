from .grammar import SpacyRulebasedEnglishGrammar
from .......grammar import PhraseNLP
from ........grammar.lang.en import EnglishPhrase
from ........grammar.lang.en import EnglishVerbPhrase, EnglishNounPhrase
from ........grammar.lang.en import EnglishDescPhrase, EnglishPrepPhrase



class SpacyRulebasedEnglishPhrase(
    SpacyRulebasedEnglishGrammar,
    EnglishPhrase, PhraseNLP
):
    """Rule-based English :mod:`spacy` phrase."""
    __slots__ = ()


class SpacyRulebasedEnglishVerbPhrase(
    SpacyRulebasedEnglishPhrase, EnglishVerbPhrase
):
    """Rule-based English :mod:`spacy` verb phrase."""
    __slots__ = ()


class SpacyRulebasedEnglishNounPhrase(
    SpacyRulebasedEnglishPhrase, EnglishNounPhrase
):
    """Rule-based English :mod:`spacy` noun phrase."""
    __slots__ = ()


class SpacyRulebasedEnglishDescPhrase(
    SpacyRulebasedEnglishPhrase, EnglishDescPhrase
):
    """Rule-based English :mod:`spacy` desc phrase."""
    __slots__ = ()


class SpacyRulebasedEnglishPrepPhrase(
    SpacyRulebasedEnglishPhrase, EnglishPrepPhrase
):
    """Rule-based English :mod:`spacy` prep phrase."""
    __slots__ = ()
