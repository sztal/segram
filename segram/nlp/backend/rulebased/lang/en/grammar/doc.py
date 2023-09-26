from .grammar import RulebasedEnglishGrammar
from ......grammar import DocNLP
from .......grammar.lang.en import EnglishDoc


class RulebasedEnglishDoc(
    RulebasedEnglishGrammar,
    EnglishDoc, DocNLP
):
    """Rule-based :mod:`spacy` English document element."""
    __slots__ = ()
