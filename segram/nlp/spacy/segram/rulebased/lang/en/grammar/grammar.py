from .......grammar import GrammarNLP
from ........grammar.lang.en import EnglishGrammar

__backend__ = __name__.rsplit(".", maxsplit=7)[1]
__method__ = __name__.rsplit(".", maxsplit=5)[1]
__lang__ = __name__.rsplit(".", maxsplit=3)[1]

class SpacyRulebasedEnglishGrammar(
    EnglishGrammar, GrammarNLP,
    register=f"{__backend__}.{__method__}.{__lang__}"
):
    """Abstract base class for :mod:`spacy` rule-based English grammar."""
    __slots__ = ()
