"""Grammar layer."""
from .abc import Grammar
from .components import Component, Verb, Noun, Prep, Desc
from .phrases import Phrase, VerbPhrase, NounPhrase, DescPhrase, PrepPhrase
from .sent import Sent
from .graph import PhraseGraph
from .conjuncts import Conjuncts
