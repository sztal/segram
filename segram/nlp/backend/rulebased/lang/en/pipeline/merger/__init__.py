"""English merger module and patterns data.

Patterns for merging multitokens.

phrasal_verb
    Currently only detection of two-component phrasal verbs
    (VERB + PARTICLE) is supported.
"""
from .......pipeline.merger import Merger


class EnglishMerger(Merger):
    """English merger pipeline component.

    See also
    --------
    segram.nlp.pipeline.merger.Merger : Base merger component.
    """
