"""Factory functions for :mod:`spacy` pipeline components."""
from typing import Sequence, Literal
from spacy.language import Language
from .base import Segram
from .... import __title__


# Language factory (core) -----------------------------------------------------

@Language.factory(
    name=__title__,
    default_config={
        "grammar": "rulebased",
        "preprocess": ["lemmatizer", "merger"],
        "alias": __title__,
        "store_data": "grammar"
    }
)
def create_base(
    nlp: Language,
    name: str,
    *,
    grammar: str,
    preprocess: Sequence,
    alias: str,
    store_data: Literal[Segram.__store_data__]
) -> Segram:
    return Segram(
        nlp=nlp,
        name=name,
        grammar=grammar,
        preprocess=preprocess,
        alias=alias,
        store_data=store_data
    )
