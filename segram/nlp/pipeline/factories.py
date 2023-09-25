"""Factory functions for :mod:`spacy` pipeline components."""
from typing import Sequence
from spacy.language import Language
from .base import Segram
from ... import __title__


# Language factory (core) -----------------------------------------------------

@Language.factory(
    name=__title__,
    default_config={
        "grammar": "rulebased",
        "preprocess": ["lemmatizer", "merger"],
        "alias": __title__,
        "lazy_data": True,
        "vectors": None
    }
)
def create_base(
    nlp: Language,
    name: str,
    *,
    grammar: str,
    preprocess: Sequence,
    alias: str,
    lazy_data: bool,
    vectors: str | Language | None
) -> Segram:
    return Segram(
        nlp=nlp,
        name=name,
        grammar=grammar,
        preprocess=preprocess,
        alias=alias,
        lazy_data=lazy_data,
        vectors=vectors
    )
