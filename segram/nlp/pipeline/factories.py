"""Factory functions for :mod:`spacy` pipeline components."""
from typing import Optional, Sequence
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
        "store_data": True,
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
    store_data: bool,
    vectors: Optional[str | Language]
) -> Segram:
    return Segram(
        nlp=nlp,
        name=name,
        grammar=grammar,
        preprocess=preprocess,
        alias=alias,
        store_data=store_data,
        vectors=vectors
    )
