"""English pipeline components factories."""
from typing import Sequence
from spacy.lang.en import English
from .merger import EnglishMerger as Merger
from .lemmatizer import EnglishLemmatizer as Lemmatizer
from ......pipeline.coref import Coref
from ........ import __title__, settings


# Lemmatizer factory ----------------------------------------------------------

@English.factory(
    name=f"{__title__}_lemmatizer",
    requires=("token.pos",),
    assigns=("token.lemma",),
    default_config={
        "patterns": {
            "package": f"{__package__}.lemmatizer",
            "filename": "patterns.json"
        },
        "validate": False
    }
)
def create_en_lemmatizer(
    nlp: English,
    name: str,
    patterns: dict[str, str],
    validate: bool
) -> Lemmatizer:
    return Lemmatizer.from_patterns(nlp, name, patterns=patterns, validate=validate)


# Merger factory --------------------------------------------------------------

@English.factory(
    name=f"{__title__}_merger",
    requires=("token.pos", "token.dep"),
    assigns=("token._.multitoken",),
    retokenizes=True,
    default_config={
        "patterns": {
            "package": f"{__package__}.merger",
            "filename": "patterns.json"
        },
        "validate": False
    }
)
def create_en_merger(
    nlp: English,
    name: str,
    patterns: dict[str, str],
    validate: bool
) -> Merger:
    return Merger.from_patterns(nlp, name, patterns=patterns, validate=validate)


# Coref factory ---------------------------------------------------------------

@English.factory(
    name=f"{__title__}_coref",
    requires=("token.pos",),
    assigns=(f"token._.{settings.spacy_alias}_corefs",),
    default_config={
        "model": "en_coreference_web_trf",
        "components": ["sentencizer", "transformer", "coref"]
    }
)
def create_en_coref(
    nlp: English,
    name: str,
    model: str,
    components: Sequence[str]
) -> Coref:
    return Coref.from_model(nlp, name, model, components)
