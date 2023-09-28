"""Module-level configuration."""
import pytest


@pytest.fixture(scope="session")
def nlp(spacy):
    model = spacy.load("en_core_web_trf")
    model.add_pipe("segram", config={
        "vectors": None
    })
    return model
