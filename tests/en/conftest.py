"""Module-level configuration."""
import pytest
import spacy


@pytest.fixture(scope="session")
def nlp():
    spacy.prefer_gpu()
    model = spacy.load("en_core_web_trf")
    model.add_pipe("segram", config={
        "store_data": "none"
    })
    return model
