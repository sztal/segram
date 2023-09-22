from .abc import VocabABC


class Vocab(VocabABC):
    """Minimal vocabulary class allowing interoperability
    with grammar classes.
    """
    def __init__(self, corpus: "Corpus") -> None:
        super().__init__(corpus)
