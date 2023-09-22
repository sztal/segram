from .abc import CorpusABC
from .vocab import Vocab


class Corpus(CorpusABC):
    """Minimal vocabulary class allowing interoperability
    with grammar classes.
    """
    def __init__(self) -> None:
        super().__init__()
        self._vocab = Vocab(self)
