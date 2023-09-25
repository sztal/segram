from .abc import DocElement
from ..nlp.tokens import Doc as DocNLP
from .. import settings


class Doc(DocElement):
    """Grammar document class.

    This is grammar equivalent of NLP documents.

    Attributes
    ----------
    doc
        Underlying NLP document.
    """
    __slots__ = ("doc",)
    alias = "Doc"

    def __init__(self, doc: DocNLP) -> None:
        setattr(doc._, f"{settings.spacy_alias}_grammar", self)
        super().__init__(doc)
