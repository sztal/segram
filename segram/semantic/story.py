from typing import Any, Self
from .abc import Semantic
from ..abc import SegramWithDocABC
from ..nlp.tokens import DocABC


class Story(Semantic, SegramWithDocABC):
    """Semantic story class.

    Attributes
    ----------
    doc
        Document object.
    defs
        Definitions of semantic elements in the story.
    """
    __slots__ = ("_doc", "_defs")

    def __init__(self, doc: DocABC) -> None:
        self._doc = doc
        self._defs = ()

    # Properties --------------------------------------------------------------

    @property
    def doc(self) -> DocABC:
        return self._doc

    @property
    def hashdata(self) -> tuple[Any, ...]:
        return (*super().hashdata, id(self))

    # Methods -----------------------------------------------------------------

    def copy(self, **kwds: Any) -> Self:
        return self.__class__(**{ "doc": self.doc, **self.data, **kwds })

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, Story)
