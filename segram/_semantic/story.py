# from __future__ import annotations
# from typing import Any, Iterator
# from collections import ChainMap
# from collections.abc import Iterable, Sequence, Mapping
# from .abc import Semantic
# from .frame import Frame
# from .elements import FrameElement
# from .elements import Actor, Action, Description, Complement
# from ..grammar import Sent, Phrase, Conjuncts
# from ..nlp import DocABC
# from ..utils.types import ChainGroup


# class Story(Sequence, Semantic):
#     """Story semantic class."""
#     __slots__ = ("_doc", "frames", "emap", "_pmap")

#     def __init__(
#         self,
#         doc: DocABC,
#         frames: Iterable[Frame] = ()
#     ) -> None:
#         self._doc = doc
#         self.frames = tuple(frames)
#         self.emap = {}
#         self._pmap = ChainMap()

#     def __repr__(self) -> str:
#         nframes = len(self)
#         text = "frame" if nframes == 1 else "frames"
#         return f"<{self.ppath()} with {nframes} {text} at {hex(id(self))}>"

#     def __hash__(self) -> int:
#         return super().__hash__()

#     def __eq__(self, other: Story) -> bool:
#         if (res := super().__eq__(other)) is NotImplemented:
#             return res
#         return res and self.frames == other.frames

#     def __len__(self) -> int:
#         return len(self.frames)

#     def __getitem__(self, idx: int | slice) -> Frame | tuple[Frame, ...]:
#         return self.frames[idx]

#     # Properties --------------------------------------------------------------

#     @property
#     def doc(self) -> DocABC:
#         return self._doc

#     @property
#     def hashdata(self) -> tuple[Any, ...]:
#         return (*super().hashdata, id(self))

#     @property
#     def pmap(self) -> Mapping[int, Phrase]:
#         return self._pmap

#     @property
#     def elements(self) -> Iterator[FrameElement]:
#         for frame in self:
#             yield from frame.elements

#     @property
#     def actors(self) -> ChainGroup:
#         return Conjuncts.get_chain(
#             e for e in self.elements
#             if isinstance(e, Actor)
#         )

#     @property
#     def actions(self) -> ChainGroup:
#         return Conjuncts.get_chain(
#             e for e in self.elements
#             if isinstance(e, Action)
#         )

#     @property
#     def descriptions(self) -> ChainGroup:
#         return Conjuncts.get_chain(
#             e for e in self.elements
#             if isinstance(e, Description)
#         )

#     @property
#     def complements(self) -> ChainGroup:
#         return Conjuncts.get_chain(
#             e for e in self.elements
#             if isinstance(e, Complement)
#         )

#     # Methods -----------------------------------------------------------------

#     def copy(self, **kwds: Any) -> Story:
#         return self.__class__(**{ "doc": self.doc, **self.data, **kwds })

#     def is_comparable_with(self, other: Story) -> bool:
#         return isinstance(other, Story)

#     def to_str(self, **kwds: Any) -> str:
#         # pylint: disable=unused-argument
#         return super().__repr__()

#     @classmethod
#     def from_sents(cls, doc: DocABC, sents: Iterable[Sent]) -> Story:
#         """Construct from a document and sequence of grammar sentences."""
#         obj = cls(doc)
#         obj.frames = tuple(Frame(obj, sent) for sent in sents)
#         return obj
