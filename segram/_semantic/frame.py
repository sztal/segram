# from __future__ import annotations
# from typing import Any, Iterator
# from types import MappingProxyType
# from functools import total_ordering
# from .abc import Semantic
# from .relations import Relation
# from .elements import FrameElement, Actor, Action, Description, Complement
# from ..grammar import Sent, Conjuncts
# from ..nlp import DocABC
# from ..utils.types import ChainGroup


# @total_ordering
# class Frame(Semantic):
#     """Semantic frame class.

#     Attributes
#     ----------
#     sent
#         Grammar sentence.
#     """
#     __slots__ = ("_story", "_sent")

#     def __init__(
#         self,
#         story: "Story",
#         sent: Sent
#     ) -> None:
#         self._story = story
#         self._sent = sent
#         self.story.pmap.maps.append(MappingProxyType(self.sent.pmap))

#     def __hash__(self) -> int:
#         return super().__hash__()

#     def __eq__(self, other: Frame) -> bool:
#         if self.is_comparable_with(other):
#             return self.sent == other.sent
#         return NotImplemented

#     def __lt__(self, other: Frame) -> bool:
#         if self.is_comparable_with(other):
#             return self.sent < other.sent
#         return NotImplemented

#     # Properties --------------------------------------------------------------

#     @property
#     def doc(self) -> DocABC:
#         return self.sent.doc

#     @property
#     def sent(self) -> Sent:
#         return self._sent

#     @property
#     def story(self) -> "Story":
#         return self._story

#     @property
#     def hashdata(self) -> int:
#         return (*super().hashdata, self.story, self.sent)

#     @property
#     def sources(self) -> ChainGroup:
#         return (sources := self.sent.sources).copy(
#             chain=tuple(g.copy(members=tuple(
#                 FrameElement.from_phrase(self, m) for m in g.members
#             )) for g in sources.chain)
#         )

#     @property
#     def elements(self) -> Iterator[FrameElement]:
#         seen = set()
#         for source in self.sent.sources:
#             for phrase in source.subtree:
#                 if phrase in seen:
#                     continue
#                 seen.add(phrase)
#                 yield FrameElement.from_phrase(self, phrase)

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
#             if isinstance(e, Action) and not e.phrase.adesc
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

#     def is_comparable_with(self, other: Any):
#         return isinstance(other, Frame)

#     def iter_relations(self, **kwds: Any) -> Iterator[Relation]:
#         """Iterate over semantic relations."""
#         for elem in self.elements:
#             yield from elem.iter_relations(**kwds)
