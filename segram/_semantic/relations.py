# # pylint: disable=redefined-builtin
# from typing import Any, Iterator, Optional
# from collections.abc import Sequence
# from functools import total_ordering
# from ..grammar import Component, Phrase
# from ..symbols import Role
# from ..nlp import TokenABC
# from ..abc import SegramABC


# @total_ordering
# class Relation(Sequence, SegramABC):
#     """Semantic relation base class."""
#     __slots__ = ("head", "description", "complement")

#     def __init__(
#         self,
#         head: Component,
#         *,
#         description: Optional[Phrase] = None,
#         complement: Optional[Phrase] = None
#     ) -> None:
#         self.head = head
#         self.description = description
#         self.complement = complement

#     def __repr__(self) -> str:
#         return self.to_str(color=True)

#     def __hash__(self) -> int:
#         return hash(tuple(self))

#     def __eq__(self, other: Sequence) -> bool:
#         if self.is_comparable_with(other):
#             return tuple(self) == other
#         return NotImplemented

#     def __lt__(self, other: Sequence) -> bool:
#         if self.check_comparable(other):
#             return tuple(self) < other
#         return NotImplemented

#     def __len__(self) -> str:
#         return len(self.slot_names)

#     def __getitem__(self, idx: int | slice) -> Any | tuple[Any, ...]:
#         name = self.slot_names[idx]
#         if isinstance(name, str):
#             return getattr(self, name)
#         return tuple(getattr(self, n) for n in name)

#     def __init_subclass__(cls) -> None:
#         super().__init_subclass__()
#         cls.init_class_attrs({
#             "__slots__": "slot_names"
#         }, check_slots=True)

#     def to_str(self, *, color: bool = False, **kwds: Any) -> str:
#         """Represent as string."""
#         return " ".join(
#             t.to_str(color=color, role=r, **kwds)
#             for t, r in self.iter_token_roles()
#         )

#     def iter_token_roles(self) -> Iterator[tuple[TokenABC, Role | None]]:
#         """Iterate over token-role pairs."""
#         def _iter():
#             for name in self.slot_names:
#                 if (part := getattr(self, name)):
#                     yield from part.iter_token_roles()
#         yield from sorted(set(_iter()))

#     def is_comparable_with(self, other: Any) -> bool:
#         return isinstance(other, Sequence)


# class ActorRelation(Relation):
#     """Actor relation."""
#     __slots__ = ()


# class ActionRelation(Relation):
#     """Action relation."""
#     __slots__ = ("subject", "dobject", "iobject")

#     def __init__(
#         self,
#         head: Component,
#         *,
#         subject: Optional[Phrase] = None,
#         dobject: Optional[Phrase] = None,
#         iobject: Optional[Phrase] = None,
#         **kwds: Any
#     ) -> None:
#         super().__init__(head, **kwds)
#         self.subject = subject
#         self.dobject = dobject
#         self.iobject = iobject


# class DescriptionRelation(Relation):
#     """Description relation."""
#     __slots__ = ("object", "verb")

#     def __init__(
#         self,
#         head: Component,
#         *,
#         object: Optional[Phrase] = None,
#         verb: Optional[Component] = None,
#         **kwds: Any
#     ) -> None:
#         super().__init__(head, **kwds)
#         self.object = object
#         self.verb = verb


# class ComplementRelation(Relation):
#     """Complement relation."""
#     __slots__ = ("object",)

#     def __init__(
#         self,
#         head: Component,
#         *,
#         object: Optional[Phrase] = None,
#         **kwds: Any
#     ) -> None:
#         super().__init__(head, **kwds)
#         self.object = object
