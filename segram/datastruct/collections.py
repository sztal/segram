"""Enhanced :mod:`collections.abc` classes implementing
generic data filtering and transformation method.
"""
from typing import Any, Iterable, Collection, Sequence
from abc import abstractmethod
from functools import total_ordering
from ..abc import SegramABC


class DataCollection(Collection, SegramABC):
    """Data colelction class."""
    __slots__ = ("members",)

    @abstractmethod
    def __init__(self, members: Collection[Any]) -> None:
        """Initialization method."""
        self.members = members

    def __eq__(self, other: Any) -> bool:
        if not self.is_comparable_with(other):
            return NotImplemented
        if isinstance(other, DataCollection):
            return self.members == other.members
        return self.members == other

    def __hash__(self) -> int:
        return super().__hash__()

    def __iter__(self) -> Iterable[Any]:
        yield from self.members

    def __len__(self) -> int:
        return len(self.members)

    def __contains__(self, other: Any) -> bool:
        return other in self.members

    def __repr__(self) -> str:
        return repr(self.members)

    @property
    def hashdata(self) -> tuple[Any, ...]:
        return (self.members,)

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, Collection)


@total_ordering
class DataSequence(DataCollection, Sequence):
    """Data sequence class."""
    __slots__ = ()

    def __init__(self, members: Sequence[Any]) -> None:
        self.members = members

    def __getitem__(self, idx: int | slice) -> Any:
        return self.members[idx]

    def __lt__(self, other: Sequence) -> bool:
        if not self.is_comparable_with(other):
            return NotImplemented
        if isinstance(other, DataSequence):
            return self.members < other.members
        return self.members < other

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(self, Sequence)


class DataGrouped(DataSequence):
    """Data grouped sequence class."""
    __slots__ = ()

    def __init__(self, members: Sequence[Collection[Any]]) -> None:
        super().__init__(members)

    def __iter__(self) -> Iterable[Any]:
        yield from self.flat

    def __getitem__(self, idx: int | slice) -> Any:
        return tuple(self.flat)[idx]

    def __len__(self) -> int:
        return sum(1 for _ in self.flat)

    @property
    def flat(self) -> DataSequence[Any]:
        return DataSequence(tuple(self.flatten()))

    @property
    def groups(self) -> DataSequence:
        return self.members

    def flatten(self) -> DataSequence[Any]:
        """Flatten groups.

        The method assumes that nesting is represented
        by instances of subclasses of :class:`DataGrouped`.
        """
        for member in self.members:
            if isinstance(member, DataGrouped):
                yield from member.flat
            else:
                yield from member
