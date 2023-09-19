from __future__ import annotations
from typing import Any
from functools import total_ordering
from collections.abc import Mapping, MutableMapping
from collections.abc import Sequence, Iterable, Iterator
from graphlib import TopologicalSorter, CycleError
from .meta import get_cname
from ..abc import SegramABC


class Namespace(MutableMapping):
    """Namespace class.

    It behaves like a dictionary with both item and attribute
    getters, setters and deletters.
    """
    def __init__(self, *args: Any, **kwds: Any) -> None:
        try:
            dct = dict(*args, **kwds)
        except TypeError as exc:
            raise TypeError(f"'{get_cname(self)}' {str(exc)}") from exc
        self.__dict__.update(dct)

    def __repr__(self) -> str:
        return f"{get_cname(self)}({self.__dict__})"

    def __iter__(self) -> Iterator[str]:
        yield from self.__dict__

    def __len__(self) -> int:
        return len(self.__dict__)

    def __getitem__(self, name: str) -> Any :
        return self.__dict__[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value

    def __delitem__(self, name: str) -> None:
        del self.__dict__[name]

    def __contains__(self, name: str) -> bool:
        return name in self.__dict__

    @property
    def names(self) -> list[str]:
        return list(self)


@total_ordering
class Group(Sequence, SegramABC):
    """Group of arbitrary coordinated objects.

    Subclasses can define additional slots corresponding
    to coordinating objects (e.g. coordinating conjunctions),
    which can be jointly returned as a single tuple through
    ``cconjs`` property, in which case they will be included
    automatically in comparison methods and ``__repr__()``.

    Attributes
    ----------
    members
        Sequence of objects.
        Stored as a tuple, so it is not mutable and can be safely hashed
        if the stored objects are hashable themselves.
    """
    __slots__ = ("members",)

    def __init__(self, members: Iterable[Any] = ()) -> None:
        self.members = tuple(members)

    def __repr__(self) -> str:
        return self.to_str()

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Any) -> bool:
        if self.is_comparable_with(other):
            return tuple(self) == tuple(other)
        return NotImplemented

    def __lt__(self, other: Any) -> bool:
        if self.is_comparable_with(other):
            return tuple(self) < tuple(other)
        return NotImplemented

    def __getitem__(self, idx: int | slice) -> tuple[Any, ...]:
        return self.members[idx]

    def __len__(self) -> int:
        return len(self.members)

    # Properties --------------------------------------------------------------

    @property
    def hashdata(self) -> tuple[Any, ...]:
        return (self.members,)

    # Methods -----------------------------------------------------------------

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, Sequence)

    def to_str(self, **kwds: Any) -> str:
        """Represent as string."""
        # pylint: disable=unused-argument
        return str(self.members)


@total_ordering
class ChainGroup(Group):
    """Chain of groups.

    Attributes
    ----------
    members
        Sequence of group objects.
    """
    __slots__ = ()

    def __init__(self, members: Iterable[Group] = ()) -> None:
        members = tuple(
            Group(m) if not isinstance(m, Group) else m
            for m in members
        )
        super().__init__(members)

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Sequence) -> bool:
        if self.is_comparable_with(other):
            return self.flat == tuple(other)
        return NotImplemented

    def __lt__(self, other: Sequence) -> bool:
        if self.is_comparable_with(other):
            return self.flat < tuple(other)
        return NotImplemented

    def __len__(self) -> int:
        return len(self.flat)

    def __getitem__(self, idx: int | slice) -> Any | tuple[Any, ...]:
        return self.flat[idx]

    # Properties --------------------------------------------------------------

    @property
    def groups(self) -> tuple[Group, ...]:
        return self.members

    @property
    def flat(self) -> tuple[Any, ...]:
        return tuple(m for g in self.members for m in g)

    @property
    def hashdata(self) -> tuple[Any, ...]:
        return self.members

    # Methods -----------------------------------------------------------------

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, Sequence)

    def to_str(self, *, color: bool = True, **kwds: Any) -> str:
        """Represent as string."""
        s = ", ".join(g.to_str(color=color, **kwds) for g in self.members)
        return f"({s})"


class Graph(MutableMapping, SegramABC):
    """Graph.

    By default it has a form of a mapping from sources to targets,
    but the order can be easily reversed using :attr:`~rev` property.

    Notes
    -----
    Only hashable objects can used as nodes.
    """
    __slots__ = ("_data", "_is_dag", "_rev")

    def __init__(self, data: Mapping[Any, tuple[Any, ...]]) -> None:
        self._data = data
        self._is_dag = None
        self._rev = None

    def __repr__(self) -> str:
        return repr(self.data)

    def __iter__(self) -> Iterator[Any]:
        yield from self.data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: Any) -> Iterable[Any]:
        return self.data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self.data[key] = value

    def __delitem__(self, key: Any) -> None:
        del self.data[key]

    # Properties --------------------------------------------------------------

    @property
    def data(self) -> dict[Any, tuple[Any, ...]]:
        return self._data

    @property
    def links(self) -> Iterator[tuple[Any, Any]]:
        for parent, children in self.items():
            for child in children:
                yield parent, child

    @property
    def rev(self) -> Graph:
        if self._rev is None:
            self._rev = self.get_rev()
        return self._rev

    @property
    def sorted(self) -> Graph:
        return self.__class__({
            k: tuple(sorted(v))
            for k, v in sorted(self.items(), key=lambda x: x[0])
        })

    @property
    def sources(self) -> tuple[Any, ...]:
        pmap = self.rev
        src = list(self.isolates)
        for parent in self.toposort().static_order():
            if pmap.get(parent):
                break
            src.append(parent)
        return tuple(sorted(src))

    @property
    def isolates(self) -> tuple[Any, ...]:
        isol = []
        rev = self.rev
        for node in self:
            if not self.get(node) and not rev.get(node):
                isol.append(node)
        return tuple(isol)

    @property
    def is_dag(self) -> bool:
        if self._is_dag is None:
            try:
                self.toposort().prepare()
                self._is_dag = True
            except CycleError:
                self._is_dag = False
        return self._is_dag

    # Methods -----------------------------------------------------------------

    def get_rev(self) -> Graph:
        """Get reversed graph (mapping children to parent sets)."""
        graph = {}
        for parent, children in self.items():
            if parent not in graph:
                graph[parent] = []
            for child in children:
                graph.setdefault(child, []).append(parent)
        return self.__class__(graph).sorted

    def update_rev(self) -> None:
        """Update reversed graph."""
        self._rev = self.get_rev()

    def is_comparable_with(self, other: Mapping) -> bool:
        return isinstance(other, Mapping)

    def toposort(self, *, reversed: bool = False) -> TopologicalSorter:
        """Get topological sorter.

        Parameters
        ----------
        reversed
            Should reversed topological sorter be returned.
        """
        # pylint: disable=redefined-builtin
        return TopologicalSorter(self if reversed else self.rev)

    def iter_hierarchy(self) -> Iterable[tuple[int, Any, Any]]:
        """Iterate over parents and children according to the implicit
        directed acyclic graph hierarchy.

        Yields
        ------
        depth
            Depth index.
        parent, child
            Parent and child nodes.
        """
        def _iter(parent, depth=0):
            if (children := self.get(parent)):
                for child in children:
                    yield depth, parent, child
                    yield from _iter(child, depth=depth+1)
            elif depth == 0:
                yield depth, parent, None
        for source in self.sources:
            yield from _iter(source, depth=0)

    @classmethod
    def from_links(
        cls,
        links: Iterable[tuple[Any, Any]]
    ) -> Graph:
        """Construct from links.

        Parameters
        ----------
        links
            Iterable of links (parent-child pairs).
        """
        graph = {}
        for parent, child in links:
            if parent not in graph:
                graph[parent] = []
            if child is not None:
                graph[parent].append(child)
                if child not in graph:
                    graph[child] = []
        return cls(graph).sorted
