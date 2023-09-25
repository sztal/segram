from typing import Any, Iterable, Mapping, MutableMapping, Self
from graphlib import TopologicalSorter, CycleError
from ..abc import SegramABC
from ..nlp.tokens import Span


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

    def __iter__(self) -> Iterable[Any]:
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
    def links(self) -> Iterable[tuple[Any, Any]]:
        for parent, children in self.items():
            for child in children:
                yield parent, child

    @property
    def rev(self) -> Self:
        if self._rev is None:
            self._rev = self.get_rev()
        return self._rev

    @property
    def sorted(self) -> Self:
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

    def get_rev(self) -> Self:
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
    ) -> Self:
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


class PhraseGraph(Graph):
    """PhraseGraph.

    By default it has a form of a mapping from sources to targets,
    but the order can be easily reversed using :attr:`~rev` property.

    Notes
    -----
    Only hashable objects can used as nodes.
    """
    __slots__ = ()

    def __repr__(self) -> str:
        return self.to_str(color=True)

    # Methods -----------------------------------------------------------------

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, PhraseGraph)

    def to_str(self, *, color: bool = False, indent: int = 4, **kwds: Any) -> str:
        """Represent as a string."""
        # pylint: disable=broad-except,unused-argument
        sep = " "*indent
        s = ""
        showed = set()
        for depth, parent, child in self.iter_hierarchy():
            if depth == 0 and parent not in showed:
                showed.add(parent)
                if s:
                    s += "\n"
                if hasattr(parent, "to_str"):
                    text = parent.to_str(color=color, only_head=True)
                else:
                    text = str(parent)
                s += f"{sep*depth}{text}"
            if child:
                sconj = child.sconj or ""
                if sconj:
                    sconj = f"({sconj}) "
                if hasattr(child, "to_str"):
                    text = child.to_str(color=color, only_head=True)
                else:
                    text = str(child)
                s += f"\n{sep*(depth+1)}{sconj}{text} [{child.dep.name}]"
        return s

    def print(self, **kwds: Any) -> None:
        """Print graph hierarchy."""
        kwds = { "color": True, **kwds }
        print(self.to_str(**kwds))

    @classmethod
    def from_data(cls, sent: Span, data: dict[int, tuple[int, int]]) -> Self:
        """Construct from data dictionary.

        Parameters
        ----------
        sent
            Sentence object.
        data
            Data dictionary.
        cdict
            Mapping from ordinal numbers to nodes.
        """
        sent = sent.grammar
        graph = {}
        for idx, children in data.items():
            parent = sent.pmap[idx]
            children = [ sent.pmap[i] for i in children ]
            graph[parent] = children
        return cls(graph)

    def to_data(self) -> dict[int, tuple[int, int]]:
        """Dump to data dictionary.

        Parameters
        ----------
        odict
            Mapping from nodes to their ordinal
            numbers within the node sequence.

        Returns
        -------
        data
            Mapping from node ordinal numbers of nodes to lists of t
            riples with two integers giving the index of the target node,
            index of subordinating conjunction token and the name
            of a dependency symbol.
        """
        data = {}
        for parent, children in self.items():
            data[parent.idx] = []
            for child in children:
                data[parent.idx].append(child.idx)
        return data
