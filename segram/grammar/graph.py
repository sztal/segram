from __future__ import annotations
from typing import Any, Self
from ..utils.types import Graph


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

    def is_comparable_with(self, other: PhraseGraph) -> bool:
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
    def from_data(cls, sent: "Sent", data: dict[int, tuple[int, int]]) -> Self:
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
