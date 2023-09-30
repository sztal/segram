from typing import Any, ClassVar, Self, Mapping
import numpy as np
from .conjuncts import PhraseGroup, Conjuncts
from .abc import SentElement
from .components import Component
from .components import Verb, Noun, Prep, Desc
from .phrases import Phrase
from .graph import PhraseGraph
from ..settings import settings
from ..nlp.tokens import Doc, Span, Token
from ..symbols import Role
from ..abc import labelled
from ..utils.misc import sort_map
from ..datastruct import DataTuple


PVType = PhraseGraph[Phrase]
component_ = labelled("component")


class Sent(SentElement):
    """Grammar sentence class.

    Components within a sentence form a directed acyclic graph
    with connections going from controlling to dependent components.

    Attributes
    ----------
    cmap
        Mapping from head token ids to components.
    pmap
        Mapping from head tokens ids to phrases.
    graph
        Component graph.
    conjuncts
        Mapping from lead components to conjunct groups.
    """
    # pylint: disable=too-many-public-methods
    __slots__ = ("graph", "conjs", "cmap", "pmap")
    alias = "Sent"
    component_names: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        sent: Span,
        *,
        cmap: Mapping[int, Component] | None = None,
        pmap: Mapping[int, Phrase] | None = None,
        graph: PhraseGraph[Phrase, tuple[Phrase, ...]] | None = None,
        conjs: Mapping[int, Conjuncts] | None = None
    ) -> None:
        super().__init__(sent)
        self.cmap = sort_map(cmap or {})
        self.pmap = sort_map(pmap or {})
        self.graph = graph
        self.conjs = conjs or {}

    def __new__(cls, *args: Any, **kwds: Any) -> None:
        obj = super().__new__(cls)
        obj.__init__(*args, **kwds)
        idx = obj.idx
        cache = obj.doc.smap
        if (cur := cache.get(idx)):
            cur.__init__(**obj.data)
            return cur
        cache[idx] = obj
        return obj

    def __len__(self) -> int:
        return len(self.sent)

    # Properties --------------------------------------------------------------

    @property
    def root(self) -> Component:
        """Root component."""
        return self.cmap[super().root.i]

    @property
    def proots(self) -> Conjuncts[Phrase]:
        """Root phrases."""
        return self.root.phrase.group

    @property
    def sources(self) -> PVType:
        return PhraseGroup(self.graph.sources)

    @property
    @component_
    def verbs(self) -> DataTuple[Verb]:
        return self.components.filter(lambda c: isinstance(c, Verb)).tuple

    @property
    @component_
    def nouns(self) -> DataTuple[Noun]:
        return self.components.filter(lambda c: isinstance(c, Noun)).tuple

    @property
    @component_
    def preps(self) -> DataTuple[Verb]:
        return self.components.filter(lambda c: isinstance(c, Prep)).tuple

    @property
    @component_
    def descs(self) -> DataTuple[Verb]:
        return self.components.filter(lambda c: isinstance(c, Desc)).tuple

    @property
    def tokens(self) -> DataTuple[Token]:
        return DataTuple(tuple(self.sent))

    @property
    def components(self) -> DataTuple[Component]:
        return DataTuple(self.cmap.values())

    @property
    def phrases(self) -> PVType:
        return PhraseGroup(self.pmap.values())

    @property
    def coverage(self) -> float:
        return sum(1 for _ in self.iter_token_roles()) / len(self.sent)

    @property
    def vector(self) -> np.ndarray[tuple[int], np.floating]:
        return self.sent.vector

    # Methods -----------------------------------------------------------------

    def similarity(self, *args: Any, **kwds: Any) -> float:
        """Structured similarity with respect to other sentence or phrase."""
        return self.Similarity(self, *args, **kwds).similarity

    @classmethod
    def from_data(cls, doc: Doc, data: dict[str, Any]) -> Self:
        """Construct from a :class:`~segram.nlp.Doc` and a data dictionary."""
        # pylint: disable=protected-access
        data = data.copy()
        sent = doc[data.pop("start"):data.pop("end")]
        data["cmap"] = {
            idx: cls.types.Component.from_data(doc, dct)
            for idx, dct in data["cmap"].items()
        }
        data["pmap"] = {
            idx: cls.types.Phrase.from_data(doc, dct)
            for idx, dct in data["pmap"].items()
        }
        data["graph"] = PhraseGraph.from_data(sent, data["graph"])
        data["conjs"] = {
            (conj := Conjuncts.from_data(sent, c))._lead: conj
            for c in data["conjs"]
        }
        return cls(sent, **data)

    def to_data(self) -> dict[str, Any]:
        """Dump to data dictionary."""
        return {
            "start": self.start,
            "end":   self.end,
            "cmap":  { idx: c.to_data() for idx, c in self.cmap.items() },
            "pmap":  { idx: p.to_data() for idx, p in self.pmap.items() },
            "graph": self.graph.to_data(),
            "conjs": [ c.to_data() for c in self.conjs.values() ]
        }

    def iter_token_roles(self) -> tuple[Token, Role | None]:
        """Iterate over token-role pairs."""
        def _iter():
            seen = set()
            for comp in self.components:
                for tok in comp.subtokens:
                    if tok not in seen:
                        seen.add(tok)
                        yield tok, comp.role
        yield from sorted(_iter(), key=lambda x: x[0])

    def to_str(self, *, color: bool = False, **kwds: Any) -> str:
        """Represent as a string."""
        # pylint: disable=unused-argument
        s = ""
        for tok, role in self.iter_token_roles():
            s += tok.to_str(color=color, role=role)+tok.whitespace
        return s

    def is_comparable_with(self, other: Any) -> None:
        return isinstance(other, Sent)

    @staticmethod
    def check_sent(span: Span) -> bool:
        """Check if a span is a proper sentence."""
        if span != span[0].sent:
            raise ValueError("'span' must be a proper sentence")

    def print(self) -> None:
        """Pretty print summary."""
        # pylint: disable=too-many-branches,not-an-iterable
        msg = settings.printer.get()
        print(msg.color(self.sent.text, bold=True), end="\n")
        for field in (*self.component_names, "graph", "conjs"):
            vals = getattr(self, field)
            if field in ("start", "end"):
                continue
            if field == "nouns":
                head = "Noun components"
            elif field == "verbs":
                head = "Verb components"
            elif field == "preps":
                head = "Prepositional components"
            elif field == "descs":
                head = "Descriptive components"
            elif field == "conjs":
                head = "Conjunct groups"
            else:
                head = "Component structure"
            if vals:
                print(msg.divider(head))
                if field == "graph":
                    vals.print()
                elif field == "conjs":
                    for obj in vals.values():
                        print(obj)
                else:
                    for comp in vals:
                        print(f"{comp.idx}:", comp)
