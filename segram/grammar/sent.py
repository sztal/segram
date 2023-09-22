from __future__ import annotations
from typing import Any, Optional, ClassVar
from collections.abc import Iterable, Mapping, Sequence
from .conjuncts import Conjuncts
from .abc import DocElement
from .components import Component
from .components import Verb, Noun
from .components import Prep, Desc
from .phrases import Phrase, VerbPhrase, NounPhrase, DescPhrase, PrepPhrase
from .graph import PhraseGraph
from ..settings import settings
from ..nlp.tokens import Doc, Span, Token
from ..symbols import Role


class Sent(Sequence, DocElement):
    """Grammar sentence class.

    Components within a sentence form a directed acyclic graph
    with connections going from controlling to dependent componentss.

    Attributes
    ----------
    start, end
        Start and end indices.
    verbs
        Verb components.
    nouns
        Noun components.
    descs
        Descriptive components.
    preps
        Prepositional components.
    graph
        Component graph.
    conjuncts
        Mapping from lead components to conjunct groups.
    """
    # pylint: disable=too-many-public-methods
    # pylint: disable=too-many-instance-attributes
    __components__ = ("verbs", "nouns", "descs", "preps")
    __slots__ = (
        "start", "end", *__components__,
        "graph", "conjs", "_cmap", "_pmap"
    )
    alias = "Sent"
    component_names: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        doc: Doc,
        start: int,
        end: int,
        verbs: Iterable[Verb] = (),
        nouns: Iterable[Noun] = (),
        descs: Iterable[Desc] = (),
        preps: Iterable[Prep] = (),
        graph: Optional[PhraseGraph[Phrase, tuple[Phrase, ...]]] = None,
        conjs: Optional[Mapping[Component, Conjuncts]] = None
    ) -> None:
        super().__init__(doc)
        self.start = start
        self.end = end
        self.check_sent(self.sent)
        self.nouns = tuple(nouns)
        self.verbs = tuple(verbs)
        self.preps = tuple(preps)
        self.descs = tuple(descs)
        self.graph = graph
        self.conjs = conjs or {}
        self._cmap = {}
        self._pmap = {}

    def __len__(self) -> int:
        return len(self.sent)

    def __getitem__(self, idx: int | slice) -> Component | tuple[Component, ...]:
        return self.sent[idx]

    def __contains__(
        self,
        other: Token | Component
    ) -> bool:
        if isinstance(other, Phrase):
            return other in self.phrases
        if isinstance(other, Component):
            return other in self.components
        if isinstance(other, Token):
            return other in self.sent
        return super().__contains__(other)

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.init_class_attrs({ "__components__": "component_names" })

    # Properties --------------------------------------------------------------

    @property
    def doc(self) -> Doc:
        return self._doc

    @property
    def idx(self) -> tuple[int, int]:
        return (self.start, self.end)

    @property
    def sent(self) -> Span:
        return self.doc[self.start:self.end]

    @property
    def cmap(self) -> Mapping[int, Component]:
        return self._cmap

    @property
    def pmap(self) -> Mapping[int, Phrase]:
        return self._pmap

    @property
    def root(self) -> Component:
        root = self.sent.root
        return next(c for c in self.components if root in c)

    @property
    def sources(self) -> tuple[Phrase, ...]:
        return Conjuncts.get_chain(self.graph.sources)

    @property
    def vps(self) -> tuple[Phrase, ...]:
        return tuple(p for p in self.phrases if isinstance(p, VerbPhrase))

    @property
    def nps(self) -> tuple[Phrase, ...]:
        return tuple(p for p in self.phrases if isinstance(p, NounPhrase))

    @property
    def dps(self) -> tuple[Phrase, ...]:
        return tuple(p for p in self.phrases if isinstance(p, DescPhrase))

    @property
    def pps(self) -> tuple[Phrase, ...]:
        return tuple(p for p in self.phrases if isinstance(p, PrepPhrase))

    @property
    def tokens(self) -> tuple[Token, ...]:
        return tuple(self.sent)

    @property
    def components(self) -> tuple[Component, ...]:
        return tuple(self.cmap.values())

    @property
    def phrases(self) -> tuple[Phrase, ...]:
        return tuple(self.pmap.values())

    @property
    def coverage(self) -> float:
        return sum(1 for _ in self.iter_token_roles()) / len(self.sent)

    # Methods -----------------------------------------------------------------

    @classmethod
    def from_data(cls, doc: Doc, data: dict[str, Any]) -> Sent:
        """Construct from a :class:`~segram.nlp.Doc` and a data dictionary."""
        sent = cls(doc, data["start"], data["end"])
        sent.nouns = tuple(
            cls.types.Noun.from_data(sent, dct)
            for dct in data.get("nouns", {}).values()
        )
        sent.verbs = tuple(
            cls.types.Verb.from_data(sent, dct)
            for dct in data.get("verbs", {}).values()
        )
        sent.preps = tuple(
            cls.types.Prep.from_data(sent, dct)
            for dct in data.get("preps", {}).values()
        )
        sent.descs = tuple(
            cls.types.Desc.from_data(sent, dct)
            for dct in data.get("descs", {}).values()
        )
        for dct in data["phrases"]:
            phrase = cls.types.Phrase.from_data(sent, dct)
            sent.pmap[phrase.idx] = phrase
        sent.graph = PhraseGraph.from_data(sent, data["graph"])
        sent.conjs = {
            (conj := Conjuncts.from_data(sent, c)).lead: conj
            for c in data["conjs"]
        }
        return sent

    def to_data(self) -> dict[str, Any]:
        """Dump to data dictionary."""
        return dict(
            start=self.start,
            end=self.end,
            nouns={ c.idx: c.to_data() for c in self.nouns },
            verbs={ c.idx: c.to_data() for c in self.verbs },
            preps={ c.idx: c.to_data() for c in self.preps },
            descs={ c.idx: c.to_data() for c in self.descs },
            phrases=[ p.to_data() for p in self.phrases ],
            graph=self.graph.to_data(),
            conjs=[ c.to_data() for c in self.conjs.values() ]
        )

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

    def is_comparable_with(self, other: Sent) -> None:
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
