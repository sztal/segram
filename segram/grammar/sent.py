from __future__ import annotations
from typing import Any, ClassVar, Self, Mapping
from .conjuncts import Conjuncts
from .abc import SentElement
from .components import Component
from .components import Verb, Noun
from .components import Prep, Desc
from .phrases import Phrase, VerbPhrase, NounPhrase, DescPhrase, PrepPhrase
from .graph import PhraseGraph
from ..settings import settings
from ..nlp.tokens import Doc, Span, Token
from ..symbols import Role
from ..utils.misc import best_matches
from ..datastruct import DataChain, DataSequence


PVType = DataChain[DataSequence[Phrase]]


class Sent(SentElement):
    """Grammar sentence class.

    Components within a sentence form a directed acyclic graph
    with connections going from controlling to dependent components.

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
    __components__ = ("verbs", "nouns", "descs", "preps")
    __slots__ = (*__components__, "graph", "conjs", "cmap", "pmap")
    alias = "Sent"
    component_names: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        sent: Span,
        *,
        verbs: DataSequence[Verb] = (),
        nouns: DataSequence[Noun] = (),
        descs: DataSequence[Desc] = (),
        preps: DataSequence[Prep] = (),
        graph: PhraseGraph[Phrase, tuple[Phrase, ...]] | None = None,
        conjs: Mapping[Component, Conjuncts] | None = None,
        pmap: Mapping[int, Phrase] | None = None,
        cmap: Mapping[int, Component] | None = None
    ) -> None:
        super().__init__(sent)
        self.nouns = DataSequence(nouns)
        self.verbs = DataSequence(verbs)
        self.preps = DataSequence(preps)
        self.descs = DataSequence(descs)
        self.graph = graph
        self.conjs = conjs or {}
        self.cmap = cmap or {}
        self.pmap = pmap or {}

    def __new__(cls, *args: Any, **kwds: Any) -> None:
        obj = super().__new__(cls)
        obj.__init__(*args, **kwds)
        cache = getattr(obj.sent.doc._, f"{settings.spacy_alias}_cache")["sents"]
        idx = obj.idx
        if (cur := cache.get(idx)):
            cur.__init__(obj.sent, **obj.data)
            return cur
        cache[idx] = obj
        return obj

    def __len__(self) -> int:
        return len(self.sent)

    def __getitem__(self, idx: int | slice) -> Component | tuple[Component, ...]:
        return self.sent[idx]

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.init_class_attrs({ "__components__": "component_names" })

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
        return Conjuncts.get_chain(self.graph.sources)

    @property
    def vps(self) -> PVType:
        return Conjuncts \
            .get_chain(p for p in self.phrases if isinstance(p, VerbPhrase))

    @property
    def nps(self) -> PVType:
        return Conjuncts \
            .get_chain(p for p in self.phrases if isinstance(p, NounPhrase))

    @property
    def dps(self) -> PVType:
        return Conjuncts \
            .get_chain(p for p in self.phrases if isinstance(p, DescPhrase))

    @property
    def pps(self) -> PVType:
        return Conjuncts \
            .get_chain(p for p in self.phrases if isinstance(p, PrepPhrase))

    @property
    def tokens(self) -> DataSequence[Token]:
        return DataSequence(tuple(self.sent))

    @property
    def components(self) -> DataSequence[Component]:
        return DataSequence(self.cmap.values())

    @property
    def phrases(self) -> PVType:
        return Conjuncts.get_chain(self.pmap.values())

    @property
    def coverage(self) -> float:
        return sum(1 for _ in self.iter_token_roles()) / len(self.sent)

    # Methods -----------------------------------------------------------------

    def similarity(self, spec: Self | dict , *args: Any, **kwds: Any) -> float:
        """Structured similarity to other sentence.

        It is computed as the average similarity between
        best matching root phrases of the two sentences.

        Parameters
        ----------
        spec
            Match specification. It may be another sentence
            or a match specification dictionary as used
            in structured similarity for phrases.
        *args, **kwds
            Passed to :meth:`segram.grammar.Phrase.similarity.
        """
        proots = self.proots
        if isinstance(spec, Sent):
            oroots = spec.roots
            return sum(score for score, *_ in best_matches(
                proots, oroots, lambda s, o: s.similarity(o, *args, **kwds)
            )) / max(len(proots), len(oroots))
        return sum(p.similarity(spec, *args, **kwds) for p in proots) / len(proots)


    @classmethod
    def from_data(cls, doc: Doc, data: dict[str, Any]) -> Self:
        """Construct from a :class:`~segram.nlp.Doc` and a data dictionary."""
        sent = cls(doc[data["start"]].sent)
        sent = doc[data["start"]].sent
        kwds = {
            "nouns": tuple(
                cls.types.Noun.from_data(doc, dct)
                for dct in data.get("nouns", {}).values()
            ),
            "verbs": tuple(
                cls.types.Verb.from_data(sent, dct)
                for dct in data.get("verbs", {}).values()
            ),
            "preps": tuple(
                cls.types.Prep.from_data(sent, dct)
                for dct in data.get("preps", {}).values()
            ),
            "descs": tuple(
                cls.types.Desc.from_data(sent, dct)
                for dct in data.get("descs", {}).values()
            ),
        }
        pmap = {}
        for dct in data["phrases"]:
            phrase = cls.types.Phrase.from_data(sent, dct)
            pmap[phrase.idx] = phrase
        pmap = dict(sorted(pmap.items(), key=lambda x: x[0]))
        graph = PhraseGraph.from_data(sent, data["graph"])
        conjs = {
            (conj := Conjuncts.from_data(sent, c)).lead: conj
            for c in data["conjs"]
        }
        return cls(sent, pmap=pmap, graph=graph, conjs=conjs, **kwds)

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
