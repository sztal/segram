from typing import Iterable
from abc import abstractmethod
from .grammar import GrammarNLP
from ..tokens import Span
from ...grammar import Phrase, Sent, PhraseGraph, Conjuncts
from ...symbols import Dep


class SentNLP(GrammarNLP, Sent):
    """Abstract base class for sentence elements
    with NLP backend methods.
    """
    __slots__ = ()

    @abstractmethod
    def find_links(self) -> Iterable[tuple[Phrase, Phrase]]:
        """Find phrase links."""

    @abstractmethod
    def find_conjs(self) -> Iterable[Conjuncts]:
        """Iterate over groups of conjoined components."""

    @abstractmethod
    def add_subs(self) -> None:
        """Add free subtree tokens to components."""

    # Methods -----------------------------------------------------------------

    @classmethod
    def from_sent(cls, sent: Span) -> Sent:
        """Construct from a sentence span object."""
        sent = cls(sent)
        for tok in sent.sent:
            cls.types.Noun.from_tok(tok)
            cls.types.Verb.from_tok(tok)
            cls.types.Prep.from_tok(tok)
            cls.types.Desc.from_tok(tok)
        sent.add_subs()
        sent.graph = PhraseGraph.from_links(sent.find_links())
        sent.conjs = { conj.lead: conj for conj in sorted(sent.find_conjs()) }
        sent.make_mutable_children()
        sent.destroy_conjunct_links()
        sent.propagate_children_conjuncts()
        sent.propagate_subjects()
        sent.propagate_descriptions()
        sent.propagate_cdesc_subclauses()
        sent.freeze_children()
        return sent

    def make_mutable_children(self) -> None:
        """Make children lists in ``self.graph`` mutable sets."""
        for phrase, children in self.graph.items():
            self.graph[phrase] = set(children)

    def destroy_conjunct_links(self) -> None:
        """Destroy original conjunct links."""
        for children in self.graph.values():
            for child in tuple(children):
                if child.dep & Dep.conj:
                    children.remove(child)

    def propagate_children_conjuncts(self) -> None:
        """Propagate conjuncts of children to their parents."""
        for phrase in self.graph:
            for child in tuple(phrase.children):
                if not child.is_lead:
                    continue
                for conj in child.conjuncts:
                    if conj.dep & Dep.conj:
                        self.graph[phrase].add(conj)

    def propagate_subjects(self) -> None:
        """Propagete subjects to subject-free conjuncts."""
        for phrase in self.graph:
            if phrase.is_lead and (conjs := phrase.conjuncts):
                for conj in conjs:
                    if conj.subj:
                        continue
                    for child in phrase.children:
                        if child.dep & Dep.subj:
                            self.graph[conj].add(child)

    def propagate_descriptions(self) -> None:
        """Propagate descriptions to description-free conjuncts."""
        for phrase in self.graph:
            if phrase.is_lead and (conjs := phrase.conjuncts):
                for conj in conjs:
                    if conj.desc:
                        continue
                    for child in phrase.children:
                        if child.dep & Dep.desc:
                            self.graph[conj].add(child)

    def propagate_cdesc_subclauses(self) -> None:
        """Propagate subclauses between clausal descriptions."""
        for phrase in self.graph:
            if phrase.dep & Dep.cdesc and phrase.is_lead \
            and (conjs := phrase.conjuncts):
                for conj in conjs:
                    if conj.subcl:
                        continue
                    for child in phrase.children:
                        if child.dep & Dep.subcl:
                            self.graph[conj].add(child)

    def freeze_children(self) -> None:
        """Freeze children lists."""
        for phrase, children in self.graph.items():
            self.graph[phrase] = tuple(sorted(children))
        self.graph.update_rev()
