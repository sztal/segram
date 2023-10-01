from typing import Iterable, Sequence
from .grammar import RulebasedEnglishGrammar
from ......grammar import SentNLP
from .......grammar.lang.en import EnglishSent
from .......grammar import Component, Phrase
from .......grammar import Conjuncts
from ......tokens import Token
from .......symbols import Dep


class RulebasedEnglishSent(
    RulebasedEnglishGrammar,
    EnglishSent, SentNLP
):
    """Rule-based :mod:`spacy` English sentence element."""
    __slots__ = ()

    def find_links(self) -> Iterable[tuple[Phrase, Phrase]]:
        """Find phrase links.

        Yields
        ------
        parent
            Parent phrase.
        child
            Child phrase.
        """
        # pylint: disable=too-many-branches
        comps = self.components
        for comp in comps:
            parents = tuple(comp.find_parents(comps))
            phrase = comp.phrase
            if not parents:
                phrase.dep = Dep.root
                yield phrase, None
            else:
                for parent in parents:
                    phrase.dep = comp.get_dep(parent)
                    phrase.sconj = comp.get_sconj(parent)
                    yield parent.phrase, phrase

    def find_conjs(self) -> Iterable[Conjuncts]:
        """Iterate over groups of conjoined comps."""
        groups = set()
        comps = self.components
        for i, comp in enumerate(self.components):
            for other in comps[i:]:
                if (cc := comp.get_cconj(other)):
                    group = (cc, tuple(sorted((comp, other))))
                    self._expand_conj_group(group, groups, comps)
        for group in groups:
            cconj, comps = group
            phrases = [ c.phrase for c in comps ]
            conjs = sorted((cconj.head, *cconj.head.conjuncts))
            left = list(conjs)[0]
            pconj = next((c for c in left.lefts if c.is_preconj), None)
            lead = next((i for i, p in enumerate(comps) if p.head == p.head.lead), 0)
            conjs = Conjuncts(phrases, lead=lead, cconj=cconj, preconj=pconj)
            for conj in conjs:
                conj._lead = conjs.lead.idx # pylint: disable=protected-access
                conj.dep |= conjs.lead.dep
            yield conjs

    def add_subs(self) -> None:
        """Add free subtree tokens to components."""
        comps = [ c.tok for c in self.components ]
        for tok in self:
            if tok in comps:
                continue
            head = tok
            while not head.is_root:
                head = head.head
                if head is tok:
                    break
                if head in comps:
                    comp = self.cmap[head.i]
                    if tok not in comp.tokens:
                        comp.sub = (*comp.sub, tok)
                    break

    def _expand_conj_group(
        self,
        group: tuple[Token, tuple[Component, ...]],
        groups: set[tuple[Component, ...]] | None = None,
        comps: Sequence[Component] | None = None
    ) -> Iterable[tuple[Component, ...]]:
        groups = groups if groups is not None else set()
        comps = comps if comps is not None else self.components
        larger = None
        cc, group = group
        for comp in comps:
            if comp not in group \
            and all(cc == comp.get_cconj(p) for p in group):
                larger = (cc, tuple(sorted((*group, comp))))
                if larger not in groups:
                    groups.add(larger)
                    self._expand_conj_group(larger, groups, comps)
        if larger is None:
            groups.add((cc, group))
