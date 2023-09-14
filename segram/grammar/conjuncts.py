from __future__ import annotations
from typing import Optional, Iterator, Self
from collections.abc import Iterable
from ..nlp import TokenABC
from ..utils.types import Group, ChainGroup


class Conjuncts(Group):
    """Group of conjoined phrases.

    Attributes
    ----------
    members
        Conjoined phrases.
    lead
        Lead component.
    cconj
        Conjunction token.
    preconj
        Preconjunction token.
    """
    __cconjs__ = ("cconj", "preconj")
    __slots__ = (*__cconjs__,)

    def __init__(
        self,
        members: Iterable["Phrase"] = (),
        lead: int = 0,
        cconj: Optional[TokenABC] = None,
        preconj: Optional[TokenABC] = None
    ) -> None:
        super().__init__(members, lead)
        self.cconj = cconj
        self.preconj = preconj

    # Properties --------------------------------------------------------------

    @property
    def components(self) -> tuple["Phrase", ...]:
        return self.members

    # Methods -----------------------------------------------------------------

    @classmethod
    def from_data(
        cls,
        sent: "Sent",
        data: dict[str, Optional[int] | list[int]],
    ) -> Self:
        """Construct from data dictionary.

        Parameters
        ----------
        sent
            Sentence object.
        data
            Data dictionary.
        cdict
            Mapping from ordinal numbers to components.
        """
        doc = sent.doc
        lead = data["lead"]
        cconj = data.get("cconj")
        pconj = data.get("preconj")
        members = [ sent.pmap[m] for m in data["members"] ]
        if cconj is not None:
            cconj = doc[cconj]
        if pconj is not None:
            pconj = doc[pconj]
        return cls(members, lead, cconj, pconj)

    def to_data(self) -> dict[str, Optional[int] | list[int]]:
        """Dump to data dictionary.

        Parameters
        ----------
        odict
            Mapping from components to their ordinal
            numbers within the sentence sequence.

        Returns
        -------
        data
            Dictionary with list of components ordinal numbers
            and and index of the conjunction token, or ``None``.
        """
        return {
            "members": [ comp.idx for comp in self.members ],
            "lead": self._lead,
            "cconj": self.cconj.i if self.cconj else None,
            "preconj": self.preconj.i if self.preconj else None
        }

    def is_comparable_with(self, other: Conjuncts) -> bool:
        return isinstance(other, Conjuncts)

    @classmethod
    def find_groups(cls, phrases: Iterable["Phrase"]) -> Iterator[Conjuncts]:
        """Find conjuncts groups in ``phrases``."""
        lmap = {}
        for phrase in phrases:
            lmap.setdefault(phrase.lead, []).append(phrase)
        for lead, members in lmap.items():
            try:
                lidx = members.index(lead)
            except ValueError:
                lidx = 0
            cconjs = lead.conjuncts.cconjs if len(members) > 1 else ()
            yield Conjuncts(members, lidx, *cconjs)

    @classmethod
    def get_chain(cls, phrases: Iterable["Phrase"]) -> ChainGroup:
        """Get chain of conjuncts groups in ``phrases``."""
        return ChainGroup(cls.find_groups(phrases))
