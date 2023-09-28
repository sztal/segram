from typing import Any, Iterable, Self
from ..nlp.tokens import Token
from ..datastruct import DataTuple
from ..utils.misc import stringify


class PhraseGroup(DataTuple):
    """Group of phrases."""

    @property
    def conjs(self) -> DataTuple["Conjuncts"]:
        """Phrases groups as conjuncts."""
        return Conjuncts.get_chain(self)


class Conjuncts(DataTuple):
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

    def __init__(
        self,
        members: Iterable["Phrase"] = (),
        *,
        lead: int = 0,
        cconj: Token | None = None,
        preconj: Token | None = None
    ) -> None:
        # pylint: disable=unused-argument
        super().__init__()
        self._lead = lead
        self.cconj = cconj
        self.preconj = preconj

    def __repr__(self) -> str:
        return self.to_str(color=True)

    def __hash__(self) -> int:
        return hash(self.hashdata)

    # Properties --------------------------------------------------------------

    @property
    def members(self) -> tuple["Phrase", ...]:
        return tuple(self)

    @property
    def lead(self) -> Any:
        return self.members[self._lead]

    @property
    def cconjs(self) -> tuple[Any, ...]:
        return tuple(getattr(self, name) for name in self.__cconjs__)

    @property
    def hashdata(self) -> tuple[Any, ...]:
        return (tuple(self), self._lead, tuple(self.cconjs))

    @property
    def data(self) -> dict[str, any]:
        return {
            "members": self.members,
            "lead": self.lead,
            "cconj": self.cconj,
            "preconj": self.preconj
        }

    # Methods -----------------------------------------------------------------

    @classmethod
    def from_data(
        cls,
        sent: "Sent",
        data: dict[str, int | list[int] | None ],
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
        sent = sent.grammar
        lead = data["lead"]
        cconj = data.get("cconj")
        pconj = data.get("preconj")
        members = [ sent.pmap[m] for m in data["members"] ]
        if cconj is not None:
            cconj = doc[cconj]
        if pconj is not None:
            pconj = doc[pconj]
        return cls(members, lead=lead, cconj=cconj, preconj=pconj)

    def to_data(self) -> dict[str, int | list[int] | None]:
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

    def to_str(self, *, color: bool = False, **kwds: Any) -> str:
        coords = \
            "|".join(
                stringify(c, color=color, **kwds)
                for c in self.cconjs if c
            ).strip()
        if coords:
            coords = f"[{coords}]"
        members = ", ".join(stringify(m, color=color, **kwds) for m in self.members)
        return f"{coords}({members})"

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, Conjuncts)

    @classmethod
    def find_groups(cls, phrases: Iterable["Phrase"]) -> Iterable[Self]:
        """Find conjuncts groups in ``phrases``."""
        # pylint: disable=protected-access
        groups = {}
        for phrase in phrases:
            groups.setdefault(phrase.group.lead.idx, []).append(phrase)
        for lead_idx, group in groups.items():
            if not group:
                continue
            if len(group) == 1:
                yield Conjuncts(group)
            else:
                yield group[0].sent.conjs[lead_idx].copy(members=group)

    @classmethod
    def get_chain(cls, phrases: Iterable["Phrase"]) -> DataTuple["Conjuncts"]:
        """Get chain of conjuncts groups in ``phrases``."""
        return DataTuple(cls.find_groups(phrases))

    def copy(self, **kwds: Any) -> Self:
        kwds = { **self.data, **kwds }
        members = kwds.pop("members", ())
        return self.__class__(members, **kwds)

# class PhraseGroup(DataChain):
#     """Phrase group class.

#     This is a chain of groups of conjoined phrases
#     enhanced with several methods for matching, grouping,
#     summarizing and aggregating information from phrases.
#     """
#     __slots__ = ()

#     def __init__(self, members: Iterable[DataTuple] = ()) -> None:
#         members = DataTuple(
#             Conjuncts(m) if not isinstance(m, Conjuncts) else m
#             for m in members
#         )
#         super().__init__(members)

#     def match(
#         self,
#         *args: Any,
#         require: Callable[Iterable["Phrase"], bool] = any,
#         **kwds: Any
#     ) -> bool:
#         """Match phrase group against a specification.

#         Parameters
#         ----------
#         *args, **kwds
#             Passed to :meth:`segram.grammar.Phrase`.
#         require
#             Function deciding whether the phrase group
#             after filtering satisfies the requirements.
#         """
#         return require(p.match(*args, **kwds) for p in self)

#     def group_by_doc(self) -> dict[str, PhraseGroup]:
#         """Group by documents."""
#         data = {}
#         for group in self.members:
#             data.setdefault(id(group.lead.doc), []).append(group)
#         final = {}
#         for v in data.values():
#             final[v[0].lead.doc.id] = self.__class__(sorted(v))
#         return final

#     def get_conjuncts(self) -> Self:
#         """Get non-trivial conjunct groups."""
#         return self.__class__([
#             m for m in self.members if len(m) > 1
#         ])

#     def group_by_head(
#         self,
#         *parts,
#         lemmatize: bool = True,
#         coref: bool = True,
#         pos: bool = True,
#         ent: bool = True,
#         lexeme: bool = True
#     ) -> dict[str, PhraseGroup]:
#         """Group by phrases by head tokens.

#         Parameters
#         ----------
#         *parts
#             Names of the parts (e.g. ``"subj"`` or ``"xcomp"``)
#             to use. Use all parts if ``None``.
#         lemmatize
#             Lemmatize token texts used as keys.
#         coref
#             Resolve coreferences (to the leading ref)
#             for use as keys.
#         pos
#             Add POS tags to keys.
#         ent
#             Add entity types to keys.
#         lexeme
#             Add ``"lexeme"`` field storing
#             lexeme objects corresponding to tokens.
#         """
#         # pylint: disable=too-many-locals
#         data = {}
#         for phrase in self:
#             tok = phrase.head.tok
#             if coref:
#                 tok = tok.coref
#             key = tok.lemma if lemmatize else tok.text
#             if pos or ent:
#                 key = (key,)
#                 if pos:
#                     key = (*key, tok.pos)
#                 if ent:
#                     key = (*key, tok.ent)
#             data \
#                 .setdefault(key, {}) \
#                 .setdefault("phrases", []).append(phrase)
#             rec = data[key]
#             if lexeme and (lkey := "lexeme") not in rec \
#             and (vocab := getattr(tok, "vocab", None)):
#                 rec[lkey] = vocab[key[0] if isinstance(key, tuple) else key]
#             for name in phrase.part_names:
#                 if parts and name not in parts:
#                     continue
#                 for part in getattr(phrase, name, ()):
#                     rec.setdefault(name, []).append(part)
#         for key in data:
#             data[key] = { k: v for k, v in data[key].items() if v }
#         return data
