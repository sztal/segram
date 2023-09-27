# pylint: disable=too-many-public-methods,no-name-in-module
from typing import Any, Iterable, Self
from abc import abstractmethod
from spacy.tokens import MorphAnalysis, Token as SpacyToken
from .abc import NLP
from ...symbols import POS, Role
from ...utils.colors import color_role
from ...utils.diff import iter_diffs, equal, IDiffType
from ... import settings


class Token(NLP):
    """Token wrapper class."""
    __slots__ = ()

    def __repr__(self) -> str:
        return self.to_str(color=True)

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Self) -> bool:
        if (res := super().__eq__(other)) is NotImplemented:
            return res
        return res and self.i == other.i

    def __lt__(self, other: Self) -> bool:
        """Is ``self`` earlier in the document than ``other``."""
        if self.is_comparable_with(other):
            return self.i < other.i
        return NotImplemented

    # Abstract properties -----------------------------------------------------

    @property
    @abstractmethod
    def is_negation(self) -> bool:
        pass
    @property
    @abstractmethod
    def is_qmark(self) -> bool:
        pass
    @property
    @abstractmethod
    def is_exclam(self) -> bool:
        pass
    @property
    @abstractmethod
    def is_intj(self) -> bool:
        pass

    # Properties --------------------------------------------------------------

    @property
    def i(self) -> int:
        return self.tok.i

    @property
    def whitespace(self) -> str:
        return self.tok.whitespace_

    @property
    def whitespace_(self) -> str:
        return self.tok.whitespace_

    @property
    def text_with_ws(self) -> str:
        return self.tok.text_with_ws

    @property
    def pos(self) -> POS:
        return POS.from_name(self.tok.pos_)

    @property
    def role(self) -> Role:
        if self.is_negation:
            return Role.NEG
        if self.is_qmark:
            return Role.QMARK
        if self.is_exclam:
            return Role.EXCLAM
        if self.is_intj:
            return Role.INTJ
        return None

    @property
    def dep(self) -> str:
        return self.tok.dep_

    @property
    def tag(self) -> str:
        return self.tok.tag_

    @property
    def lemma(self) -> str:
        return self.tok.lemma_

    @property
    def ent(self) -> str:
        return self.tok.ent_type_

    @property
    def ent_tag(self) -> str:
        tag = self.tok.ent_iob_
        if (typ := self.tok.ent_type_):
            tag += "-"+typ
        return tag

    @property
    def doc(self) -> "Doc":
        return self.sns(self.tok.doc)

    @property
    def sent(self) -> "Span":
        return self.sns(self.tok.sent)

    @property
    def head(self) -> Self:
        return self.sns(self.tok.head)

    @property
    def morph(self) -> MorphAnalysis:
        return self.tok.morph

    @property
    def left_edge(self) -> Self:
        return self.sns(self.tok.left_edge)

    @property
    def right_edge(self) -> Self:
        return self.sns(self.tok.right_edge)

    @property
    def ancestors(self) -> Iterable[Self]:
        for tok in self.tok.ancestors:
            yield self.sns(tok)

    @property
    def conjuncts(self) -> tuple[Self, ...]:
        return tuple(self.sns(c) for c in self.tok.conjuncts)

    @property
    def children(self) -> Iterable[Self]:
        for child in self.tok.children:
            yield self.sns(child)

    @property
    def lefts(self) -> Iterable[Self]:
        for tok in self.tok.lefts:
            yield self.sns(tok)

    @property
    def rights(self) -> Iterable[Self]:
        for tok in self.tok.rights:
            yield self.sns(tok)

    @property
    def subtree(self) -> Iterable[Self]:
        for tok in self.tok.subtree:
            yield self.sns(tok)

    @property
    def corefs(self) -> tuple[Self, ...]:
        # pylint: disable=protected-access,redefined-outer-name
        if (refs := getattr(self._, f"{settings.spacy_alias}_corefs", None)):
            return tuple(self.doc[ref] for ref in refs)
        return ()

    @property
    def coref(self) -> Self:
        """Return main coreferred token or self."""
        if (refs := self.corefs):
            return refs[0]
        return self

    # Methods -----------------------------------------------------------------

    def to_str(
        self,
        *,
        color: bool = False,
        **kwds: Any
    ) -> str:
        """Represent as a string.

        Parameters
        ----------
        color
            Use colors.
        **kwds
            Passed to :func:`~segram.utils.colors.color_role`.
            They can be used to override the fixed token role
            with contextual roles using ``role`` keyword argument.
        """
        refs = self.corefs
        if refs:
            refs = ",".join(r.to_str(color=False) for r in refs)
            refs = f"[{refs}]"
            rrole = kwds.get("role")
            if rrole is Role.BG:
                refs = color_role(refs, **{ **kwds, "role": rrole })
        else:
            refs = ""
        kwds = { "role": self.role, **kwds }
        return f"{color_role(self.text, color=color, **kwds)}{refs}"

    def nbor(self, *args: Any, **kwds: Any) -> Self:
        return self.sns(self.tok.nbor(*args, **kwds))

    def is_ancestor(self, other: SpacyToken | Self) -> bool:
        if isinstance(other, Token):
            other = other.tok
        return self.tok.is_ancestor(other)


# Register comparison functions for testing -----------------------------------

@equal.register
def _(obj: Token, other: Token, *, strict: bool = True) -> bool:
    return equal(obj.doc, other.doc, strict=strict) \
        and (obj.i == other.i)
@iter_diffs.register
def _(obj: Token, other: Token, *, strict: bool = True) -> IDiffType:
    if not equal(obj, other, strict=strict):
        yield "TOKEN", obj, other
