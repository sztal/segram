from __future__ import annotations
from typing import Any, Optional
from abc import abstractmethod
from ..base import NLPToken, attr
from ....symbols import POS, Role
from ....utils.colors import color_role
from ....utils.diff import iter_diffs, equal, IDiffType


class TokenABC(NLPToken):
    """Token abstract base class."""
    __slots__ = ()

    def __repr__(self) -> str:
        return self.to_str(color=True)

    def __hash__(self) -> int:
        return hash((hash(self.doc), self.i))

    def __eq__(self, other: TokenABC) -> bool:
        if (res := super().__eq__(other)) is NotImplemented:
            return res
        return res and self.i == other.i

    def __lt__(self, other: TokenABC) -> bool:
        """Is ``self`` earlier in the document than ``other``."""
        if self.is_comparable_with(other):
            return self.i < other.i
        return NotImplemented

    # Attr-properties ---------------------------------------------------------

    @attr
    @property
    @abstractmethod
    def i(self) -> int:
        """Token index within the document sequence."""

    @attr
    @property
    @abstractmethod
    def whitespace(self) -> str:
        """Whitespace following the token."""

    @attr
    @property
    @abstractmethod
    def lemma(self) -> str:
        """Lemmatized token text."""

    @attr
    @property
    @abstractmethod
    def pos(self) -> POS:
        """Part-of-speech tag (UDEP)."""

    @attr
    @property
    @abstractmethod
    def role(self) -> Role:
        """Fixed token role."""

    @attr
    @property
    @abstractmethod
    def ent(self) -> Role:
        """Named entity type."""

    @attr
    @property
    @abstractmethod
    def corefs(self) -> Optional[tuple[TokenABC, ...]]:
        """Coreference tokens."""

    # Properties --------------------------------------------------------------

    @property
    @abstractmethod
    def text_with_ws(self) -> str:
        """Token text with following whitespace."""

    @property
    @abstractmethod
    def is_negation(self) -> bool:
        """Is negation token."""

    @property
    @abstractmethod
    def is_qmark(self) -> bool:
        """Is question mark token."""

    @property
    @abstractmethod
    def is_exclam(self) -> bool:
        """Is exclamation mark token."""

    @property
    @abstractmethod
    def is_intj(self) -> bool:
        """Is interjection token."""

    @property
    @abstractmethod
    def sent(self) -> "SpanABC":
        """Sentence object containing the token."""

    # Properties --------------------------------------------------------------

    @property
    def coref(self) -> TokenABC:
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


# Comparison functions for testing --------------------------------------------

@equal.register
def _(obj: TokenABC, other: TokenABC, *, strict: bool = True) -> bool:
    return equal(obj.doc, other.doc, strict=strict) \
        and (obj.i == other.i)
@iter_diffs.register
def _(obj: TokenABC, other: TokenABC, *, strict: bool = True) -> IDiffType:
    if not equal(obj, other, strict=strict):
        yield "TOKEN", obj, other
