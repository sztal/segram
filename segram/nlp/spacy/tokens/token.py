# pylint: disable=too-many-public-methods,no-name-in-module
from __future__ import annotations
from typing import Any, Iterable
from spacy.tokens import Token, MorphAnalysis
from spacy.vocab import Vocab
from .abc import SpacyNLPToken
from ...abc import DocABC, SpanABC, TokenABC
from ....symbols import POS, Role


class SpacyTokenABC(TokenABC, SpacyNLPToken):
    """Enhanced token class."""
    __slots__ = ()

    # Properties --------------------------------------------------------------

    @property
    def i(self) -> int:
        return self.tok.i

    @property
    def text(self) -> str:
        return self.tok.text

    @property
    def whitespace(self) -> str:
        return self.tok.whitespace_

    @property
    def text_with_ws(self) -> str:
        return self.tok.text_with_ws

    @property
    def vocab(self) -> Vocab:
        return self.tok.vocab

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
    def doc(self) -> DocABC:
        return self.sns(self.tok.doc)

    @property
    def sent(self) -> SpanABC:
        return self.sns(self.tok.sent)

    @property
    def head(self) -> SpacyTokenABC:
        return self.sns(self.tok.head)

    @property
    def morph(self) -> MorphAnalysis:
        return self.tok.morph

    @property
    def left_edge(self) -> SpacyTokenABC:
        return self.sns(self.tok.left_edge)

    @property
    def right_edge(self) -> SpacyTokenABC:
        return self.sns(self.tok.right_edge)

    @property
    def ancestors(self) -> Iterable[SpacyTokenABC]:
        for tok in self.tok.ancestors:
            yield self.sns(tok)

    @property
    def conjuncts(self) -> tuple[SpacyTokenABC, ...]:
        return tuple(self.sns(c) for c in self.tok.conjuncts)

    @property
    def children(self) -> Iterable[SpacyTokenABC, ...]:
        for child in self.tok.children:
            yield self.sns(child)

    @property
    def lefts(self) -> Iterable[SpacyTokenABC, ...]:
        for tok in self.tok.lefts:
            yield self.sns(tok)

    @property
    def rights(self) -> Iterable[SpacyTokenABC, ...]:
        for tok in self.tok.rights:
            yield self.sns(tok)

    @property
    def subtree(self) -> Iterable[SpacyTokenABC, ...]:
        for tok in self.tok.subtree:
            yield self.sns(tok)

    # Methods -----------------------------------------------------------------

    def nbor(self, *args: Any, **kwds: Any) -> SpacyTokenABC:
        return self.sns(self.tok.nbor(*args, **kwds))

    def is_ancestor(self, descendant: SpacyTokenABC | Token) -> bool:
        if isinstance(descendant, SpacyTokenABC):
            return self.tok.is_ancestor(descendant.tok)
        return self.tok.is_ancestor(descendant)

    def has_morpth(self) -> bool:
        return self.tok.has_morph()
