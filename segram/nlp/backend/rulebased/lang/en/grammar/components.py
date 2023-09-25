from typing import Iterable
from spacy.tokens import Token
from .grammar import RulebasedEnglishGrammar
from ......grammar import ComponentNLP
from .......grammar.lang.en import EnglishComponent
from .......grammar.lang.en import EnglishVerb, EnglishNoun
from .......grammar.lang.en import EnglishPrep, EnglishDesc
from ......tokens import Token
from .......symbols import Dep, Tense, Modal, Mood


class RulebasedEnglishComponent(
    RulebasedEnglishGrammar,
    EnglishComponent, ComponentNLP
):
    """Rule-based English :mod:`spacy` grammar component."""
    __slots__ = ()

    @classmethod
    def find_neg(cls, tok: Token) -> Token | None:
        if tok.is_neg or tok.is_no:
            return tok
        return None

    @classmethod
    def find_qmark(cls, tok: Token) -> Token | None:
        if tok.is_qmark:
            return tok
        return None

    @classmethod
    def find_exclam(cls, tok: Token) -> Token | None:
        if tok.is_exclam:
            return tok
        return None

    @classmethod
    def find_intj(cls, tok: Token) -> Token | None:
        if tok.is_intj:
            return tok
        return None

    def get_dep(self, parent: EnglishComponent) -> Dep | None:
        """Get dependency between ``self`` and ``parent``."""
        # pylint: disable=too-many-return-statements
        # pylint: disable=too-many-branches
        tok = self.head
        head = parent.head
        dep = Dep(0)
        if tok.is_conj:
            dep |= Dep.conj
        if tok.is_preplike:
            return dep | Dep.prep
        if tok.is_subj:
            dep |= Dep.subj
        if tok.is_agent:
            return dep | Dep.agent
        if head.is_preplike:
            if tok.is_advmod:
                return dep | Dep.desc
            dep |= Dep.pobj
        if head.is_nounlike:
            if tok.is_adj or tok.is_poss or tok.is_amod or tok.is_acomp:
                dep |= Dep.desc
            if tok.is_acl:
                dep |= Dep.acl
            if tok.is_relcl:
                dep |= Dep.relcl
            if tok.is_noun_mod:
                dep |= Dep.nmod
            if tok.is_appos:
                dep |= Dep.appos
        if head.is_verblike:
            if tok.is_subjpass or tok.is_dobj:
                dep |= Dep.dobj
            if tok.is_iobj:
                dep |= Dep.iobj
            if tok.is_oprd or tok.is_attr or tok.is_acomp:
                dep |= Dep.adesc
            if (tok.is_adj or tok.is_nounlike) \
            and (tok.is_ccomp or tok.is_advcl):
                dep |= Dep.cdesc
            if tok.is_adv:
                dep |= Dep.desc
            if head.is_imp_mood and tok.is_npadvmod:
                dep |= Dep.subj
        if head.is_agent:
            dep |= Dep.subj
        if tok.is_verblike and not tok.is_acomp and not tok.is_xcomp \
        and not dep & (Dep.conj | Dep.desc):
            dep |= Dep.subcl
        if tok.is_xcomp:
            dep |= Dep.xcomp
        return dep or Dep.misc


class RulebasedEnglishVerb(
    RulebasedEnglishComponent, EnglishVerb
):
    """Rule-based English :mod:`spacy` verb component."""
    __slots__ = ()
    __inherit_from_lead__ = ("part",)

    @classmethod
    def is_head(cls, tok: Token) -> bool:
        return tok.is_vp_head

    @classmethod
    def get_tense(cls, verb: EnglishVerb) -> Tense:
        """Get tense of ``verb``."""
        for aux in verb.aux:
            if aux.tok.lemma_ in ("will", "shall"):
                return Tense.FUTURE
            if aux.tok.lemma_ == "have":
                return Tense.PAST
            tense = aux.tense
            if tense:
                return tense
        return verb.tok.tense or Tense.PRESENT

    @classmethod
    def get_modal(cls, verb: EnglishVerb) -> Tense:
        """Get modality of ``verb``."""
        for aux in verb.aux:
            lemma = aux.tok.lemma_
            if lemma in ("can", "could"):
                return Modal.ABILITY
            if lemma in ("may", "might"):
                return Modal.POSSIBILITY
            if lemma in ("must",):
                return Modal.NECESSITY
            if lemma in ("should", "ought"):
                return Modal.OBLIGATION
            if lemma in ("need",):
                return Modal.NEED
        return Modal.NULL

    @classmethod
    def get_mood(cls, verb: EnglishVerb) -> Mood:
        """Get mood of ``verb``."""
        if verb.tok.lead.is_root and "Inf" in verb.tok.morph.get("VerbForm"):
            return Mood.IMP
        return Mood.REAL

    @classmethod
    def find_part(cls, tok: Token) -> Token | None:
        """Find particle token."""
        if tok.is_part and not tok.is_neg:
            return tok
        return None

    @classmethod
    def find_aux(cls, tok: Token) -> tuple[Token, ...]:
        """Find (verb) auxiliary tokens."""
        if tok.is_aux_verb and not tok.is_part and not tok.is_conj:
            yield tok

    @classmethod
    def find_expl(cls, tok: Token) -> Token | None:
        """Find expletive token."""
        if tok.is_expl:
            return tok
        return None


class RulebasedEnglishNoun(
    RulebasedEnglishComponent, EnglishNoun
):
    """Rule-based English :mod:`spacy` noun component."""
    __slots__ = ()
    __inherit_from_lead__ = ("det",)

    @classmethod
    def is_head(cls, tok: Token) -> bool:
        return tok.is_np_head

    @classmethod
    def find_det(cls, tok: Token) -> Token | None:
        if tok.is_det and not tok.is_no:
            return tok
        return None


class RulebasedEnglishPrep(
    RulebasedEnglishComponent, EnglishPrep
):
    """Rule-based English :mod:`spacy` preposition component."""
    __slots__ = ()

    @classmethod
    def is_head(cls, tok: Token) -> bool:
        return tok.is_pp_head

    @classmethod
    def find_preps(cls, tok: Token) -> Iterable[Token]:
        """Find preposition chain."""
        if tok.is_prep and not tok.is_conj:
            yield tok
            for child in tok.children:
                yield from cls.find_preps(child)


class RulebasedEnglishDesc(
    RulebasedEnglishComponent, EnglishDesc
):
    """Rule-based English :mod:`spacy` description component."""
    __slots__ = ()
    __post_init__ = ("det",)

    @classmethod
    def is_head(cls, tok: Token) -> bool:
        return tok.is_dp_head

    @classmethod
    def find_mod(cls, tok: Token) -> Iterable[Token]:
        """Find modifier tokens."""
        if tok.is_desc_mod:
            yield tok

    @classmethod
    def find_det(cls, tok: EnglishDesc) -> Token | None:
        """Find determiner token."""
        for t in tok.mod:
            if t.is_desc_mod:
                for child in t.children:
                    if child.is_det:
                        return child
        return None
