# pylint: disable=no-name-in-module,too-many-public-methods
# pylint: disable=consider-using-ternary
from __future__ import annotations
from typing import Self
from spacy.symbols import DET, NUM, neg, det, expl
from spacy.symbols import PRON, NOUN, PROPN
from spacy.symbols import nsubj, nsubjpass, csubj, csubjpass
from spacy.symbols import agent, appos
from spacy.symbols import dobj, iobj
from spacy.symbols import VERB, AUX, PART, auxpass
from spacy.symbols import ADJ, amod
from spacy.symbols import ADV, advmod
from spacy.symbols import attr
from spacy.symbols import poss, nmod, npadvmod
from spacy.symbols import ADP, prep, pobj
from spacy.symbols import relcl, acl, advcl
from spacy.symbols import ccomp, pcomp, acomp, xcomp
from spacy.symbols import CCONJ, SCONJ, conj, mark, preconj
from spacy.symbols import INTJ
from ......tokens import Token
from .......symbols import Tense


class RulebasedEnglishToken(Token):
    """Enhanced token class for rulebased English grammar."""
    __slots__ = ()

    @property
    def lead(self) -> Self:
        # pylint: disable=redefined-outer-name
        for conj in self.conjuncts:
            if not conj.is_conj:
                return conj
        return self

    @property
    def tense(self) -> Tense:
        t = self.tok.morph.get("Tense")
        if "Past" in t:
            return Tense.PAST
        if "Pres" in t:
            return Tense.PRESENT
        return None

    # Flags for nouns and noun components -------------------------------------

    # Noun-like flags
    @property
    def is_pron(self) -> bool:
        return self.tok.pos == PRON
    @property
    def is_noun(self) -> bool:
        return self.tok.pos in (NOUN, PROPN)
    @property
    def is_num(self) -> bool:
        return self.tok.pos == NUM
    @property
    def is_num_noun(self) -> bool:
        return self.is_num and not self.is_nummod
    @property
    def is_nounlike(self) -> bool:
        return (self.is_noun or self.is_pron or self.is_num_noun)

    # Component head flags
    @property
    def is_np_head(self) -> bool:
        return (self.is_nounlike or self.is_noun_mod) \
            and not self.is_poss and not self.is_expl \
            and not self.is_desc_mod

    # Noun-modifier flags
    @property
    def is_nmod(self) -> bool:
        return self.tok.dep == nmod
    @property
    def is_npadvmod(self) -> bool:
        return self.tok.dep == npadvmod
    @property
    def is_compound(self) -> bool:
        return self.tok.dep_ == "compound"
    @property
    def is_nummod(self) -> bool:
        return self.tok.dep_ == "nummod"
    @property
    def is_noun_mod(self) -> bool:
        return self.is_nmod or self.is_nummod or self.is_npadvmod \
            or self.is_compound

    # Flags for verbs and verb components -------------------------------------

    # Verb-like flags
    @property
    def is_verb(self) -> bool:
        return self.tok.pos == VERB
    @property
    def is_aux(self) -> bool:
        return self.tok.pos == AUX
    @property
    def is_auxpass(self) -> bool:
        return self.tok.dep == auxpass
    @property
    def is_verblike(self) -> bool:
        return self.is_verb or self.is_aux or self.is_auxpass
    @property
    def is_aux_verb(self) -> bool:
        return (self.is_aux or self.is_auxpass) \
            and self.head.is_verblike \
            and not self.is_root \
            and not self.is_ccomp \
            and not self.is_advcl
    # Verb object descrition flags
    @property
    def is_oprd(self) -> bool:
        return self.dep == "oprd"
    @property
    def is_obj_desc(self) -> bool:
        return (head := self.head).is_verblike and self != head and (
            self.is_adj or self.is_acomp or self.is_oprd or self.is_attr \
            or (self.is_ccomp and not self.is_verblike)
        )
    # Component head flags
    @property
    def is_vp_head(self) -> bool:
        return not self.lead.is_aux_verb and self.is_verblike \
            and not self.is_amod \
            and not self.is_prep
    @property
    def is_desc_vp_head(self) -> bool:
        return self.is_vp_head and (
            any(c.is_obj_desc for c in self.children) \
            or (lead := self.lead).is_advcl and lead.head.lead.is_advcl
        )

    # Subject flags -----------------------------------------------------------

    @property
    def is_nsubj(self) -> bool:
        return self.tok.dep == nsubj
    @property
    def is_csubj(self) -> bool:
        return self.tok.dep == csubj
    @property
    def is_subj(self) -> bool:
        return self.is_nsubj or self.is_csubj
    @property
    def is_nsubjpass(self) -> bool:
        return self.tok.dep == nsubjpass
    @property
    def is_csubjpass(self) -> bool:
        return self.tok.dep == csubjpass
    @property
    def is_subjpass(self) -> bool:
        return self.is_nsubjpass or self.is_csubjpass
    @property
    def is_ccomp_nsubj(self) -> bool:
        return self.is_nsubj and self.head.is_ccomp

    # Object flags ------------------------------------------------------------

    @property
    def is_dobj(self) -> bool:
        return self.tok.dep == dobj
    @property
    def is_iobj(self) -> bool:
        return self.tok.dep == iobj or self.tok.dep_ == "dative"
    @property
    def is_pobj(self) -> bool:
        return self.tok.dep == pobj

    # Flags for prepositions and prepositional components ---------------------

    @property
    def is_adp(self) -> bool:
        return self.tok.pos == ADP
    @property
    def is_prep(self) -> bool:
        return self.tok.dep == prep
    @property
    def is_preplike(self) -> bool:
        return (self.is_adp and not self.is_agent) or self.is_prep
    @property
    def is_pp_head(self) -> bool:
        return (self.is_preplike or self.is_agent) \
            and not self.lead.head.is_preplike

    # Flags for descriptive components ----------------------------------------

    @property
    def is_adj(self) -> bool:
        return self.tok.pos == ADJ
    @property
    def is_adv(self) -> bool:
        return self.tok.pos == ADV
    @property
    def is_advmod(self) -> bool:
        return self.tok.dep == advmod
    @property
    def is_amod(self) -> bool:
        return self.tok.dep == amod
    @property
    def is_poss(self) -> bool:
        return self.tok.dep == poss
    @property
    def is_appos(self) -> bool:
        return self.tok.dep == appos
    # Description flags
    @property
    def is_dp_head(self) -> bool:
        return not (lead := self.lead).is_neg \
        and not lead.is_desc_mod and (
            self.is_adj or self.is_adv or self.is_poss \
            or (self.is_verblike and self.is_amod)
        )
    # Description modifier flags
    @property
    def is_desc_mod(self) -> bool:
        return ((head := self.head).is_adv or head.is_adj) \
        and (self.is_advmod or self.is_npadvmod or self.is_amod)

    # Clause flags ------------------------------------------------------------

    @property
    def is_acl(self) -> bool:
        return self.tok.dep == acl
    @property
    def is_advcl(self) -> bool:
        return self.tok.dep == advcl
    @property
    def is_relcl(self) -> bool:
        return self.tok.dep == relcl

    # Complement flags --------------------------------------------------------

    @property
    def is_xcomp(self) -> bool:
        return self.tok.dep == xcomp
    @property
    def is_ccomp(self) -> bool:
        return self.tok.dep == ccomp
    @property
    def is_acomp(self) -> bool:
        return self.tok.dep == acomp
    @property
    def is_pcomp(self) -> bool:
        return self.tok.dep == pcomp

    # Conjunction flags -------------------------------------------------------

    @property
    def is_preconj(self) -> bool:
        return self.tok.dep == preconj
    @property
    def is_cconj(self) -> bool:
        return self.tok.pos == CCONJ and not self.is_preconj
    @property
    def is_sconj(self) -> bool:
        return self.tok.pos == SCONJ
    @property
    def is_conj(self) -> bool:
        return self.tok.dep == conj
    @property
    def is_mark(self) -> bool:
        return self.tok.dep == mark

    # Other flags -------------------------------------------------------------

    @property
    def is_root(self) -> bool:
        return self.tok.dep_ == "ROOT"
    @property
    def is_imp_mood(self) -> bool:
        return self.lead.is_root and "Inf" in self.morph.get("VerbForm")
    @property
    def is_det(self) -> bool:
        return self.tok.pos == DET
    @property
    def is_neg(self) -> bool:
        return self.tok.dep == neg
    @property
    def is_cconj_neg(self) -> bool:
        return (self.is_cconj and self.tok.lemma_ == "nor") \
            or (self.is_preconj and self.tok.lemma_ == "neither")
    @property
    def is_no(self) -> bool:
        return self.tok.dep == det and self.tok.lemma_ in ("no", "never")
    @property
    def is_negation(self) -> bool:
        return self.is_neg or self.is_no or self.is_cconj_neg
    @property
    def is_part(self) -> bool:
        return self.tok.pos == PART
    @property
    def is_agent(self) -> bool:
        return self.tok.dep == agent
    @property
    def is_expl(self) -> bool:
        return self.tok.dep == expl
    @property
    def is_attr(self) -> bool:
        return self.tok.dep == attr
    @property
    def is_punct(self) -> bool:
        return self.tok.is_punct
    @property
    def is_qmark(self) -> bool:
        return self.is_punct and self.tok.lemma_ == "?"
    @property
    def is_exclam(self) -> bool:
        return self.is_punct and self.tok.lemma_ == "!"
    @property
    def is_intj(self) -> bool:
        return self.tok.pos == INTJ
