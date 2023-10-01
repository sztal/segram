"""Symbols are used for denoting discrete enumerable types
of entities of a given class, for instance semantic roles
such as subjects or direct objects of verb.

The advantage of using symbols, implemented based on :class:`enum.Flag`,
is that it allows combining and filtering based on symbols using binary
boolean operators.
"""
from typing import Self
from enum import Flag, auto

__all__ = ("POS", "Role", "Tense", "Modal", "Mood")


class Symbol(Flag):
    """Base class for symbols."""
    # pylint: disable=function-redefined,invalid-overridden-method
    def __str__(self) -> str:
        if self.value == 0:
            return ""
        return super().__str__().split(".")[-1].lower()

    @property
    def name(self) -> str:
        return str(self).lower() or None

    @classmethod
    def from_name(cls, name: str) -> Self:
        neg = False
        if name.startswith("~"):
            name = name[1:]
            neg = True
        parts = name.split("|")
        sym =  cls(0)
        for part in parts:
            sym |= getattr(cls, part)
        return ~sym if neg else sym


class POS(Symbol):
    """Universal dependencies POS tags.

    See definitions `here <https://universaldependencies.org/u/pos/>`_.
    """
    # Open class words
    ADJ   = auto()
    ADV   = auto()
    INTJ  = auto()
    NOUN  = auto()
    PROPN = auto()
    VERB  = auto()
    # Closed class words
    ADP   = auto()
    AUX   = auto()
    CCONJ = auto()
    DET   = auto()
    NUM   = auto()
    PART  = auto()
    PRON  = auto()
    SCONJ = auto()
    # Other
    PUNCT = auto()
    SYM   = auto()
    X     = auto()
    # Combinations and aliases
    OTHER    = X

    @classmethod
    def from_name(cls, name: str) -> Self:
        return super().from_name(name.upper())


class Role(Symbol):
    """Syntactic role symbols.

    Unlike POS tags (and standard syntactic dependencies),
    syntactic roles are not fixed but may be phrase-specific.
    Thus, the same token can play different syntactic roles in
    different phrases, e.g. it can be a direct object in one verb
    phrase and a subject in another.

    Moreover, roles are used for marking tokens of which functions
    can not be simply determined by their POS or dependency tags,
    e.g. negations, even though such roles may often be fixed and
    not phrase-specific.

    Notes
    -----
    Currently, roles are used primarily for printing purposes,
    i.e. coloring tokens in console outputs, and therefore
    also selected, most important, roles are defined. Moreover,
    it is still an open question what roles should be defined.

    Attributes
    ----------
    VERB
        Verb or a verb-like predicate.
    NOUN
        Noun, typically the head of a noun phrase.
    SUBJ
        Subject (active, passive, nominal or clausal etc.).
    DOBJ
        Direct object of a verb or a description.
    IOBJ
        Indirect object of a verb (dative).
    PREP
        Preposition.
    POBJ
        Object of a preposition.
    PROOT
        Root of a preposition.
    DESC
        Description. This includes adjectives, adverbs,
        and adjectival and adverbial modifiers as well as
        any other sort of construction used to directly
        describe nouns, verbs and prepositions.
    BG
        Background element that should be not emphasized
        visually when printing, e.g. printed in gray.
    NEG
        Negation.
    QMARK
        Question mark.
    EXCLAM
        Exclamation mark.
    INTJ
        Interjection.
    """
    # Component-specific roles
    VERB   = auto()
    NOUN   = auto()
    SUBJ   = auto()
    DOBJ   = auto()
    IOBJ   = auto()
    PREP   = auto()
    POBJ   = auto()
    PROOT  = auto()
    DESC   = auto()
    BG     = auto()
    # Fixed roles
    NEG    = auto()
    QMARK  = auto()
    EXCLAM = auto()
    INTJ   = auto()

    @classmethod
    def from_name(cls, name: str) -> POS:
        return super().from_name(name.upper())


class Dep(Symbol):
    """Component dependency symbols.

    Attributes
    ----------
    root
        Sentence root.
    subj
        Subject.
    dobj
        Direct object.
    iobj
        Indirect object.
    pobj
        Prepositional object.
    prep
        Preposition.
    subcl
        Subclause.
    relcl
        Relative clause.
    acl
        Clausal modifier of noun (adnominal clause).
    xcomp
        Open clausal complement.
    desc
        Description.
    cdesc
        Clausal description.
    adesc
        Adjectival complement description.
    nmod
        Modifier of nominal.
    appos
        Appositional modifier.
    agent
        Agent token (introducing passive subjects).
    conj
        Conjunct.
    misc
        Miscellaneous (all other dependency roles).
    """
    # pylint: disable=invalid-name
    root   = auto()
    subj   = auto()
    dobj   = auto()
    iobj   = auto()
    pobj   = auto()
    prep   = auto()
    subcl  = auto()
    relcl  = auto()
    acl    = auto()
    xcomp  = auto()
    desc   = auto()
    adesc  = auto()
    cdesc  = auto()
    nmod   = auto()
    appos  = auto()
    agent  = auto()
    conj   = auto()
    misc   = auto()

    @classmethod
    def from_name(cls, name: str) -> POS:
        return super().from_name(name.lower())

    @property
    def role(self) -> Role | None:
        # pylint: disable=too-many-return-statements
        names = self.name.split("|")
        for name in names:
            try:
                if "subj" in name:
                    return Role.SUBJ
                if "obj" in name:
                    return Role.from_name(name)
                if "desc" in name:
                    return Role.DESC
                if name.endswith("cl"):
                    return Role.VERB
                if name in ("nmod", "appos"):
                    return Role.NOUN
                if name in ("agent", "prep"):
                    return Role.PREP
            except AttributeError:
                pass
        return None


class Tense(Symbol):
    """Tense symbols.

    Attributes
    ----------
    PAST
        Past tense
    PRESENT
        Present tense.
    FUTURE
        Future tense.
    """
    PAST    = auto()
    PRESENT = auto()
    FUTURE  = auto()

    @classmethod
    def from_name(cls, name: str) -> POS:
        return super().from_name(name.upper())


class Modal(Symbol):
    """Modal symbols.

    Attributes
    ----------
    NULL
        No modality.
    ABILITY
        Ability mood (e.g. 'can' or 'could' in English).
    POSSIBILITY
        POssibility mode (e.g. 'may' or 'might' in English).
    NECESSITY
        Necessity mode (e.g. 'must' in English).
    OBLIGATION
        Obligation mode (e.g. 'shoud' or 'ought to' in English).
    NEED
        Need moge (e.g. 'need' in English).
    """
    NULL = auto()
    ABILITY = auto()
    POSSIBILITY = auto()
    NECESSITY = auto()
    OBLIGATION = auto()
    NEED = auto()

    @classmethod
    def from_name(cls, name: str) -> POS:
        return super().from_name(name.upper())


class Mood(Symbol):
    """Grammatical mood.

    Currently only few selected moods are implemented.

    Attributes
    ----------
    REAL
        Standard indicative mood (realis).
    IMP
        Imperative mood.
    """
    REAL = auto()
    IMP = auto()

    @classmethod
    def from_name(cls, name: str) -> POS:
        return super().from_name(name.upper())
