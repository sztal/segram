# pylint: disable=abstract-method
from typing import Any, Iterable, Sequence, ClassVar
from abc import abstractmethod
from .grammar import GrammarNLP
from ..tokens import Token
from ...grammar import Component
from ...symbols import POS, Role, Dep


class ComponentNLP(GrammarNLP, Component):
    """Abstract base class for grammar components
    with NLP backend methods.
    """
    __slots__ = ()
    __post_init__ = ()
    __inherit_from_lead__ = ()
    post_init: ClassVar[tuple[str, ...]] = ()
    inherit_from_lead: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls.init_class_attrs({
            "__inherit_from_lead__": "inherit_from_lead",
            "__post_init__":  "post_init"
        }, check_slots=False)
        for typ, prefix, names in zip(
            ["component", "attr"],
            ["find", "get"],
            [cls.token_names, cls.attr_names]
        ):
            missing = [
                meth for n in names
                if not hasattr(cls, (meth := f"{prefix}_{n}"))
            ]
            if missing:
                raise TypeError(f"missing '{typ}' discovery methods: {missing}")

    # Abstract methods --------------------------------------------------------

    @classmethod
    @abstractmethod
    def is_head(cls, tok: Token) -> bool:
        """Test for head token."""
        raise NotImplementedError

    @abstractmethod
    def get_dep(self, parent: Component) -> Dep | None:
        """Get dependency between ``self`` and ``parent``."""
        raise NotImplementedError

    # Methods -----------------------------------------------------------------

    def is_child_of(self, comp: Component) -> bool:
        """Is ``self`` a child of ``comp``."""
        if not comp.head.is_ancestor(self.head):
            return False
        return comp.head == self.head.head and not self.head.is_root

    def find_parents(self, comps: Sequence[Component]):
        """Find parents of ``self`` contained in ``comps``."""
        for comp in comps:
            if self.is_child_of(comp):
                yield comp

    def get_sconj(self, parent: Component) -> Token | None:
        """Get conjunction subordinating ``self`` to ``parent``."""
        if self.head.head == parent.head:
            for tok in self.subtokens:
                if tok.is_sconj:
                    return tok
        return None

    def get_cconj(self, other: Component) -> Token | None:
        """Get conjunction token coordinating ``self`` and ``other``."""
        conjs = self.head.conjuncts
        if other.head not in conjs:
            return None
        for conj in (self.head, *conjs):
            for child in conj.children:
                if child.is_cconj:
                    return child
        return None

    @classmethod
    def from_tok(
        cls,
        tok: Token,
        pos: POS | None = None,
        role: Role | None = None,
        **kwds: Any
    ) -> Component:
        """Construct from a token.

        Parameters
        ----------
        tok
            NLP token object.
        pos
            POS tag to assign to the tok token.
            Determined automatically if ``None``.
        role
            Syntactic role assigned to the tok token
            and the entire component. Determined autoamtically if ``None``.
        **kwds
            Passed to :meth:`__init__`.
        """
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-locals
        if not cls.is_head(tok):
            return None

        def add_tok(tok, name, slots):
            if isinstance(tok, Iterable):
                tok = tuple(tok)
                if tok:
                    slots.setdefault(name, []).extend(tok)
                    return tok
            elif tok:
                slots[name] = tok
                return tok
            return None

        role = role or cls.__role__
        if isinstance(role, str):
            role = Role.from_name(role)
        typ = cls.get_comp_type(role, tok.pos)
        if typ is not cls:
            return typ.from_tok(tok, pos, role=role, **kwds)
        slots = {}
        finders = {
            name: getattr(cls, f"find_{name}")
            for name in cls.token_names
            if name not in cls.post_init
        }
        for child in tok.children:
            for name, finder in finders.items():
                if (v := slots.get(name)) and isinstance(v, Token):
                    continue
                if add_tok(finder(child), name, slots):
                    break
        # Apply finders to lead children for missing tokens -------------------
        if tok != (lead := tok.lead):
            for name in cls.inherit_from_lead:
                if name in slots:
                    continue
                finder = getattr(cls, f"find_{name}")
                slots[name] = next((finder(c) for c in lead.children), None)
        comp = cls(tok, role=role, **slots, **kwds)
        # Apply post-init finders ---------------------------------------------
        for name in comp.__class__.post_init:
            finder = getattr(comp.__class__, f"find_{name}")
            tok = add_tok(finder(comp), name, slots)
            setattr(comp, name, tok)
        # Get and set attributes ----------------------------------------------
        for attr in cls.attr_names:
            if attr not in kwds:
                val = getattr(cls, f"get_{attr}")(comp)
                setattr(comp, attr, val)
        return comp
