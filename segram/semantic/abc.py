from __future__ import annotations
from typing import Iterable, Self, Any, Optional, ClassVar
from abc import abstractmethod
from itertools import islice
from ..abc import SegramABC
from ..grammar import Phrase
from ..nlp.tokens import TokenABC
from ..symbols import Role
from ..utils.types import Namespace


class SemanticNamespace(Namespace):
    Actant: type[SemanticElement]
    Event: type[SemanticElement]


class Semantic(SegramABC):
    """Abstract base class for semantic classes."""
    types: ClassVar[SemanticNamespace] = SemanticNamespace()
    __slots__ = ()


class SemanticElement(Semantic):
    """Semantic element class.

    Attributes
    ----------
    story
        Story object.
    base
        Base phrase.
    end
        End phrase.
    """
    alias: ClassVar[str] = "SElem"
    __slots__ = ("_story", "base", "end")
    reverse_base: ClassVar[bool] = False

    def __init__(
        self,
        story: "Story",
        base: Phrase,
        end: Optional[Phrase] = None
    ) -> None:
        self._story = story
        if self.reverse_base and end is not None:
            base, end = end, base
        self.base = base
        self.end = end

    def __new__(cls, *args: Any, **kwds: Any) -> None:
        obj = super().__new__(cls)
        obj.__init__(*args, **kwds)
        if (cur := obj.story.emap.get(obj.base)):
            cur.__init__(obj.story, **obj.data)
            return cur
        obj.story.emap[obj.base] = obj
        return obj

    def __repr__(self) -> str:
        return self.to_str(color=True)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if (alias := getattr(cls, "alias", None)):
            cls.types[alias] = cls

    # Properties --------------------------------------------------------------

    @property
    def story(self) -> "Story":
        return self._story

    @property
    def depth(self) -> int:
        """Phrasal depth of the base phrase."""
        return self.base.depth

    @property
    def head(self) -> int:
        """Head component of the base phrase."""
        return self.base.head

    # Constructors ------------------------------------------------------------

    @classmethod
    def from_phrase(cls, story: "Story", phrase: Phrase) -> Iterable[Self]:
        """Construct from phrase."""
        if cls.matches(phrase):
            if (extend := getattr(cls, "extend", None)):
                has_end = False
                for sub in extend(phrase):
                    if (ends := getattr(cls, "ends", None)):
                        if ends(sub):
                            yield cls(story, phrase, sub)
                            has_end = True
                    else:
                        cn = cls.__name__
                        raise NotImplementedError(
                            f"'{cn}' implements 'extend' but not 'ends'"
                        )
                if not has_end:
                    yield cls(story, phrase, None)
            else:
                yield cls(story, phrase, None)

    # Methods -----------------------------------------------------------------

    @classmethod
    @abstractmethod
    def matches(cls, phrase: Phrase) -> bool:
        """Check if phrase matches the base requirements of element."""

    @classmethod
    def iter_subtree(cls, phrase: Phrase) -> Iterable[Phrase]:
        """Iterate over the proper subtree of phrase."""
        yield from islice(phrase.subtree, 1, None)

    @classmethod
    def iter_suptree(cls, phrase: Phrase) -> Iterable[Phrase]:
        """Iterate over the proper supertree of phrase."""
        yield from islice(phrase.suptree, 1, None)

    def to_str(self, *, color: bool = False, **kwds: Any) -> str:
        """Represent as string.

        Arguments are passed to :meth:`segram.nlp.TokenABC.to_str`.
        """
        return self.stringify_tokens(
            self.prepare_token_roles(self.iter_token_roles()),
            color=color, **kwds
        )

    def is_comparable_with(self, other: Any) -> bool:
        return isinstance(other, SemanticElement)

    def iter_token_roles(self, **kwds: Any) -> Iterable[tuple[TokenABC, Role | None]]:
        """Iterate over token-role pairs.

        ``**kwds`` are passed to :meth:`segram.grammar.Phrase.iter_token_roles`.
        """
        yield from self.base.iter_token_roles(**kwds)
        if self.end:
            yield from self.end.iter_token_roles(**kwds)

    def prepare_token_roles(
        self,
        *tokroles: Iterable[tuple[TokenABC, Role | None]],
        role: Optional[Role] = None
    ) -> tuple[tuple[TokenABC, Role | None]]:
        """Prepare and sort token-role pairs before printing.

        Parameters
        ----------
        *tokroles
            Iterables with token-role pairs.
        role
            Optional role to superimpose on all tokens.
        """
        seen = set()
        show = []
        for tr in tokroles:
            for t, r in tr:
                if t in seen:
                    continue
                seen.add(t)
                show.append((t, r))
        tr = sorted(show, key=lambda x: x[0])
        if role is None:
            yield from tr
        else:
            for t, _ in tr:
                yield t, role

    def stringify_tokens(
        self,
        tokroles: Iterable[tuple[TokenABC, Role | None]],
        **kwds: Any
    ) -> str:
        """Stringify tokens from token-role pairs.

        tokroles
            Iterable with token-role pairs.
        **kwds
            Passed to :meth:`nlp.tokens.TokenABC.to_str`.
        """
        return " ".join(t.to_str(role=r, **kwds) for t, r in tokroles)
