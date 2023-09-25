from typing import Any, NamedTuple, Sequence, Iterable, Self
from collections import Counter
from spacy.tokens import Doc as SpacyDoc
from spacy.language import Language
from .tokens import Doc
from .. import settings


class TokenDistributions(NamedTuple):
    text: Counter
    lemma: Counter


class Corpus(Sequence):
    """Corpus class.

    Attributes
    ----------
    resolve_coref
        If ``True`` then token coreferences are resolved when
        calculating token text and lemma frequency distributions.
    """
    def __init__(self, *, resolve_coref: bool = True) -> None:
        self._dmap = {}
        self.dist = TokenDistributions(Counter(), Counter())
        self.resolve_coref = resolve_coref

    def __getitem__(self, idx: int | slice) -> Doc | tuple[Doc, ...]:
        return self.docs[idx]

    def __len__(self) -> int:
        return len(self._dmap)

    def __contains__(self, doc: Doc) -> bool:
        if isinstance(doc, Doc):
            return hash(doc) in self._dmap
        cn = self.__class__.__name__
        dn = doc.__class__.__name__
        raise NotImplementedError(f"'{cn}' cannot contain '{dn}' objects")

    # Properties --------------------------------------------------------------

    @property
    def docs(self) -> tuple[Doc]:
        return tuple(self._dmap.values())

    # Methods -----------------------------------------------------------------

    def add_doc(self, doc: Doc) -> None:
        """Add document to the corpus."""
        if isinstance(doc, SpacyDoc):
            doc = getattr(doc._, settings.spacy_alias)
        if doc not in self:
            self._dmap[hash(doc)] = doc
            if self.resolve_coref:
                self.dist.text.update(t.coref.text for t in doc)
                self.dist.lemma.update(t.coref.lemma for t in doc)
            else:
                self.dist.text.update(t.text for t in doc)
                self.dist.lemma.update(t.lemma for t in doc)

    def add_docs(self, docs: Iterable[Doc]) -> None:
        """Add documents to the corpus."""
        for doc in docs:
            self.add_doc(doc)

    @classmethod
    def from_texts(
        cls,
        nlp: Language,
        *texts: str,
        pipe_kws: dict[str, Any] | None = None,
        **kwds: Any
    ) -> Self:
        """Construct from texts.

        Parameters
        ----------
        nlp
            Language model to use to parse texts.
        *texts
            Texts to parse.
        pipe_kws
            Keyword arguments passed to :meth:`spacy.language.Language.pipe`.
        **kwds
            Passed :meth:`__init__`.
            Vocabulary is taken from the language model.
        """
        obj = cls(**kwds)
        pipe_kws = pipe_kws or {}
        obj.add_docs(
            getattr(d._, settings.spacy_alias)
            for d in nlp.pipe(texts, **pipe_kws)
        )
        return obj
