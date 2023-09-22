# pylint: disable=no-name-in-module
from typing import Any, Optional, NamedTuple, Sequence, Iterable, Self
from collections import Counter
from spacy.tokens import Doc as SpacyDoc
from spacy.vocab import Vocab
from spacy.vectors import Vectors
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
    vocab
        Main vocabulary store of the corpus.
    vectors
        Word vector table. If then ``vocab.vectors`` are used
        as fallback if possible. If an instance of :class:`segram.language.Language`
        or :class:`segram.vocab.Vocab` is provided then the vector table
        is extracted automatically.
    resolve_coref
        If ``True`` then token coreferences are resolved when
        calculating token text and lemma frequency distributions.
    """
    def __init__(
        self,
        vocab: Vocab,
        *,
        vectors: Optional[Vectors] = None,
        resolve_coref: bool = True
    ) -> None:
        self._dmap = {}
        self.vocab = vocab
        self.dist = TokenDistributions(Counter(), Counter())
        if isinstance(vectors, Language):
            vectors = vectors.vocab.vectors
        elif isinstance(vectors, Vocab):
            vectors = vectors.vectors
        self._vectors = vectors
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

    @property
    def vectors(self) -> Vectors:
        return self.vectors or self.vocab.vectors

    @property
    def has_vectors(self) -> bool:
        return bool(self.vectors.data.size)

    # Methods -----------------------------------------------------------------

    def add_doc(self, doc: Doc) -> None:
        """Add document to the corpus."""
        if isinstance(doc, SpacyDoc):
            doc = getattr(doc._, settings.spacy_alias)
        if doc not in self:
            self._dmap[hash(doc)] = doc
            self.dist.text.update(t.coref.text for t in doc)
            self.dist.lemma.update(t.coref.lemma for t in doc)

    def add_docs(self, docs: Iterable[Doc]) -> None:
        """Add documents to the corpus."""
        for doc in docs:
            self.add_doc(doc)

    @classmethod
    def from_texts(
        cls,
        nlp: Language,
        *texts: str,
        pipe_kws: Optional[dict[str, Any]] = None,
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
        obj = cls(nlp.vocab, **kwds)
        pipe_kws = pipe_kws or {}
        obj.add_docs(
            getattr(d._, settings.spacy_alias)
            for d in nlp.pipe(texts, **pipe_kws)
        )
        return obj
