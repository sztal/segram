from typing import Any, Sequence, Iterable, Self, Literal
from collections import Counter
from spacy.tokens import Doc as SpacyDoc
from spacy.language import Language
from tqdm.auto import tqdm
from .tokens import Doc, Token
from .. import settings
from ..datastruct import DataIterable, DataTuple


class Corpus(Sequence):
    """Corpus class.

    Attributes
    ----------
    token_dist
        Token distribution.
    count
        Count raw words, lowercased words or lemmas.
    resolve_coref
        If ``True`` then token coreferences are resolved when
        calculating token text and lemma frequency distributions.
    """
    _count_vals = ("words", "lower", "lemmas")

    def __init__(
        self,
        nlp: Language,
        *,
        count_method: Literal[*_count_vals] = "lemmas",
        resolve_coref: bool = True
    ) -> None:
        self._check_count_method(count_method)
        self._dmap = {}
        self.nlp = nlp
        self.token_dist = Counter()
        self.count_method = count_method
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

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        ndoc = len(self)
        count = self.count_method
        at = hex(id(self))
        dword = "doc" if ndoc == 1 else "docs"
        return f"<{cn} with {ndoc} {dword} and count_method=\"{count}\" at {at}>"

    # Properties --------------------------------------------------------------

    @property
    def docs(self) -> DataIterable[Doc]:
        return DataIterable(self._dmap.values())

    # Methods -----------------------------------------------------------------

    def add_doc(self, doc: Doc | str) -> None:
        """Add document to the corpus.

        The method recognizes identical documents
        and do not add the same ones more than once.
        The identity check is based on :meth:`segram.nlp.Doc.id`.

        See also
        --------
        segram.nlp.Doc.id : persistent document identifier.
        segram.nlp.Doc.coredata : data used to generate the identifier.
        """
        if isinstance(doc, str):
            doc = self.nlp(doc)
        if isinstance(doc, SpacyDoc):
            doc = getattr(doc._, settings.spacy_alias)
        if doc not in self:
            self._dmap[doc.id] = doc
            self.token_dist += self._count_toks(doc)

    def add_docs(
        self,
        docs: Iterable[Doc | str],
        *,
        progress: bool = False,
        **kwds: Any
    ) -> None:
        """Add documents to the corpus.

        ``**kwds`` are passed to :func:`tqdm.tqdm` with
        ``progress`` used to switch the progress bar
        (i.e. it is used as ``disable=not progress``).
        """
        for doc in tqdm(docs, disable=not progress, **kwds):
            self.add_doc(doc)

    def count_tokens(self, what: Literal[_count_vals]) -> None:
        """(Re)count tokens.

        ``what`` specifies what kind of tokens should be counted.
        Recount is done only when necessary, i.e. when the call
        changes the previous count_method method.
        """
        self._check_count_method(what)
        if what != self.count_method:
            self.count_method = what
            self.token_dist = Counter()
            for doc in self:
                self.token_dist += self._count_toks(doc)

    def copy(self) -> Self:
        """Make a copy.

        Language model object is passed but not copied.
        Document objects are copied.
        """
        # pylint: disable=protected-access
        kwds = {
            "count_method": self.count_method,
            "resolve_coref": self.resolve_coref
        }
        obj = self.__class__(self.nlp, **kwds)
        obj._dmap = { idx: doc.copy() for idx, doc in self._dmap.items() }
        obj.token_dist = self.token_dist.copy()
        return obj

    @classmethod
    def from_texts(
        cls,
        nlp: Language,
        *texts: str,
        pipe_kws: dict[str, Any] | None = None,
        progress: bool = False,
        tqdm_kws: dict[str, Any] | None = None,
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
        obj = cls(nlp, **kwds)
        pipe_kws = pipe_kws or {}
        tqdm_kws = tqdm_kws or {}
        obj.add_docs((
            getattr(d._, settings.spacy_alias)
            for d in nlp.pipe(texts, **pipe_kws)
        ), progress=progress, **tqdm_kws)
        return obj

    # Internals ---------------------------------------------------------------

    def _count_toks(self, toks: Iterable[Token]) -> Counter:
        toks = DataTuple(toks)
        if self.resolve_coref:
            toks = toks.get("coref")
        if self.count_method == "lemma":
            toks = toks.get("lemma")
        else:
            toks = toks.get("text")
            if self.count_method == "lower":
                toks.map(str.lower)
        return Counter(toks)

    def _check_count_method(self, what: str) -> None:
        if what not in self._count_vals:
            raise ValueError(f"'count' has to be one of {self._count_vals}")
