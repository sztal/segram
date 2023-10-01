# pylint: disable=no-name-in-module
from typing import Any, Iterable, Self, Literal, Mapping
import os
import pickle
from collections import Counter
from importlib import import_module
from spacy.tokens import Doc as SpacyDoc, DocBin
from spacy.language import Language
from spacy.vocab import Vocab
from tqdm.auto import tqdm
from .tokens import Doc, Token
from ..datastruct import DataIterator, DataTuple
from ..nlp.pipeline.base import Segram
from ..utils.misc import prefer_gpu_vectors, ensure_cpu_vectors
from .. import __title__


class Corpus(Mapping):
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
    _attrs = (
        "HEAD", "TAG", "POS", "DEP", "LEMMA",
        "MORPH", "ENT_IOB", "ENT_TYPE", "ENT_KB_ID"
    )

    def __init__(
        self,
        vocab: Vocab,
        nlp: Language | None = None,
        *,
        count_method: Literal[*_count_vals] = "lemmas",
        resolve_coref: bool = True
    ) -> None:
        self._check_count_method(count_method)
        self._dmap = {}
        self.vocab = vocab
        self.nlp = nlp
        self.token_dist = Counter()
        self.count_method = count_method
        self.resolve_coref = resolve_coref
        self.meta = None

    def __getitem__(self, key: int) -> Doc:
        return self._dmap[key]

    def __len__(self) -> int:
        return len(self._dmap)

    def __iter__(self) -> int:
        yield from self._dmap

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
    def docs(self) -> DataIterator[Doc]:
        return DataIterator(self._dmap.values())

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

        Raises
        ------
        AttributeError
            If a language model is not defined under the attribute
            ``self.nlp``.
        """
        if isinstance(doc, str):
            if not self.nlp:
                raise AttributeError(
                    "corpus has been initialized without language model, ",
                    "so documents passed as strings cannot be parsed."
                )
            doc = self.nlp(doc)
        alias = getattr(doc._, __title__+"_alias")
        if not self.meta:
            self.meta = getattr(doc._, alias+"_meta")
        if isinstance(doc, SpacyDoc):
            doc = getattr(doc._, alias+"_sns")
        if doc not in self:
            self._dmap[doc.id] = doc.grammar
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

    def get_docbin(
        self,
        attrs: Iterable[str] = _attrs,
        user_data: bool = True
    ) -> DocBin:
        """Get documents packed as :class:`spacy.tokens.DocBin`.

        Parameters
        ----------
        attrs
            Token attributes to serialize.
        user_data
            Should user data be stored.
            Setting to ``True`` requires clearing the cached grammar
            objects linked to all tokens, spans and docs to allow for
            serialization. This does not affect any functionalities
            of existing documents, but temporarily affects performance
            as the cache must be first reconstructed during further use.
        """
        if user_data:
            for doc in self.docs:
                Doc.clear_user_data(doc.doc.tok.user_data)
        docs = self.docs.get("doc").get("tok")
        dbin = DocBin(attrs, store_user_data=user_data, docs=docs)
        return dbin

    def ensure_cpu_vectors(self) -> None:
        """Ensure that word vectors are stored on CPU."""
        ensure_cpu_vectors(self.vocab)

    def prefer_gpu_vectors(self, *args: Any, **kwds: Any) -> bool:
        """Put word vectors on GPU if possible.

        Arguments are passed to :func:`segram.utils.misc.prefer_gpu_vectors`.
        """
        prefer_gpu_vectors(self.vocab, *args, **kwds)

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
        obj = cls(nlp.vocab, nlp, **kwds)
        pipe_kws = pipe_kws or {}
        tqdm_kws = tqdm_kws or {}
        obj.add_docs(nlp.pipe(texts, **pipe_kws), progress=progress, **tqdm_kws)
        return obj

    def to_data(
        self,
        *,
        vocab: bool = True,
        nlp: bool = False
    ) -> dict[str, Any]:
        """Dump to data dictionary.

        Parameters
        ----------
        vocab
            Should ``self.vocab`` be used.
        nlp
            Should ``self.nlp`` be used.
        """
        data = {
            "token_dist": dict(self.token_dist),
            "count_method": self.count_method,
            "resolve_coref": self.resolve_coref,
            "meta": self.meta
        }
        if vocab:
            data["vocab"] = self.vocab.to_bytes()
        if nlp and self.nlp:
            data["nlp"] = {
                "module": self.nlp.__class__.__module__,
                "name": self.nlp.__class__.__name__,
                "data": self.nlp.to_bytes()
            }
        if self._dmap:
            data["docs"] = self.get_docbin().to_bytes()
        return data

    @classmethod
    def from_data(cls, data: dict[str, Any], **kwds: Any) -> Self:
        """Construct from data dictionary.

        ``**kwds`` are passed to :meth:`add_docs`.
        """
        # pylint: disable=no-value-for-parameter
        meta = data.pop("meta")
        grammar, lang = meta["segram_grammar"].split(".")
        alias = meta["segram_alias"]
        Segram.import_extensions(grammar, lang, alias).register()
        vocab = data["vocab"]
        if not isinstance(vocab, Vocab):
            vocab = Vocab().from_bytes(vocab)
        data["vocab"] = vocab
        if (nlp := data.get("nlp")):
            if not isinstance(nlp, Language):
                dct = nlp
                nlp = getattr(import_module(dct["module"]), dct["name"])()
                nlp = nlp.from_bytes(dct["data"])
            data["nlp"] = nlp
        total = None
        if (docs := data.pop("docs", ())):
            docbin = DocBin().from_bytes(docs)
            total = len(docbin)
            docs = docbin.get_docs(vocab)
        token_dist = Counter(data.pop("token_dist"))
        corpus = cls(**data)
        corpus.meta = meta
        corpus.token_dist = token_dist
        corpus.add_docs(docs, **{ "total": total, **kwds })
        return corpus

    def to_disk(
        self,
        path: str | bytes | os.PathLike,
        **kwds: Any
    ) -> None:
        """Save to disk.

        ``**kwds`` are passed to :meth:`to_data`.
        """
        with open(path, "wb") as fh:
            pickle.dump(self.to_data(**kwds), fh)

    @classmethod
    def from_disk(
        cls,
        path: str | bytes | os.PathLike,
        *,
        vocab: Vocab | bytes | None = None,
        nlp: Language | bytes | None = None,
        **kwds: Any
    ) -> Self:
        """Construct from disk.

        Parameters
        ----------
        nlp, vocab
            Use ``vocab`` and ``nlp`` to pass an arbitrary vocabulary
            and/or language model for initializing corpus. Useful when a corpus
            has been saved to disk with ``vocab=False`` and/or ``nlp=False``.
        **kwds
            Passed to :meth:`from_data`.
        """
        with open(path, "rb") as fh:
            data = pickle.load(fh)
            if vocab:
                data["vocab"] = vocab
            if nlp:
                data["nlp"] = nlp
            return cls.from_data(data, **kwds)

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
