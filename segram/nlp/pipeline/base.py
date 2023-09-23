"""Base Segram module.

It implements the _Segram_ pipe component providing
all main semantic grammar transformations and related auxiliary methods.
"""
from __future__ import annotations
from typing import Any, Optional, Sequence, ClassVar, Mapping
from types import MappingProxyType
from importlib import import_module
from time import time
import spacy
from spacy.tokens import Doc
from spacy.language import Language
from spacy.pipeline.pipe import Pipe
from ..extensions import SpacyExtensions
from ... import __title__, __version__, settings
from ...utils.meta import get_cname
from ...utils.registries import models as models_registry
from ...utils.registries import vectors as vectors_registry


class Segram(Pipe):
    """Semantic grammar pipeline component.

    It extends :mod:`spacy` token classes with semantic
    grammar methods and related functionalities such as
    custom preprocessing (e.g. merging and corrected lemmatization).

    Attributes
    ----------
    nlp
        Language model object.
    name
        Name of the component.
    extensions
        Module defining custom :mod:`spacy` extensions.
    grammar
        Label of grammar implementation.
    meta
        Metadata dictionary with details on :mod:`spacy`
        and `segram` models being used.
    """
    __initialized__: ClassVar[Mapping[int, bool]] = MappingProxyType({})

    def __init__(
        self,
        nlp: Language,
        name: str,
        *,
        grammar: str,
        preprocess: Sequence[str],
        alias: str = __title__,
        store_data: bool = True,
        vectors: Optional[str | Language] = None
    ) -> None:
        """Initialization method.

        Parameters
        ----------
        preprocess
            List of :mod:`segram` pipeline components to use for preprocessing
            documents before applying the main :mod:`segram` pipe.
            If ``None`` then all available preprocessing components are used.
        alias
            Set ``spacy_alias`` in the global settings.
            It is used for namespacing extension attributes added
            by :mod:`segram` in order to avoid collision with other
            packages.
        vectors
            Vector table to use instead of the vectors provided by the main
            model. Must be provided by the name of a model or the model
            object itself, so the it is possible to keep track of the model
            name.
        """
        if not alias:
            raise ValueError(
                f"'{get_cname(self)}' must define non-empty 'alias' "
                "for naming and prefixing Spacy extension attributes"
            )
        if (sa := settings.spacy_alias) is not None and sa != alias:
            raise ValueError(
                f"'{get_cname(self)}' tries to overwrite "
                f"existing `segram` alias '{sa}' with '{alias}'."
                "All segram pipeline components running in the same "
                "process must use the same alias."
            )
        settings.spacy_alias = alias
        self.nlp = nlp
        self.name = name
        self.store_data = store_data
        self.extensions = self.import_module(grammar, nlp.lang)
        self.grammar = f"{grammar}.{nlp.lang}"
        if isinstance(vectors, str):
            vectors = spacy.load(vectors)
        vectors_name = None
        if isinstance(vectors, Language):
            vectors_name = self.get_model_name(vectors)
            vectors = vectors.vocab.vectors
            vectors_registry.register(vectors_name, func=vectors)
        elif vectors:
            vcn = vectors.__class__.__name__
            raise ValueError(f"'vectors' cannot be assigned with instance of '{vcn}'")
        models_registry.register(self.get_model_name(nlp), func=nlp)
        self.meta = {
            "name":                  self.name,
            __title__+"_version":    __version__,
            __title__+"_grammar":    self.grammar,
            "spacy_alias":           alias,
            "spacy_version":         spacy.__version__,
            "lang":                  self.nlp.meta["lang"],
            "model":                 self.nlp.meta["name"],
            "model_version":         self.nlp.meta["version"],
            "model_description":     self.nlp.meta["description"],
            "model_vectors":         vectors_name
        }
        self.configure_pipeline(*preprocess)
        if not self.__initialized__.get(self.id, False):
            self.init_extensions()

    def __call__(self, doc: Doc) -> Doc:
        alias = settings.spacy_alias
        meta = self.meta.copy()
        setattr(doc._, f"{alias}_meta", meta)
        setattr(doc._, f"{alias}_cache", {})
        setattr(doc._, f"{alias}_model", self.get_model_name(self.nlp))
        setattr(doc._, f"{alias}_vectors", meta["model_vectors"])
        if self.store_data:
            start = time()
            data = {
                (sent.start, sent.end): \
                    sent.get_grammar(use_data=False).to_data()
                for sent in getattr(doc._, alias).sents
            }
            setattr(doc._, f"{alias}_grammar_data", data)
            elapsed = time()-start
            meta["segram_computation_time"] = elapsed
        return doc

    # Properties --------------------------------------------------------------

    @property
    def id(self) -> int:
        """Hash id of the component."""
        return hash(tuple(self.meta.items()))

    @property
    def initialized(self) -> bool:
        return self.__initialized__[self.id]

    # Methods -----------------------------------------------------------------

    def import_module(self, grammar: str, lang: str) -> SpacyExtensions:
        """Import NLP module from grammar label and language code.

        Returns
        -------
        extensions
            :class:`~segram.nlp.extensions.SpacyExtensions` instance.
        """
        path = f"{__title__}.nlp.backend.{grammar}.lang.{lang}"
        module = import_module(path)
        kwds = {}
        for tok_type in ("Doc", "Span", "Token"):
            try:
                kwds[tok_type.lower()] = getattr(module, tok_type)
            except AttributeError as exc:
                raise AttributeError(
                    f"module does not define nor import '{tok_type}' class; "
                    "'spacy' backends must provide enhanced "
                    "'Doc', 'Span' and 'Token' classes"
                ) from exc
        return SpacyExtensions(**kwds)

    def init_extensions(self) -> None:
        """Initialize custom :mod:`spacy` attributes."""
        self.extensions.register()
        self.__class__.__initialized__ = MappingProxyType({
            **self.__class__.__initialized__,
            self.id: True
        })

    def configure_pipeline(self, *components: str, **kwds: Any) -> None:
        """Configure secondary :mod:`segram` pipeline components.

        Parameters
        ----------
        *components
            Pipeline component names.
        **kwds
            Passed to :meth:`~spacy.language.Language.add_pipe`.
        """
        components = tuple(
            f"{settings.spacy_alias}_{c}" for c in components
            if not c.startswith(settings.spacy_alias+"_")
        )
        pipes = [ self.normalize_pipe_name(pipe) for pipe in components ]
        for pipe in pipes:
            if pipe not in self.nlp.pipe_names:
                self.nlp.add_pipe(pipe, **kwds)

    @staticmethod
    def normalize_pipe_name(pipe: str) -> str:
        """Normalize pipeline component name."""
        if "." in pipe:
            _, pipe = pipe.split(".")
        prefix = settings.spacy_alias+"_"
        if not pipe.startswith(prefix):
            pipe = prefix + pipe
        return pipe

    def get_config(self) -> dict:
        """Get current config dictionary."""
        return self.nlp.config["components"][self.name].copy()

    @staticmethod
    def get_model_name(nlp: Language) -> str:
        """Get language model name."""
        return f"{nlp.meta['lang']}_{nlp.meta['name']}"
