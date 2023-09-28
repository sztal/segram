"""Base Segram module.

It implements the _Segram_ pipe component providing
all main semantic grammar transformations and related auxiliary methods.
"""
from typing import Any, Sequence, ClassVar
from importlib import import_module
import numpy as np
import spacy
from spacy.tokens import Doc
from spacy.language import Language
from spacy.pipeline.pipe import Pipe
from ..extensions import SpacyExtensions
from ... import __title__, __version__
from ...utils.meta import get_cname
from ...utils.registries import models as models_registry


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
    __initialized__: ClassVar[bool] = False

    def __init__(
        self,
        nlp: Language,
        name: str,
        *,
        grammar: str,
        preprocess: Sequence[str],
        alias: str = __title__,
        vectors: str | Language | None = None
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
        self.alias = alias
        self.nlp = nlp
        self.name = name
        self.extensions = self.import_module(grammar, nlp.lang)
        self.grammar = f"{grammar}.{nlp.lang}"
        if isinstance(vectors, str):
            vectors = spacy.load(vectors, enable="tok2vec", vocab=nlp.vocab)
        elif vectors:
            vcn = vectors.__class__.__name__
            raise ValueError(f"'vectors' must be provided as a language model or a name, not '{vcn}'")
        models_registry.register(self.get_model_name(nlp), func=nlp)
        gpu = not isinstance(self.nlp.vocab.vectors.data, np.ndarray)
        numpy = np
        if gpu:
            numpy = import_module("cupy")
        self.numpy = numpy
        self.meta = {
            "name":               self.name,
            __title__+"_alias":   alias,
            __title__+"_version": __version__,
            __title__+"_grammar": f"{grammar}.{nlp.lang}",
            "spacy_version":      spacy.__version__,
            "spacy_gpu":          gpu,
            "model":              self.get_model_info(nlp),
            "vectors":            self.get_model_info(vectors) if vectors else None
        }
        self.configure_pipeline(*preprocess)
        if not self.__initialized__:
            self.init_extensions()

    def __call__(self, doc: Doc) -> Doc:
        meta = self.meta.copy()
        setattr(doc._, __title__+"_alias", self.alias)
        setattr(doc._, f"{self.alias}_meta", meta)
        setattr(doc._, f"{self.alias}_numpy", self.numpy)
        return doc

    # Properties --------------------------------------------------------------

    @property
    def id(self) -> int:
        """Hash id of the component."""
        hashdata = []
        for k, v in self.meta.items():
            if isinstance(v, dict):
                v = tuple(v.items())
            hashdata.append((k, v))
        return hash(tuple(hashdata))

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
        kwds = { "alias": self.alias }
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
        self.__class__.__initialized__ = True

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
            f"{self.alias}_{c}" for c in components
            if not c.startswith(self.alias+"_")
        )
        pipes = [ self.normalize_pipe_name(pipe) for pipe in components ]
        for pipe in pipes:
            if pipe not in self.nlp.pipe_names:
                self.nlp.add_pipe(pipe, **kwds)

    def normalize_pipe_name(self, pipe: str) -> str:
        """Normalize pipeline component name."""
        if "." in pipe:
            _, pipe = pipe.split(".")
        prefix = self.alias+"_"
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

    @staticmethod
    def get_model_info(nlp: Language) -> str:
        """Get language model information."""
        return {
            "lang":        nlp.meta["lang"],
            "name":        nlp.meta["name"],
            "version":     nlp.meta["version"],
            "description": nlp.meta["description"]
        }
