"""Segram coreference pipeline component."""
from typing import Any, Sequence, Self
import os
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Alignment
from .base import Segram
from ... import __title__
from ...utils.meta import get_cname


class Coref:
    """Coreference resolution pipeline component
    based on :mod:`spacy` coref component.

    Attributes
    ----------
    name
        Pipe name.
    model
        Language model for coreference resolution.
    """
    def __init__(
        self,
        nlp: Language,
        name: str,
        model: Language,
        components: Sequence[str] | None = None
    ) -> None:
        """Initilization method.

        Parameters
        ----------
        nlp
            Main language model.
        model
            Name of a coreference language model.
        components
            Names of pipeline component names to include.
            Use all if ``None``.

        Raises
        ------
        ValueError
            If ``components`` are empty but not ``None``.
        """
        try:
            segram = dict(nlp.pipeline)[__title__]
            self.alias = segram.alias
        except KeyError as exc:
            raise RuntimeError(
                f"'{name}' component can be "
                f"initialized only after '{__title__}'"
            ) from exc
        self.nlp = nlp
        self.name = name
        self.model = model
        if components is not None and not components:
            raise ValueError(
                f"'{get_cname(self)}' pipe initizialized "
                "with empty 'components' list"
            )
        if components:
            self.model.select_pipes(enable=components)

    def __call__(self, doc: Doc) -> Doc:
        cdoc = self.model(doc.text)
        s1 = list(map(str, doc))
        s2 = list(map(str, cdoc))
        align = Alignment.from_strings(s1, s2)
        for spans in cdoc.spans.values():
            cluster = [ int(align.y2x.data[t.i]) for s in spans for t in s ]
            self.set_corefs(doc, cluster)
        getattr(doc._, f"{self.alias}_meta")["coref"] = \
            Segram.get_model_info(self.model)
        return doc

    def set_corefs(self, doc: Doc, cluster: Sequence[int]) -> None:
        """Set proper coreferences from pronoun tokens to closest
        non-pronoun neighbors within the ``cluster``.

        Notes
        -----
        Coreferences are stored as token indexes (integers)
        in ``_ref`` custom attribute on tokens.
        """
        # pylint: disable=protected-access
        alias = self.alias
        proper = []
        pronouns = []
        for i in cluster:
            if getattr(doc[i]._, alias+"_sns").is_pron:
                pronouns.append(i)
            else:
                proper.append(i)
        for i in pronouns:
            closest = None
            for j in proper:
                if closest is None or abs(i-j) < closest:
                    closest = j
            if closest is not None:
                corefs = set()
                corefs.add(closest)
                for conj in doc[closest].conjuncts:
                    corefs.add(conj.i)
                setattr(doc[i]._, f"{alias}_corefs", tuple(sorted(corefs)))

    @classmethod
    def from_model(
        cls,
        nlp: Language,
        name: str,
        model: str,
        components: Sequence[str] | None = None,
        **kwds: Any
    ) -> Self:
        """Initialize from model name.

        ``**kwds`` are passed to :func:`spacy.load`.
        """
        model = spacy.load(model, **kwds)
        return cls(nlp, name, model, components)

    def to_disk(self, path: str | bytes | os.PathLike, **kwds: Any) -> None:
        """Serialize the coreference model to disk."""
        self.model.to_disk(path, **kwds)

    def from_disk(self, path: str | bytes | os.PathLike, **kwds: Any) -> Self:
        """Load from disk."""
        self.model = spacy.load(path, **kwds)
        return self
