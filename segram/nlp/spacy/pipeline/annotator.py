from __future__ import annotations
from typing import Any, Iterable, Mapping
from spacy.tokens import Token
from spacy.language import Language
from spacy.pipeline import AttributeRuler
from ....utils.resources import JSONResource


class Annotator(AttributeRuler):
    """Annotator class adding token flags used in semantic grammar analysis.

    It is implemented as :py:class:`spacy.pipeline.AttributeRuler` subclass
    and uses standard token POS and DEP attributes to define a set of custom
    semantic grammar flags.
    """
    @classmethod
    def from_patterns(
        cls,
        nlp: Language,
        name: str,
        *,
        patterns: Mapping[str, str],
        **kwds: Any
    ) -> Annotator:
        """Initialize from a patterns packahe resource."""
        obj = cls(nlp.vocab, name, **kwds)
        patterns = JSONResource.from_package(**patterns).get()
        obj.init_extension_attrs(patterns)
        obj.add_patterns(patterns)
        return obj

    def init_extension_attrs(self, patterns: Iterable[dict]) -> None:
        for pattern in patterns:
            attrs = pattern.get("attrs", {})
            for attr, val in attrs.get("_", {}).items():
                if not Token.has_extension(attr):
                    Token.set_extension(attr, default=not val)
