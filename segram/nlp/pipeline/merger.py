from __future__ import annotations
from spacy.tokens import Doc
from spacy.util import filter_spans
from .annotator import Annotator


class Merger(Annotator):
    """Merger class for merging standard multitokens found in a given language
    (such as phrasal verbs in English) as well as multitoken entities.
    """
    def __call__(self, doc: Doc) -> Doc:
        """Retokenize document and merge multitokens."""
        # matches = self.matcher(doc)
        spans = []
        for _, start, end in self.matcher(doc):
            span = doc[start:end]
            spans.append(span)
        with doc.retokenize() as retokenizer:
            for span in filter_spans(spans):
                root = span.root
                attrs = {
                    "POS": root.pos_,
                    "DEP": root.dep_,
                    "MORPH": str(root.morph),
                    "ENT_ID": root.ent_id_,
                    "ENT_IOB": root.ent_iob_,
                    "ENT_TYPE": root.ent_type_
                }
                retokenizer.merge(span, attrs=attrs)
        return doc
