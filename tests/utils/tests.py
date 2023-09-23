"""Test sets for grammar classes."""
from __future__ import annotations
from typing import Any, Optional, Iterable, Callable
from pathlib import Path
from copy import deepcopy
import json
from murmurhash import hash_unicode
from segram import settings
from segram.grammar import Sent
from segram.nlp.tokens import Doc, Span
from segram.utils.types import Namespace
from segram.utils.meta import get_cname
from segram.utils.resources import JSONResource
from .diffs import GrammarDiff


class TestSet:
    """Base class for test cases for grammar classes.

    Attributes
    ----------
    data
        Raw resource data.
    cmap
        Dictionary mapping text hashes to case data.
        Initialized from a list of case dictionaries.
    """
    def __init__(
        self,
        nlp: Callable,
        resource: JSONResource,
        *,
        callback: Optional[Callable] = None,
        **kwds: Any
    ) -> None:
        """Initialization method.

        Parameters
        ----------
        nlp
            Callable converting raw texts into
            :class:`~segram.nlp.Doc` instances.
        resource
            Resource handler.
        callback
            Callback function to be called on documents
            produced by ``self.nlp()``. It can be used to
            postprocess documents to ensure that they are
            really :class:`~segram.nlp.Doc` instances.
        **kwds
            Additional metadata saved as a namesapce object
            under the ``self.ns`` attribute.
        """
        self.nlp = nlp
        self.resource = resource
        self.callback = callback
        self.data = resource.get()
        if "cases" not in self.data:
            self.data["cases"] = []
        self.cmap = self.make_cmap(deepcopy(self.data["cases"]))
        self.ns = Namespace(**kwds)

    def __repr__(self) -> str:
        mid = f" using '{str(self.path)}'"
        return f"<{get_cname(self)}{mid} at {hex(id(self))}>"

    def __len__(self) -> int:
        return len(self.cmap)

    def __getitem__(self, key: int | str) -> DocTestCase:
        key = self.hash_key(key)
        data = self.cmap[key]
        if "expected" not in data:
            data["expected"] = []
        doc = self.make_doc(data["text"])
        doc = DocTestCase(self, key, doc, bad=data.get("bad"))
        return doc

    def __setitem__(self, key: int | str, case: dict[str, Any]) -> None:
        self.cmap[self.hash_key(key)] = case

    def __delitem__(self, key: int | str) -> None:
        del self.cmap[self.hash_key(key)]

    def __iter__(self) -> Iterable[Doc]:
        for key in self.cmap:
            yield self[key]

    def __add__(self, text: str) -> None:
        return self.add(text)

    # Properties --------------------------------------------------------------

    @property
    def path(self) -> str:
        return Path(self.resource.path)

    @property
    def cases(self) -> list[DocTestCase]:
        return list(self.cmap.values())

    @property
    def keys(self) -> list[int]:
        return list(self.cmap.keys())

    @property
    def meta(self) -> dict[str, Any]:
        return self.data["meta"]

    # Methods -----------------------------------------------------------------

    @classmethod
    def from_package(
        cls,
        nlp: Callable,
        package: str,
        filename: str,
        **kwds: Any
    ) -> TestSet:
        """Construct from package and resource names."""
        resource = JSONResource.from_package(package, filename)
        return cls(nlp, resource, **kwds)

    def make_doc(self, text: str) -> Doc:
        """Make document object."""
        doc = self.nlp(text)
        if self.callback:
            doc = self.callback(doc)
        return doc

    def add(
        self,
        text: str,
        index: Optional[int] = None,
        *,
        expected: Iterable[dict[str, Any]] = ()
    ) -> Doc:
        """Add new test case.

        Parameters
        ----------
        text
            Raw text.
        expected
            Iterable with expected results for sentences.

        Returns
        -------
        case
            Document test case object.
        """
        doc = self.make_doc(text)
        key = self.hash_key(text)
        case = DocTestCase(self, key, doc)
        data = { "text": text, "expected": list(expected) }
        if index is not None:
            cmap = list(self.cmap.items())
            cmap.insert(index, (key, data))
            self.cmap = dict(cmap)
        else:
            self[key] = data
        case.print()
        return case

    def save(self, **kwds: Any) -> None:
        """Save test data, but first ask.

        This allows saving data modified during
        test inspection, for instance when ``results``
        are correct and ``expected`` values are not.
        """
        # pylint: disable=too-many-locals
        msg = settings.printer.get()
        meta = self.data["meta"].copy()
        orig = self.make_cmap(self.data["cases"])

        overlap = set(self.cmap) & set(orig)
        added = set(self.cmap) - set(orig)
        removed = set(orig) - set(self.cmap)
        diff = [
            (k, v["text"]) for k in overlap
            if (v := self.cmap[k]) != orig[k]
        ]
        added = [ (k, self.cmap[k]["text"]) for k in added ]
        removed = [ (k, orig[k]["text"]) for k in removed ]

        if diff:
            print(msg.divider(f"changed documents ({len(diff)})"))
            for key, sent in diff:
                print(f"{key}: {sent}")
        if added:
            print(msg.divider(f"added documents ({len(added)})"))
            for key, sent in added:
                print(f"{key}: {sent}")
        if removed:
            print(msg.divider(f"removed documents ({len(removed)})"))
            for key, sent in removed:
                print(f"{key}: {sent}")
        if diff or added or removed:
            while True:
                answer = input("\nDo you want to save the changes? [y/N]\n").lower()
                if not answer or answer in ("n", "no"):
                    break
                if answer in ("y", "yes"):
                    cases = list(self.cmap.values())
                    for c in cases:
                        if (key := "expected") in c:
                            c[key] = list(sorted(
                                c[key],
                                key=lambda d: d["start"]
                            ))
                    data = dict(meta=meta, cases=cases)
                    kwds = { "cls": self.JSONEncoder, "indent": 4, **kwds }
                    self.resource.write(data, json_kws=kwds)
                    break
                print("only 'y/yes' and 'n/no' answers are accepted")

    @staticmethod
    def hash_key(key: str | int) -> int:
        """Make hash key."""
        return hash_unicode(key) if isinstance(key, str) else key

    def make_cmap(
        self,
        cases: Iterable[dict[str, Any]]
    ) -> dict[int, dict[str, Any]]:
        """Make cases dictionary from a cases list."""
        return {
            self.hash_key(case["text"]): case
            for case in cases
        }

    class JSONEncoder(json.JSONEncoder):
        """Custom JSON encoder not indenting cases data."""
        def encode(self, o):
            o = o.copy()
            cases = o["cases"]
            o["cases"] = ["@" for _ in range(len(cases))]
            o = super().encode(o)
            orig = self.indent
            self.indent = None
            for case in cases:
                string = super().encode(case)
                o = o.replace("\"@\"", string, 1)
            self.indent = orig
            return o


class DocTestCase:
    """Document test case.

    Attributes
    ----------
    tests
        Parent test set.
    doc
        Tested document object.
    key
        Key for the case data.
    bad
        Set of sentences marked as 'bad',
        for which tests should always fail.
    """
    def __init__(
        self,
        tests: TestSet,
        key: int,
        doc: Doc,
        *,
        bad: Optional[Iterable[int]] = None
    ) -> None:
        self.tests = tests
        self.key = key
        self.doc = doc
        self.bad = set(bad or ())

    def __repr__(self) -> str:
        return str(self.doc)

    def __len__(self) -> int:
        return len(self.expected)

    def __iter__(self) -> Iterable[SentTestCase]:
        yield from self.cases

    # Properties --------------------------------------------------------------

    @property
    def idx(self) -> int:
        return list(self.tests.cmap.keys()).index(self.key)

    @property
    def data(self) -> dict[str, Any]:
        return self.tests.cmap[self.key]

    @property
    def expected(self) -> list[dict[str, Any]]:
        return self.data["expected"]

    @property
    def text(self) -> str:
        return self.data["text"]

    @property
    def sents(self) -> Iterable[Span]:
        yield from self.doc.sents

    @property
    def cases(self) -> Iterable[SentTestCase]:
        for i, sent in enumerate(self.sents):
            yield SentTestCase(self, i, sent)

    @property
    def p(self) -> None:
        self.print()

    # Methods -----------------------------------------------------------------

    def print(self) -> None:
        for case in self.cases:
            case.rprint()

    def add(
        self,
        text: str,
        offset: int = 0,
        **kwds: Any
    ) -> DocTestCase:
        """Insert new document case after ``self``.

        Parameters
        ----------
        text
            Text of the new document.
        offset
            Offset to add/subtract from the index
            of the current document.
        **kwds
            Passed to :meth:`TestSet.add`
            other than ``index`` argument.
        """
        idx = self.idx+1+offset
        return self.tests.add(text, idx, **kwds)

class SentTestCase:
    """Sentence test case.

    Attributes
    ----------
    parent
        Parent document test case.
    i
        Index of sentence within document sequence.
    sent
        NLP sentence object.
    results
        Results.
    """
    # pylint: disable=too-many-public-methods
    def __init__(self, parent: DocTestCase, i: int, sent: Span) -> None:
        self.parent = parent
        self.i = i
        self.sent = sent
        self.results = self.sent.get_grammar(use_data=None)

    def __repr__(self) -> str:
        msg = settings.printer.get()
        fg = "good" if self.is_correct() else "fail"
        icon = msg.icons.get(fg)
        return msg.color(f"{icon} {self.sent}", fg=fg, bold=True)

    # Properties --------------------------------------------------------------

    @property
    def text(self) -> str:
        return self.sent.text

    @property
    def key(self) -> int:
        return self.parent.key

    @property
    def idx(self) -> int:
        return self.parent.idx

    @property
    def edata(self) -> dict[str, Any]:
        return self.parent.expected[self.i]

    @property
    def expected(self) -> Sent:
        return self.results.types.Sent.from_data(
            doc=self.sent.doc,
            data=self.edata
        )

    @property
    def doc(self) -> Doc:
        return self.sent.doc

    @property
    def tests(self) -> TestSet:
        return self.parent.tests

    @property
    def diff(self) -> GrammarDiff:
        return GrammarDiff(self.results, self.expected)

    @property
    def rp(self) -> None:
        self.rprint()

    @property
    def ep(self) -> None:
        self.eprint()

    @property
    def p(self) -> None:
        self.print()

    @property
    def is_bad(self) -> bool:
        return self.i in self.parent.bad

    # Methods -----------------------------------------------------------------

    def equal(self, s1, s2, *, strict: bool = True) -> bool:
        """Compare two sentences.

        Parameters
        ----------
        strict
            When ``strict=False`` then normal equality
            test is used. When ``strict=True`` then more
            stringent equality conditions are checked using
            :meth:`~segram.grammar.Grammar.equal`.
        """
        return s1.equal(s2, strict=strict)

    def is_correct(self, **kwds: Any) -> bool:
        """Check if case runs correctly."""
        if self.is_bad:
            return False
        try:
            return self.equal(self.results, self.expected, **kwds)
        except IndexError:
            return False

    def is_dag(self) -> bool:
        """Check if sentence is a proper DAG."""
        return self.results.graph.is_dag

    def serialize(self) -> Sent:
        """Get diff for serialized doc."""
        serialized = self.results.from_data(self.doc, self.results.to_data())
        return GrammarDiff(self.results, serialized, strict=False)

    def simple_conversion(self) -> Sent:
        """Conversion to generic :mod:`segram` NLP document
        and generic language class without NLP backend is correct.
        """
        doc = self.doc.__class__.from_data(self.doc.data)
        grammar = self.results.grammars.get(doc.lang)
        other = grammar.types.Sent.from_data(doc, self.results.to_data())
        return GrammarDiff(self.results, other, strict=False)

    def rprint(self) -> None:
        self.results.print()

    def eprint(self) -> None:
        self.expected.print()

    def print(self) -> None:
        print(f"\n\n{self}", end="\n")
        self.rprint()

    def save(self) -> None:
        """Save case results data."""
        data = self.results.to_data()
        expected = self.tests.cmap[self.key]["expected"]
        if self.i < len(expected):
            expected[self.i] = data
        else:
            expected.append(data)

    def ok(self) -> None:
        """Save and mark as correct."""
        self.save()
        if self.i in self.parent.bad:
            bad = self.tests.cmap[self.key].get("bad")
            if bad:
                bad.remove(self.i)

    def bad(self) -> None:
        """Mark case as incorrect."""
        if self.i not in self.parent.bad:
            self.parent.bad.add(self.i)
            self.tests.cmap[self.key].setdefault("bad", []).append(self.i)

    def add(
        self,
        text: str,
        offset: int = 0,
        **kwds: Any
    ) -> DocTestCase:
        """Insert new document case after ``self.parent``.

        Parameters
        ----------
        text
            Text of the new document.
        offset
            Offset to add/subtract from the index
            of the current document.
        **kwds
            Passed to :meth:`TestSet.add`
            other than ``index`` argument.
        """
        return self.parent.add(text, offset, **kwds)
