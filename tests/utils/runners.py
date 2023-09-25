"""Base test runner class."""
from typing import Any, Callable
from functools import partial
import re
import ipdb
import pytest
from IPython import embed
from spacy.tokens import Doc as SpacyDoc
from spacy.language import Language
from segram import settings
from segram.nlp.tokens import Doc
from segram.utils.resources import JSONResource
from segram.utils.versioning import is_correct_version
from segram.utils.meta import get_cname
from .tests import TestSet, SentTestCase


class PyTestRunner:
    """Base class for :mod:`pytest` test runner classes.

    It detects data submodule based on ``lang`` class initilization
    argument and loads a resource file based on ``resource`` argument.

    Attributes
    ----------
    config
        PyTest config object.

    Notes
    -----
    Fixtures and tests defined by test runners depend on module
    level fixture ``nlp`` defined in an appropriate ``conftest`` file.
    """
    callback = None
    config = None
    tests = None

    def __init_subclass__(
        cls,
        lang: str | None = None,
        resource: str | None = None
    ) -> None:
        if not lang:
            return
        cls.lang = lang
        cls.resource = cls.get_resource(lang, resource)

    # Fixtures ----------------------------------------------------------------

    @pytest.fixture(scope="class")
    def testset(self, nlp, config):
        kwds = dict(
            interactive=config.getoption("--interactive"),
            keys=config.getoption("--keys"),
            skip_all=config.getoption("--skip-all"),
            skip=config.getoption("--skip"),
            accept_all=config.getoption("--accept-all")
        )
        match = config.getoption("--match")
        if match:
            kwds.update(match=re.compile(match, re.IGNORECASE))
        tests = self.get_tests(nlp, **kwds)
        self.__class__.config = config
        self.__class__.tests = tests
        self.validate_metadata(nlp, tests)
        yield tests
        if tests.ns["interactive"]:
            if self.user_wants_shell():
                self.embed()
        tests.save()

    @pytest.fixture(scope="class")
    def keys(self, config):
        return self._parse_keys_arg(config.getoption("--keys"))

    @pytest.fixture(scope="class")
    def key(self, keys, request):
        if self.tests.ns.skip_all:
            pytest.skip("tests have been disabled")
        key = request.param
        key_in = keys is not None and key in keys
        skip = self.tests.ns.skip
        if skip and (keys is None or key_in):
            pytest.skip("skipping selected document")
        elif not skip and not key_in and keys is not None:
            pytest.skip("not in the selected set of document keys")
        return key

    # Properties --------------------------------------------------------------

    @property
    def shell(self) -> Callable:
        return partial(embed, colors="neutral")

    @property
    def off(self) -> None:
        self.disable()

    @property
    def skip(self) -> None:
        self.disable()

    # Methods -----------------------------------------------------------------

    def disable(self) -> None:
        """Disable tests."""
        self.tests.ns.disabled = True

    @classmethod
    def get_resource(cls, lang: str, resource: str) -> JSONResource:
        """Get resource handler."""
        return JSONResource.from_package(f"tests.{lang}.data", resource)

    @classmethod
    def casekeys(cls, lang: str, resource: str) -> list[int]:
        """Get case keys."""
        resource = cls.get_resource(lang, resource)
        return [
            TestSet.hash_key(c["text"])
            for c in resource.get()["cases"]
        ]

    def get_tests(self, nlp: Callable, **kwds: Any) -> TestSet:
        return TestSet(nlp, self.resource, callback=self.callback, **kwds)

    def check(self, case: SentTestCase) -> None:
        """Check ``case``."""
        if self.tests.ns.accept_all:
            case.ok()
        else:
            assert case.equal(case.results, case.expected, strict=True)
            assert case.results.graph.is_dag
            diff = case.serialize()
            assert not diff()
            diff = case.simple_conversion()
            assert not diff()
            assert pytest.approx(case.results.coverage) == 1

    def check_doc(self, tests, key: str) -> None:
        if tests.ns.get("disabled"):
            pytest.exit("tests have been disabled")
        text = tests.cmap[key]["text"]
        if (rx := tests.ns.get("match")) and not rx.search(text):
            pytest.skip("document does not match the pattern")
        # Run checks for sentences
        for case in tests[key]:
            self.check_sent(case)

    def check_sent(self, case: SentTestCase) -> None:
        ns = case.tests.ns
        if (rx := ns.get("match")) and not rx.search(case.text):
            pytest.skip("sentence does not match the pattern")
        try:
            self.check(case)
        except AssertionError as exc:
            if ns.interactive:
                pass
            else:
                raise exc
        if ns.interactive:
            case.print()
            print(f"\n{case}")
            print(case.diff)
            ipdb.set_trace()

    def user_wants_shell(self) -> bool:
        """Ask whether interactive shell allowing creating
        new cases on the fly should be spawned.

        Returns
        -------
        bool
            Yes/no answer of the user.
        """
        while True:
            answer = input("\nDo you want run interacive session? [y/N]\n").lower()
            if not answer or answer in ("n", "no"):
                return False
            if answer in ("y", "yes"):
                return True

    def validate_metadata(self, nlp: Callable, tests: TestSet) -> None:
        """Check if model metadata matches test requirements."""
        raise NotImplementedError

    # Internals ---------------------------------------------------------------

    def _parse_keys_arg(self, keys: str) -> set[str]:
        if not keys:
            return None
        if "," in keys and ":" in keys:
            raise ValueError(
                "'keys' argument cannot be a slice "
                "and a list at the same time"
            )
        if "," in keys:
            return list(map(int, keys.split(",")))
        if ":" in keys:
            dockeys = self.tests.keys
            start, stop, *_ = keys.split(":")
            start = int(start) if start else None
            stop = int(stop) if stop else None
            nkeys = 0
            nfound = 0
            for key in (start, stop):
                if key is not None:
                    nkeys += 1
                    if key in dockeys:
                        nfound += 1
            if nkeys == 2 and nfound == 1:
                raise ValueError(
                    "inconsistent range of keys, "
                    "as only one of the two corresponds "
                    f"to a test case in '{get_cname(self)}'"
                )
            if nfound == 0:
                return set()
            start = dockeys.index(start) if start is not None else None
            stop = dockeys.index(stop) if stop is not None else None
            if start is not None and stop is not None and start > stop:
                start, stop = stop, start
            index = slice(start, stop)
            return dockeys[index]
        return set([int(keys)])


class SpacyTestRunner(PyTestRunner):
    """Test runner for :mod:`spacy`."""

    @staticmethod
    def callback(doc: SpacyDoc) -> Doc:
        return getattr(doc._, settings.spacy_alias)

    def validate_metadata(self, nlp: Language, tests: TestSet) -> None:
        """Check if model metadata matche test requirements.

        Raises
        ------
        TypeError
            If metadata do not agree.
        """
        meta = dict(nlp.pipeline)[settings.spacy_alias].meta
        tests = self.tests.meta

        for package in ("spacy", "segram"):
            field = f"{package}_version"
            if not is_correct_version(
                v=(version := meta[field]),
                constraints=(allowed := tests[field])
            ):
                raise TypeError(
                    f"'{package}' version is '{version}' "
                    f"is not consistent with the constraint '{allowed}'"
                )

        if (mlang := meta["model"]["lang"]) != (tlang := tests["lang"]):
            raise TypeError(f"model language is '{mlang}' but tests are for '{tlang}'")

        model = meta["model"]
        name = f"{mlang}_{model['name']}"
        version = model["version"]
        allowed = { m["name"]: m["version"] for m in tests["models"] }
        if model["name"] not in allowed:
            raise TypeError(f"test set is not for '{name}' model")
        if not is_correct_version(version, (const := allowed[model["name"]])):
            raise TypeError(
                f"'{name}' model version '{version}' "
                f"is not consistent with the constraint '{const}'"
            )
