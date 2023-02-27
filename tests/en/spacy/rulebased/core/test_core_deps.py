import pytest
from .....utils.runners import SpacyTestRunner


class TestSpacyRulebasedEnglishGrammar(
    SpacyTestRunner,
    lang=(lang := "en"),
    resource=(resource := "en-core-deps.json")
):
    @pytest.mark.parametrize(
        "key", SpacyTestRunner.casekeys(lang, resource),
        indirect=True
    )
    def test_grammar(self, testset, key):
        self.check_doc(testset, key)
