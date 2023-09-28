"""*PyTest* configuration and general purpose fixtures."""
# pylint: disable=redefined-outer-name
import pytest
import spacy as spacy_module


@pytest.fixture(scope="session")
def config(request):
    return request.config

@pytest.fixture(scope="session")
def spacy(config):
    if config.getoption("--cpu"):
        spacy_module.require_cpu()
    else:
        spacy_module.prefer_gpu()
    return spacy_module

# Custom options --------------------------------------------------------------

def pytest_addoption(parser):
    """Custom `pytest` command-line options."""
    parser.addoption(
        "--interactive", action="store_true", default=False,
        help="Run tests one-by-one interactively."
    )
    parser.addoption(
        "--match", action="store", default=None,
        help="Run tests only for matching sentences."
    )
    parser.addoption(
        "--keys", action="store", default=None, type=str,
        help="Run only tests with selected document keys. "
            "Keys may be provided as a single integer, "
            "a comma separated list of integers, "
            "or as a subset of keys specified using the slice notation."
    )
    parser.addoption(
        "--skip", action="store_true", default=False,
        help="Skip selected tests instead of running them."
    )
    parser.addoption(
        "--skip-all", action="store_true", default=False,
        help="Skip all tests."
    )
    parser.addoption(
        "--accept-all", action="store_true", default=False,
        help="Accept and update expected data for all tests."
    )
    parser.addoption(
        "--cpu", action="store_true", default=False,
        help="Enforce using CPU even when GPU is available."
    )
