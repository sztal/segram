"""Package metadata."""
from importlib.metadata import version

__title__   = __name__.split(".", maxsplit=1)[0]
__version__ = version(__title__)
