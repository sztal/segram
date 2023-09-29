"""Segram top module."""
from .about import __version__, __title__
from .settings import settings
from .utils.registries import grammars
from .grammar import Doc
from .semantic import Story
from .nlp import Corpus
from .datastruct import DataIterator, DataList, DataTuple, DataDict
from .utils.misc import ensure_cpu_vectors, prefer_gpu_vectors
