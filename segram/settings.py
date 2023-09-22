"""Global package settings.

Attributes
----------
printing
    Configurable printing objects
    implemented as :class:`~segram.colors.Printer` instances.
"""
from typing import Optional
from .utils.settings import Settings
from .utils.colors import printer_settings


class SegramSettings(Settings):
    spacy_alias: Optional[str]
    printer: Settings

settings = SegramSettings(
    printer=printer_settings,
    spacy_alias=None
)
