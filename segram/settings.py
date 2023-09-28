"""Global package settings.

Attributes
----------
printing
    Configurable printing objects
    implemented as :class:`~segram.colors.Printer` instances.
"""
from .utils.settings import Settings
from .utils.colors import printer_settings


class SegramSettings(Settings):
    printer: Settings

settings = SegramSettings(
    printer=printer_settings,
)
