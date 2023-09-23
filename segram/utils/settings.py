"""Utilities for managing package options."""
from __future__ import annotations
from typing import Any, Optional
from ..datastruct import Namespace


class Settings(Namespace):
    """Settings manager for handling multiple option sets.

    It support ``with`` statement, which allow setting
    temporary values to any option group. If ``default``
    attribute is defined, it is used as a defult key when
    using :meth:`get()` without arguments.
    """
    def __init__(self, **kwds: Any) -> None:
        if (default := kwds.pop("default", None)) is not None:
            self.default = default
        super().__init__(**kwds)

    def __enter__(self) -> Settings:
        for group in self.__dict__.values():
            group.maps.appendleft({})

    def __exit__(self, exc_type: type, exc_value: Any, exc_tb: Any) -> None:
        for group in self.__dict__.values():
            group.maps.popleft()

    def get(self, key: Optional[str] = None, default: Any = None, /) -> Any:
        """Get option value or a default.

        Instance attribute ``self.__default__`` is used
        instead of ``key`` when ``key=None``.
        """
        # pylint: disable=arguments-differ
        if key is None:
            key = getattr(self, "default", None)
        return super().get(key, default)
