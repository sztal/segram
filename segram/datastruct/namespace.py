from typing import Any
from collections.abc import MutableMapping
from collections.abc import Iterable
from ..utils.meta import get_cname


class Namespace(MutableMapping):
    """Namespace class.

    It behaves like a dictionary with both item and attribute
    getters, setters and deletters.
    """
    def __init__(self, *args: Any, **kwds: Any) -> None:
        try:
            dct = dict(*args, **kwds)
        except TypeError as exc:
            raise TypeError(f"'{get_cname(self)}' {str(exc)}") from exc
        self.__dict__.update(dct)

    def __repr__(self) -> str:
        return f"{get_cname(self)}({self.__dict__})"

    def __iter__(self) -> Iterable[str]:
        yield from self.__dict__

    def __len__(self) -> int:
        return len(self.__dict__)

    def __getitem__(self, name: str) -> Any :
        return self.__dict__[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value

    def __delitem__(self, name: str) -> None:
        del self.__dict__[name]

    def __contains__(self, name: str) -> bool:
        return name in self.__dict__

    @property
    def names(self) -> list[str]:
        return list(self)
