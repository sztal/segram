"""Object property matcher and related utilities."""
from typing import Self, Optional, Any, Literal, Callable
from typing import Iterable, Mapping
import re


class Matcher:
    """Matcher class.

    Matcher implements a flexible framework for matching/testing
    properties of complex nested objects.

    Attribtues
    ----------
    start
        Start match specification.
        This is the base matching from which
        further search can be started.
    """
    _missing_opts = ("raise", "warn", "ignore")
    _parser_spec_type = tuple[
        tuple[str, ...],
        Optional[Callable[[Iterable], Iterable]],
        Optional[Callable[[Iterable], bool]]
    ]
    _rx_select = re.compile(r"\[(\d*):?(\d*):?(\d*)?\]")
    _rx_cmp = re.compile(r"@(~?[\w<>=][<>=\w\d]+)", re.IGNORECASE)

    def __init__(
        self,
        start: Mapping[str, Any],
        *,
        missing: Literal[_missing_opts] = _missing_opts[0]
    ) -> None:
        self.start = start
        self.along = None
        self.until = None
        self.missing = missing
        self._check_missing(missing)

    # Methods -----------------------------------------------------------------

    def search(self, along: str, until=Mapping[str, Any]) -> Self:
        """Specify search configuration."""
        self.along = along
        self.until = until
        return self

    def match(self, obj: Any, spec: Mapping[str, Any]) -> Self:
        """Match object properties against a specification."""
        missing = self.missing
        for key, cond in spec.items():
            cfunc = self._get_cond_func(cond)
            attrs, *select = self._parse_key(key)
            try:
                value = self._select_attr(obj, *attrs)
            except AttributeError as exc:
                if missing == "raise":
                    raise exc
                if missing == "warn":
                    # TODO: implement
                    raise NotImplementedError from exc
                continue
            if select:
                sfunc, tfunc = select
                value = sfunc(value)
                if tfunc:
                    n_matches = sum(1 for v in value if cfunc(v))
                    return bool(tfunc(n_matches))
                return all(cfunc(v) for v in value)
            return bool(cfunc(value))

    def execute(self, obj: Any, *, reverse: bool = False) -> Iterable[tuple[Any, ...]]:
        """Execute search programme.

        Parameters
        ----------
        obj
            Starting object.
        reverse
            Should result chains be returned in the reversed order.
        """
        def main(obj):
            if not self.match(obj, self.start):
                return
            attrs, *select = self._parse_key(self.along)
            if select:
                sfunc, tfunc = select
                if tfunc is not None:
                    raise ValueError(
                        "select specification for search cannot "
                        f"use the numeric test part as in '{self.along}'"
                    )
                def _iter(obj, chain=()):
                    if self.match(obj, self.until):
                        yield tuple(chain)
                    else:
                        for adj in sfunc(self._select_attr(obj, *attrs)):
                            new_chain = list(chain)
                            new_chain.append(adj)
                            yield from _iter(adj, chain=new_chain)
                yield from _iter(obj)
            else:
                chain = [obj]
                while not self.match(obj, self.until):
                    try:
                        obj = self._select_attr(obj, *attrs)
                    except AttributeError:
                        break
                    if obj is None or obj is obj:
                        break
                    chain.append(obj)
                yield chain

        if reverse:
            yield from reversed(tuple(main(obj)))
        else:
            yield from main(obj)


    # Internals ---------------------------------------------------------------

    def _parse_key(self, key: str) -> _parser_spec_type:
        """Parse specification key."""
        has_cmp = False
        for attr in key.split("."):
            if has_cmp:
                raise ValueError(
                    f"invalid key '{key}'; "
                    "comparison can be placed only at the end"
                )
            smatch = self._rx_select(attr)
            cmatch = self._rx_cmp(attr)
            for match in (smatch, cmatch):
                if match:
                    attr = match.sub(r"", attr)

    @staticmethod
    def _get_cmp(cmp: str, value: Any) -> Callable[[Any], bool]:
        if cmp == "=":
            return lambda x: x == value
        if cmp == "<=":
            return lambda x: x <= value
        if cmp == ">=":
            return lambda x: x >= value
        if cmp == "<":
            return lambda x: x < value
        if cmp == ">":
            return lambda x: x > value
        raise ValueError(f"incorrect comparison '{cmp}'")

    @staticmethod
    def _get_cond_func(cond: Any) -> Callable[[Any], bool]:
        if isinstance(cond, re.Pattern):
            return cond.search
        if isinstance(cond, Callable):
            return cond
        return lambda x: x == cond

    def _check_missing(self, missing: str) -> None:
        if missing not in self._missing_opts:
            raise ValueError(f"incorrect 'missing' value: '{missing}'")

    def _select_attr(self, obj: Any, *attrs: str) -> Any:
        for attr in attrs:
            try:
                obj = getattr(obj, attr)
            except AttributeError as exc:
                try:
                    obj = obj[attr]
                except KeyError:
                    raise exc
                return None
        return obj
