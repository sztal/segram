from typing import Any
from itertools import zip_longest
from textwrap import shorten
import re
from segram.grammar import Grammar
from segram import settings


class GrammarDiff:
    """Base class for diffs between grammar objects.

    Attributes
    ----------
    results, expected
        Grammar objects to compare.
    strict
        Should strict comparison be done
    """
    rx_ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def __init__(
        self,
        res: Grammar,
        exp: Grammar,
        *,
        strict: bool = True,
        max_width: int = 160
    ) -> None:
        self.results = res
        self.expected = exp
        self.strict = strict
        self.max_width = max_width
        self.msg = settings.printer.get()

    def __repr__(self) -> str:
        return self.to_str()

    def __call__(self) -> dict[str, Any]:
        """Compute diff between ``self.res`` and ``self.exp``."""
        diff = []
        for name, r, e in self.results.iter_diffs(self.expected, strict=self.strict):
            diff.append({
                "name": name,
                "res": r,
                "exp": e
            })
        return diff

    # Methods -----------------------------------------------------------------

    def to_str(self) -> str:
        """Generate diff string.

        Parameters
        ----------
        color
            Should colors be used.
        """
        s = ""
        for diff in self():
            name, r, e = diff.values()
            name = str(name)
            if "PhraseGraph" in name:
                r = r.to_str() if r else r
                e = e.to_str() if e else e
            elif isinstance(r, tuple | list) and isinstance(e, tuple | list):
                r = "\n".join(map(str, r))
                e = "\n".join(map(str, e))
            else:
                r = repr(r)
                e = repr(e)
            div = self.msg.divider(name).lstrip()
            s += div+"\n\n"
            s += self.side_by_side(r, e)
            s += "\n"
        return s

    def side_by_side(self, s1: str, s2: str) -> str:
        """Present two strings side by side."""
        lines1 = s1.split("\n")
        lines2 = s2.split("\n")
        strings = [
            (self.shorten(_s1), self.shorten(_s2))
            for _s1, _s2 in zip_longest(lines1, lines2)
        ]
        maxwidth = max(self.strlens(s[0])[0] for s in strings)
        parts = []
        for _s1, _s2 in strings:
            _s1 = self.rpad(_s1, maxwidth)
            _s2 = self.rpad(_s2, maxwidth)
            div = self.msg.color("|", fg="white", bg="red", bold=True)
            parts.append(_s1+" "+div+" "+_s2)
        return "\n".join(parts)

    def strlens(self, s: str) -> tuple[int, int]:
        """Lengths of ``s`` with and without ANSI escape characters."""
        return len(self.rx_ansi_escape.sub(r"", s)), len(s)

    def rpad(self, s: str, width: int | None = None) -> str:
        """Right-pad string to be consistent with ``width``."""
        width = width or self.max_width
        l, _ = self.strlens(s)
        if (diff := width - l) > 0:
            s += " "*diff
        return s

    def shorten(self, s: str) -> str:
        """Shorten string so it fits in the half of ``self.max_width``."""
        if not s:
            return ""
        a, _ = self.strlens(s)
        if (ratio := a / self.max_width) > 1:
            s = shorten(s, int(len(s)/ratio))
        return s
