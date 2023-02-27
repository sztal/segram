"""Printing methods for visualization."""
# pylint: disable=redefined-outer-name
from typing import Any, Optional
from wasabi import color, Printer as WasabiPrinter
from .settings import Settings
from ..symbols import Role


class Printer(WasabiPrinter):

    def color(
        self,
        text: str,
        fg: Optional[int | str] = None,
        bg: Optional[int | str] = None,
        bold: bool = False,
        underline: bool = False
    ) -> str:
        """Color text by applying ANSI escape sequence.

        Parameters
        ----------
        text
            Text to be formatted.
        fg, bg
            Foreground and background colors as integers or their
            names to be found in ``self.colors`` when passed as strings.
            ``None`` is used when the name is not found.
        bold
            Format text in bold.
        underline
            Underline text.
        """
        fg = self.colors.get(fg) if isinstance(fg, str) else fg
        bg = self.colors.get(bg) if isinstance(bg, str) else bg
        return color(text, fg, bg, bold, underline)


printer_settings = Settings(
    default="dark",
    dark = Printer(colors={
        # Standard colors
        "good":   10, # green
        "fail":    9, # red
        "warn":   11, # yellow
        "info":   14, # cyan
        # Grammar colors
        "subj":    10, # green
        "verb":   196, # red
        "noun":   220, # gold
        "dobj":    12, # blue
        "iobj":     3, # yellow
        "xcomp":  209, # orange
        "pobj":     6, # cyan
        "desc":   219, # pink
        "prep":   191, # yellowish-greenish
        "neg":    256, # white
        "intj":   215, # orange
        # Grammar background colors
        "bg_neg": 196, # red
    }, no_print=True),
    light = Printer(colors={
        # Grammar colors
        "subj":     2, # green
        "verb":     1, # red
        "noun":   136, # goldish
        "dobj":     4, # blue
        "iobj":     3, # gold
        "xcomp":  202, # orange
        "pobj":    30, # cyan
        "desc":   200, # pink
        "prep":    90, # yellowish-greenish
        "neg":    256, # white
        "intj":   210, # orangish
        # Grammar background colors
        "bg_neg": 196, # red
    }, no_print=True)
)


def color_role(
    text: str,
    role: Optional[str | Role] = None,
    *,
    color: bool = True,
    cmap: Optional[str] = None,
    **kwds: Any
) -> str:
    """Color token based on its syntactic role.

    Parameters
    ----------
    tok
        Token as string.
    role
        Phrasal role of ``tok``.
        If it is not recognized then no coloring is done.
    color
        Should coloring be done.
    cmap
        Name of the colormap to use.
        If ``None`` then default colormap is used.
    **kwds
        Passed to :meth:`~segram.colors.Printer.color`.
    """
    msg = printer_settings.get(cmap)
    if color:
        if role is not None:
            role = str(role)
        kwds = dict(fg=role, bg="bg_"+role if role else None)
    return msg.color(text, **kwds)
