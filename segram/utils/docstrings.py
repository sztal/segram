"""Utility classes and methods for docstrings."""
from __future__ import annotations
from typing import Any, Literal, Callable
from functools import wraps, partial
import re


class NumpyDocString:
    """Numpy docstring parser.

    Attributes
    ----------
    sections
        Sections dictionary.
    """
    def __init__(
        self,
        docstring_or_sections: str | dict[str, Any],
        /
    ) -> None:
        """Initialization method.

        Parameters
        ----------
        docstring_or_sections
            Raw docstring text or sections dictionary.
        """
        if not docstring_or_sections or isinstance(docstring_or_sections, str):
            docstring_or_sections = self.parse_sections(docstring_or_sections)
        self.sections = docstring_or_sections

    @property
    def text(self) -> str:
        docstring = []
        for section in self.sections.values():
            header = section.get("header")
            content = section.get("content")
            lines = []
            if header:
                lines.append(header)
            if content:
                lines.append(content)
            docstring.append("\n".join(lines))
        docstring = "\n".join(docstring).strip()
        return docstring

    @staticmethod
    def parse_sections(text: str) -> dict[str, str]:
        """Parse NumpyDoc style docstring sections.

        Parameters
        ----------
        text
            Docstring text.

        Returns
        -------
        dict
            Mapping from section names to their
            raw content.
        """
        dct = {}
        if not text:
            return dct
        rx_sep = re.compile(r"(?<=\s)\n(?=\s)")
        rx_div = re.compile(r"^\s*-+\s*$")
        sections = rx_sep.split(text)
        dct["Header"] = { "content": sections[0] }
        dct["Description"] = { "content": "" }
        for section in sections[1:]:
            lines = section.split("\n")
            if len(lines) >= 2 and rx_div.search(lines[1]):
                title = lines[0].strip()
                head = "\n".join(lines[:2])
                content = "\n".join(lines[2:])
                dct[title] = {
                    "header": head,
                    "content": content
                }
            else:
                content = "\n".join(lines)
                dct["Description"]["content"] += content
        return dct

    def merge(
        self,
        other: NumpyDocString,
        *,
        __default: Literal["append", "replace"] = "append",
        **kwds: Any
    ) -> NumpyDocString:
        """Merge with ``other`` docstring.

        Parameters
        ----------
        __default
            Default merging behavior. If ``"append"`` then section
            content from ``other`` is appended to that of ``self``.
            If ``"replace"`` then replaces it.
        **kwds
            Keyword arguments can be used to set different
            merging policies (``"append"`` or ``"merge"``)
            other sections. The policy for the ``"header"``
            section is by default set to ``"replace"``,
            so it has to be overriden here change this.

        Returns
        -------
        doc
            New :class:`NumpyDocString` object.
        """
        kwds = { "Header": "replace", **kwds }
        sections = { k: v.copy() for k, v in self.sections.items() }
        for name, section in other.sections.items():
            section = section.copy()
            if name in sections:
                action = kwds.get(name.title(), __default)
                if action == "append":
                    sections[name]["content"] += section["content"]
                elif action == "replace":
                    sections[name]["content"] = \
                        section["content"].rstrip()+"\n"
                else:
                    raise ValueError(f"incorrect merge policy '{action}'")
            else:
                sections[name] = section
        return self.__class__(sections)

def _inherit_docstring(
    typ: type,
    spec: dict[str, Any] | None = None,
    *,
    default: Literal["replace", "append"] = "append",
    stop_at: type = object
) -> Callable:
    """Inherit docstring from parent.

    See :func:`inherit_docstrings` for details.
    """
    def merge_docs(obj, parent):
        # pylint: disable=attribute-defined-outside-init
        odoc = NumpyDocString(obj.__doc__)
        pdoc = NumpyDocString(parent.__doc__)
        return pdoc.merge(odoc, __default=default, **spec).text

    spec = spec or {}
    if not isinstance(typ, type):
        raise TypeError("only types can inherit docstrings")
    mro = []
    for c in typ.mro()[1:]:
        if c is stop_at:
            break
        mro.append(c)
    if not mro:
        return typ
    parent = mro[0]
    typ.__doc__ = merge_docs(typ, parent)
    for name, obj in typ.__dict__.items():
        if not isinstance(obj, Callable):
            continue
        parent_obj = getattr(parent, name, None)
        if name in parent.__abstractmethods__ and parent_obj:
            obj.__doc__ = merge_docs(obj, parent_obj)
    return typ

def inherit_docstring(*args: Any, **kwds: Any) -> Callable:
    """Decorator for inheriting and mergin docstrings from
    parent classes.

    Parameters
    ----------
    obj
        Class or a method.
    which
        Whether to inherit from the direct parent class
        or from the first abstract base class in MRO.
    spec
        Dictionary mapping section names to either
        ``"append"`` or ``"replace"``, which sets
        merge policies for different sections.
    default
        Default mergin policy.
    """
    @wraps(_inherit_docstring)
    def decorator(obj, *args, **kwds) -> Callable:
        return _inherit_docstring(obj, *args, **kwds)
    if not args or not isinstance(args[0], Callable):
        return partial(decorator, *args, **kwds)
    return decorator(*args)
