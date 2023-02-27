"""Versioning utilities."""
import re
import operator
from packaging import version


def is_correct_version(v: str, constraints: str) -> bool:
    """Check if a version string is correct with respect to constraints.

    Parameters
    ----------
    v
        Semantic version string.
    constraints
        Constraints for semantic versions in one of the following forms:

        * ``>version``
        * ``>=version``
        * ``<version``
        * ``<=version``
        * ``[constraint1],[constraint2]`` where the two constraints
        follow given consistent bounds using the forms above.

    Returns
    -------
    bool
        Indicate whether the constraints are met.
    """
    v = version.parse(v)
    ops = {
        "<=": operator.le,
        ">=": operator.ge,
        "<": operator.lt,
        ">": operator.gt
    }
    rx = re.compile(r"[<>]?[=]?", re.IGNORECASE)
    for const in constraints.strip().split(","):
        const = const.strip()
        for k, op in ops.items():
            if const.startswith(k):
                req = rx.sub(r"", const).strip()
                req = version.parse(req)
                if op(v, req):
                    return True
    return False
