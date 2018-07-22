"""
Helpers for functional programming style.
"""


def head(l):
    """
    Returns the first element of a list if available or an empty list otherwise.
    """
    if l:
        return l[0]
    else:
        return []

