"""
Helpers for functional programming style.
"""
import itertools


def head(l):
    """
    Returns the first element of a list if available or an empty list otherwise.
    """
    if l:
        return l[0]
    else:
        return []


def nth(generator, n):
    """
    Return the nth element of a generator.
    """
    return list(itertools.islice(generator, n))[n-1]
