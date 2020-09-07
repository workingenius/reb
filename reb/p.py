from typing import List, Optional
from itertools import permutations

from .pattern import PTNode
from .pattern import InvalidPattern
from .pattern import Pattern, PInChars, PNotInChars, PTag, PRepeat, PAny, PExample, PAdjacent, PAnyChar, PStarting, PEnding
from .cache import PCached


def pcached(call):
    # as decorator
    def ncall(*args, **kwargs):
        ptn = call(*args, **kwargs)
        return PCached(ptn)
    return ncall


class P(object):
    @staticmethod
    @pcached
    def ic(chars: str) -> Pattern:
        """ANY Char"""
        return PInChars(chars)

    @staticmethod
    @pcached
    def nic(chars: str) -> Pattern:
        """Not In Chars"""
        return PNotInChars(chars)

    @staticmethod
    def tag(pattern, tag) -> Pattern:
        """tag a pattern"""
        return PTag(Pattern.make(pattern), tag=tag)

    @staticmethod
    @pcached
    def repeat(pattern, _from: int = None, _to: int = None, greedy=True, exact: int = None) -> Pattern:
        """repeat a pattern some times

        if _to is None, repeat time upbound is not limited
        """
        if exact is not None:
            _from = exact
            _to = exact

        if _from is None:
            _from = 0

        if _to is not None and _to < _from:
            raise InvalidPattern('Repeat upper bound less than lower bound')
        
        return PRepeat(Pattern.make(pattern), _from=_from, _to=_to, greedy=greedy)

    n = repeat

    @classmethod
    @pcached
    def n01(cls, pattern, greedy=True) -> Pattern:
        """A pattern can be both match or not"""
        return cls.repeat(Pattern.make(pattern), 0, 1, greedy=greedy)

    @staticmethod
    @pcached
    def any(*patterns) -> Pattern:
        """Try to match patterns in order, select the first one match"""
        return PAny([Pattern.make(p) for p in patterns])

    @staticmethod
    @pcached
    def pattern(pattern) -> Pattern:
        return Pattern.make(pattern)

    @staticmethod
    def example(*args):
        pat = None
        exs = []

        for arg in args:
            if isinstance(arg, Pattern):
                if pat is not None:
                    raise ValueError('more than one pattern is given')
                pat = arg
            elif isinstance(arg, str):
                exs.append(arg)
            elif isinstance(arg, list):
                for e in arg:
                    assert isinstance(e, str)
                exs.extend(arg)

        return PExample(pat, exs)

    @staticmethod
    @pcached
    def onceeach(*patterns, seperator=None):
        """For given patterns, appear once for each (without caring order)"""
        ptn_lst = [Pattern.make(p) for p in patterns]
        ptn_sep = Pattern.make(seperator)

        alt_lst: List[Pattern] = []
        for sub_ptn_lst in permutations(ptn_lst):
            if sub_ptn_lst and ptn_sep:
                _sub_ptn_lst = [None] * (len(sub_ptn_lst) * 2 - 1)
                for i in range(len(_sub_ptn_lst)):
                    if i % 2 == 0:
                        _sub_ptn_lst[i] = sub_ptn_lst[i // 2]
                    else:
                        _sub_ptn_lst[i] = ptn_sep
                sub_ptn_lst = _sub_ptn_lst
            alt_lst.append(
                PAdjacent(sub_ptn_lst)
            )
        return PAny(alt_lst)

    ANYCHAR: Pattern
    STARTING: Pattern
    ENDING: Pattern
    NEWLINE: Pattern


P.ANYCHAR = PCached(PAnyChar())
P.STARTING = PCached(PStarting())
P.ENDING = PCached(PEnding())
P.NEWLINE = PCached(P.any(P.pattern('\r\n'), P.ic('\r\n')))
