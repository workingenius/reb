"""Reb plain Implementation"""

from typing import Iterator
from functools import singledispatch

from .parse_tree import PTNode
from .pattern import (
    Pattern,
    PText, PAnyChar, PTag, PNotInChars, PInChars,
    PAny, PRepeat)


__all__ = [
    'compile_pattern',
    'FinderPlain'
]


@singledispatch
def compile_pattern(pattern: Pattern) -> FinderPlain:
    raise TypeError


class FinderPlain(object):
    def __init__(self, pattern: Pattern):
        self.pattern: Pattern = pattern

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        """Match pattern from <text>, start at <start>, iterate over all possible matches as PTNode"""
        raise NotImplementedError

    def finditer(self, text: str) -> Iterator[PTNode]:
        """Find pattern in text, yield them one after another"""
        cur = 0
        ll = len(text)

        while cur <= ll:
            m = False
            for pt in self.match(text, cur):
                m = True
                yield pt
                if pt.index1 > cur:
                    cur = pt.index1
                else:
                    cur += 1
                break
            if not m:
                cur += 1


@compile_pattern.register(PText)
class FText(FinderPlain):
    def __init__(self, pattern: PText):
        super().__init__(pattern)
        self.text = pattern.text

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        if text[start: start + len(self.text)] == self.text:
            yield PTNode(text, start=start, end=start + len(self.text))


@compile_pattern.register(PAnyChar)
class FAnyChar(FinderPlain):
    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        if start < len(text):
            yield PTNode(text, start=start, end=start + 1)


@compile_pattern.register(PTag)
class FTag(FinderPlain):
    def __init__(self, pattern: PTag):
        super().__init__(pattern)
        self.finder = compile_pattern(pattern.pattern)
        self.tag = pattern.tag

    def match(self, text, start=0) -> Iterator[PTNode]:
        for pt in self.finder.match(text, start):
            pt.tag = self.tag
            yield pt


@compile_pattern.register(PNotInChars)
class FNotInChars(FinderPlain):
    def __init__(self, pattern: PNotInChars):
        super().__init__(pattern)
        self.chars = pattern.chars

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        if start >= len(text):
            return
        elif text[start] not in self.chars:
            yield PTNode(text, start=start, end=start + 1)


@compile_pattern.register(PInChars)
class FInChars(FinderPlain):
    def __init__(self, pattern: PInChars):
        super().__init__(pattern)
        self.chars = pattern.chars

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        if start >= len(text):
            return
        elif text[start] in self.chars:
            yield PTNode(text, start=start, end=start + 1)


@compile_pattern.register(PAny)
class FAny(FinderPlain):
    def __init__(self, pattern: PAny):
        super().__init__(pattern)
        self.finders = [compile_pattern(p) for p in pattern.patterns]

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        for finder in self.finders:
            for pt in finder.match(text, start):
                yield pt


class FRepeat(FinderPlain):
    def __init__(self, pattern: PRepeat):
        super().__init__()
        self.finder = compile_pattern(pattern.pattern)
        self._from = pattern._from
        self._to = pattern._to
        self.greedy = pattern.greedy
        self.sub = self._prepare()