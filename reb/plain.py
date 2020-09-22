"""Reb plain Implementation"""

from typing import Iterator, List, Optional
from functools import singledispatch

from .parse_tree import PTNode, VirtualPTNode
from .pattern import (
    Finder,
    Pattern,
    PText, PAnyChar, PTag, PNotInChars, PInChars,
    PAny, PRepeat, PAdjacent,
    PExample,
    PStarting, PEnding)


__all__ = [
    'compile_pattern',
    'FinderPlain'
]


def compile_pattern(pattern: Pattern) -> Finder:
    return _compile_pattern(pattern)


@singledispatch
def _compile_pattern(pattern: Pattern) -> 'FinderPlain':
    raise TypeError


class FinderPlain(Finder):
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


@_compile_pattern.register(PText)
class FText(FinderPlain):
    def __init__(self, pattern: PText):
        super().__init__(pattern)
        self.text = pattern.text

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        if text[start: start + len(self.text)] == self.text:
            yield PTNode(text, start=start, end=start + len(self.text))


@_compile_pattern.register(PAnyChar)
class FAnyChar(FinderPlain):
    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        if start < len(text):
            yield PTNode(text, start=start, end=start + 1)


@_compile_pattern.register(PTag)
class FTag(FinderPlain):
    def __init__(self, pattern: PTag):
        super().__init__(pattern)
        self.finder = _compile_pattern(pattern.pattern)
        self.tag = pattern.tag

    def match(self, text, start=0) -> Iterator[PTNode]:
        for pt in self.finder.match(text, start):
            pt.tag = self.tag
            yield pt


@_compile_pattern.register(PNotInChars)
class FNotInChars(FinderPlain):
    def __init__(self, pattern: PNotInChars):
        super().__init__(pattern)
        self.chars = pattern.chars

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        if start >= len(text):
            return
        elif text[start] not in self.chars:
            yield PTNode(text, start=start, end=start + 1)


@_compile_pattern.register(PInChars)
class FInChars(FinderPlain):
    def __init__(self, pattern: PInChars):
        super().__init__(pattern)
        self.chars = pattern.chars

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        if start >= len(text):
            return
        elif text[start] in self.chars:
            yield PTNode(text, start=start, end=start + 1)


@_compile_pattern.register(PAny)
class FAny(FinderPlain):
    def __init__(self, pattern: PAny):
        super().__init__(pattern)
        self.finders = [_compile_pattern(p) for p in pattern.patterns]

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        for finder in self.finders:
            for pt in finder.match(text, start):
                yield pt


@_compile_pattern.register(PRepeat)
class FRepeat(FinderPlain):
    def __init__(self, pattern: PRepeat):
        super().__init__(pattern)
        self.finder = _compile_pattern(pattern.pattern)
        self._from = pattern._from
        self._to = pattern._to
        self.greedy = pattern.greedy
        sub = self._prepare(pattern, self.finder, self._from, self._to)
        self.sub = (FReversed(sub) if self.greedy else sub)

    def _prepare(self, pattern: Pattern, finder: FinderPlain, _from: int, _to: int = None) -> FinderPlain:
        tail = None
        if _to is None:
            tail = FRepeat0n(finder)
        elif isinstance(_to, int) and _from < _to:
            tail = FRepeat0n(finder, _to - _from)
        sub = [finder] * _from
        if tail:
            sub = sub + [tail]
        return FAdjacent(pattern, sub)

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        for pt in self.sub.match(text, start):
            yield pt


@_compile_pattern.register(PAdjacent)
def compile_padjacent(pattern: PAdjacent):
    return FAdjacent(pattern, [_compile_pattern(p) for p in pattern.patterns])


class FAdjacent(FinderPlain):
    def __init__(self, pattern: Pattern, finders: List[FinderPlain]):
        super().__init__(pattern)
        self.finders: List[FinderPlain] = finders

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        idx_ptn = 0
        idx_pos = start
        mtc_stk: List[Iterator[PTNode]] = [self.finders[idx_ptn].match(text, idx_pos)]
        res_stk: List[Optional[PTNode]] = [None]

        while True:
            try:
                res_nxt = next(mtc_stk[-1])
            except StopIteration:
                idx_ptn -= 1
                if idx_ptn < 0:
                    return
                mtc_stk.pop()
                res_stk.pop()
                assert res_stk[-1] is not None
                idx_pos = res_stk[-1].index1
            else:
                # assert res_stk[-1] != res_nxt
                res_stk[-1] = res_nxt
                idx_ptn += 1
                if idx_ptn < len(self.finders):
                    idx_pos = res_nxt.index1
                    mtc_stk.append(self.finders[idx_ptn].match(text, idx_pos))
                    res_stk.append(None)
                else:
                    yield PTNode.lead(res_stk)  # type: ignore
                    idx_ptn -= 1
                    assert res_stk[-1] is not None
                    idx_pos = res_stk[-1].index0


@_compile_pattern.register(PExample)
def compile_pexample(pattern: PExample):
    return _compile_pattern(pattern.pattern)


@_compile_pattern.register(PStarting)
class FStarting(FinderPlain):
    def match(self, text, start=0):
        if start == 0:
            yield PTNode(text, start=start, end=start)


@_compile_pattern.register(PEnding)
class FEnding(FinderPlain):
    def match(self, text, start=0):
        if start == len(text):
            yield PTNode(text, start=start, end=start)


class FReversed(FinderPlain):
    def __init__(self, finder: FinderPlain):
        self.finder: FinderPlain = finder

    def match(self, text, start=0):
        pts = []
        for pt in self.finder.match(text, start):
            pts.append(pt)
        for pt in reversed(pts):
            yield pt


class FRepeat0n(FinderPlain):
    def __init__(self, finder: FinderPlain, _to: int = None):
        self.finder: FinderPlain = finder
        self._to: Optional[int] = _to

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        if self._to is not None and (self._to <= 0):
            return
        # Node List NeXT
        nl_nxt = [PTNode(text, start, start)]
        yield VirtualPTNode.lead(nl_nxt)  # x* pattern can always match empty string
        nl_que = [nl_nxt]
        while nl_que:
            # Node List PREvious, which has already failed
            nl_pre = nl_que.pop(0)
            if self._to is not None and (len(nl_pre) - 1 >= self._to):
                continue
            for n2 in self.finder.match(text, nl_pre[-1].index1):
                if not n2:
                    # repeat expect it's sub pattern to proceed
                    continue
                nl_nxt = nl_pre + [n2]
                yield VirtualPTNode.lead(nl_nxt)
                nl_que.append(nl_nxt)
