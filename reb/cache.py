from typing import List, Iterator, Optional
from collections import deque

from .pattern import Pattern, PTNode


class Cache(object):
    def __init__(self):
        self.table = deque()
        self.index = 0

    def release(self, before: int) -> None:
        """desert all cached data whose index < <before>"""
        if before <= self.index:
            return
        pos = before - self.index
        if pos >= len(self.table):
            self.table.clear()
            self.index = before
            return
        for i in range(pos):
            self.table.popleft()
        self.index += pos

    def updated_match(self, pattern: Pattern, text: str, start: int):
        pos = start - self.index
        if pos < 0:
            return pattern.match(text, start, cache=self)
        if pos >= len(self.table):
            self.table.extend({} for i in range(pos - len(self.table) + 1))
            assert len(self.table) == (pos + 1)
        dct = self.table[pos]
        cac = dct.get(pattern)
        if cac is None:
            dct[pattern] = 1
            return pattern.match(text, start, cache=self)
        elif isinstance(cac, CachedMatch):
            return iter(cac)
        elif cac > 0:
            cac = CachedMatch(pattern.match(text, start, cache=self))
            dct[pattern] = cac
            return iter(cac)
            

class CachedMatch(object):
    def __init__(self, match: Iterator[PTNode]):
        self.match: Iterator[PTNode] = match
        self.done: List[PTNode] = []
        self.is_all_done = False

    def __iter__(self):
        # NON THREAD SAFE
        i = 0
        while True:
            if i < len(self.done):
                yield self.done[i]
                i += 1
            elif self.is_all_done:
                return
            elif not self.is_all_done:
                try:
                    nxt = next(self.match)
                except StopIteration:
                    self.is_all_done = True
                    return
                else:
                    self.done.append(nxt)
                    yield nxt
                    i += 1


class PCached(Pattern):
    def __init__(self, pattern: Pattern):
        self.pattern: Pattern = pattern

    def match(self, text, start=0, cache=None) -> Iterator[PTNode]:
        # if not isinstance(self.pattern, PCached):
        #     print(start, self.pattern)
        # return self.pattern.match(text, start, cache=cache)

        if isinstance(cache, Cache):
            return cache.updated_match(self.pattern, text, start)
        else:
            return self.pattern.match(text, start, cache=cache)

    @property
    def re(self):
        return self.pattern.re

    # TODO should not have this two property

    @property
    def _from(self):
        return self.pattern._from

    @property
    def _to(self):
        return self.pattern._to
