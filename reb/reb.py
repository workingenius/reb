from itertools import permutations
from typing import List, Optional, Iterator

from .parse_tree import PTNode, VirtualPTNode


class RebException(Exception):
    pass


class InvalidPattern(RebException):
    pass


class ExampleFail(RebException):
    pass


class Pattern(object):
    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        """Match pattern from <text>, start at <start>, iterate over all possible matches as PTNode"""
        raise NotImplementedError

    @classmethod
    def make(cls, o):
        if isinstance(o, Pattern):
            return o
        elif isinstance(o, str):
            return PText(o)
        else:
            raise InvalidPattern

    def __add__(self, pattern) -> 'Pattern':
        return PAdjacent([self, self.make(pattern)])

    def __radd__(self, pattern) -> 'Pattern':
        return PAdjacent([self.make(pattern), self])

    def __or__(self, pattern) -> 'Pattern':
        return PClause([self, self.make(pattern)])

    def __ror__(self, pattern) -> 'Pattern':
        return PClause([self.make(pattern), self])

    def extract(self, text: str) -> List[PTNode]:
        """Extract info from text by the pattern, and return every match, forming a parse tree"""
        return list(self.extractiter(text))

    extractall = extract

    def extractiter(self, text: str) -> Iterator[PTNode]:
        """Extract info from text by the pattern, and return every match, forming parse trees"""
        for n in self.finditer(text):
            if n:
                yield n

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

    def findall(self, text: str) -> List[str]:
        sl = []
        for n in self.finditer(text):
            sl.append(n.string)
        return sl

    def pformat(self):
        return str(self)

    def pp(self):
        print(self.pformat())

    @property
    def re(self):
        from .to_re import to_re
        return to_re(self)


class PText(Pattern):
    """A plain pattern that just match text as it is"""

    def __init__(self, text: str):
        self.text = text

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        if text[start: start + len(self.text)] == self.text:
            yield PTNode(text, start=start, end=start + len(self.text))

    def __repr__(self):
        return repr(self.text)


class PAnyChar(Pattern):
    """A pattern that match any character"""

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        if start < len(text):
            yield PTNode(text, start=start, end=start + 1)

    def __repr__(self):
        return '.'


class PTag(Pattern):
    def __init__(self, pattern, tag):
        self.pattern = pattern
        assert tag is not None
        self.tag = tag

    def match(self, text, start=0) -> Iterator[PTNode]:
        for pt in self.pattern.match(text, start):
            pt.tag = self.tag
            yield pt

    def __repr__(self):
        return '(?<{}>:{})'.format(self.tag, self.pattern)


class PNotInChars(Pattern):
    def __init__(self, chars: str):
        self.chars: str = chars

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        if start >= len(text):
            return
        elif text[start] not in self.chars:
            yield PTNode(text, start=start, end=start + 1)

    def __repr__(self):
        return '[^{}]'.format(self.chars)


class PInChars(Pattern):
    def __init__(self, chars: str):
        self.chars: str = chars

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        if start >= len(text):
            return
        elif text[start] in self.chars:
            yield PTNode(text, start=start, end=start + 1)

    def __repr__(self):
        return '[{}]'.format(self.chars)


class PAny(Pattern):
    def __init__(self, patterns):
        self.patterns: List[Pattern] = patterns

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        for pattern in self.patterns:
            for pt in pattern.match(text, start):
                yield pt

    def __or__(self, pattern) -> Pattern:
        return PClause(list(self.patterns) + [self.make(pattern)])

    def __ror__(self, pattern) -> Pattern:
        return PClause([self.make(pattern)] + list(self.patterns))

    def __repr__(self):
        return '|'.join(['(' + str(p) + ')' for p in self.patterns])


class PClause(PAny):
    """A Pattern Clause
    
    A pattern with many clauses is basically the same as PAny, but with example checks.
      1. A clause must pass all example checks, that is, must has search results.
      2. Clauses, should not conflict with each other.
         That is to say, search result of whole pattern should be the same as a single clause
    """
    def __init__(self, patterns: List[Pattern]):
        super().__init__([p for p in patterns])

        exa_lst: List[Example] = []
        for p in patterns:
            if isinstance(p, PExample):
                exa_lst.extend(p.examples)
        
        for exa in exa_lst:
            if self.extract(exa.text) != exa.extraction:
                raise ExampleFail('Clauses conflict occurs')


class PRepeat(Pattern):
    def __init__(self, pattern: Pattern, _from: int, _to: int = None, greedy=False):
        self.pattern: Pattern = pattern
        if _to is not None:
            assert _to >= _from
        self._from: int = _from
        self._to: Optional[int] = _to
        self.greedy: bool = greedy
        sub = self._prepare(pattern, _from, _to)
        self.sub = (PReversed(sub) if greedy else sub)

    def _prepare(self, pattern: Pattern, _from: int, _to: int = None) -> Pattern:
        tail = None
        if _to is None:
            tail = PRepeat0n(self.pattern)
        elif isinstance(_to, int) and _from < _to:
            tail = PRepeat0n(self.pattern, _to - _from)
        sub = [pattern] * _from
        if tail:
            sub = sub + [tail]
        return PAdjacent(sub)

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        for pt in self.sub.match(text, start):
            yield pt

    def __repr__(self):
        to = self._to if isinstance(self._to, int) else ''
        return '(%s){%s,%s}' % (self.pattern, self._from, to)


class PAdjacent(Pattern):
    def __init__(self, patterns: List[Pattern]):
        assert patterns
        assert len(patterns) >= 1
        self.patterns = patterns

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        idx_ptn = 0
        idx_pos = start
        mtc_stk: List[Iterator[PTNode]] = [self.patterns[idx_ptn].match(text, idx_pos)]
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
                if idx_ptn < len(self.patterns):
                    idx_pos = res_nxt.index1
                    mtc_stk.append(self.patterns[idx_ptn].match(text, idx_pos))
                    res_stk.append(None)
                else:
                    yield PTNode.lead(res_stk)  # type: ignore
                    idx_ptn -= 1
                    assert res_stk[-1] is not None
                    idx_pos = res_stk[-1].index0

    def __add__(self, pattern) -> Pattern:
        return PAdjacent(list(self.patterns) + [self.make(pattern)])

    def __radd__(self, pattern) -> Pattern:
        return PAdjacent([self.make(pattern)] + list(self.patterns))

    def __repr__(self):
        return ''.join('({})'.format(p) for p in self.patterns)


# Several helper Patterns


class PRepeat0n(Pattern):
    def __init__(self, pattern: Pattern, _to: int = None):
        self.pattern: Pattern = pattern
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
            for n2 in self.pattern.match(text, nl_pre[-1].index1):
                if not n2:
                    # repeat expect it's sub pattern to proceed
                    continue
                nl_nxt = nl_pre + [n2]
                yield VirtualPTNode.lead(nl_nxt)
                nl_que.append(nl_nxt)


class PReversed(Pattern):
    def __init__(self, pattern: Pattern):
        self.pattern: Pattern = pattern

    def match(self, text, start=0):
        pts = []
        for pt in self.pattern.match(text, start):
            pts.append(pt)
        for pt in reversed(pts):
            yield pt


class Example(object):
    def __init__(self, text: str, pattern: Pattern):
        self.text: str = text
        self.pattern: Pattern = pattern
        self.extraction = pattern.extract(text)
        if not self.extraction:
            raise ExampleFail


class PExample(Pattern):
    """A Pattern with example"""

    def __init__(self, pattern: Pattern, examples):
        self.pattern: Pattern = pattern
        self.examples: List[Example] = [Example(e, pattern) for e in examples if e]

    @property
    def has_example(self) -> bool:
        return bool(self.examples)

    def __repr__(self):
        return '(' + repr(self.pattern) + ') **with {} examples**'.format(len(self.examples))

    def match(self, text, start=0):
        # should not reach this line
        return self.pattern.match(text, start)


class PStarting(Pattern):
    def match(self, text, start=0):
        if start == 0:
            yield PTNode(text, start=start, end=start)


class PEnding(Pattern):
    def match(self, text, start=0):
        if start == len(text):
            yield PTNode(text, start=start, end=start)


class P(object):
    @staticmethod
    def ic(chars: str) -> Pattern:
        """ANY Char"""
        return PInChars(chars)

    @staticmethod
    def nic(chars: str) -> Pattern:
        """Not In Chars"""
        return PNotInChars(chars)

    @staticmethod
    def tag(pattern, tag) -> Pattern:
        """tag a pattern"""
        return PTag(Pattern.make(pattern), tag=tag)

    @staticmethod
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
    def n01(cls, pattern, greedy=True) -> Pattern:
        """A pattern can be both match or not"""
        return cls.repeat(Pattern.make(pattern), 0, 1, greedy=greedy)

    @staticmethod
    def any(*patterns) -> Pattern:
        """Try to match patterns in order, select the first one match"""
        return PAny([Pattern.make(p) for p in patterns])

    @staticmethod
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


P.ANYCHAR = PAnyChar()
P.STARTING = PStarting()
P.ENDING = PEnding()
P.NEWLINE = P.any(P.pattern('\r\n'), P.ic('\r\n'))
