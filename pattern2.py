from typing import List, Optional, Iterable


class MatchFail(Exception):
    pass


class InvalidPattern(Exception):
    pass


class PTNode(object):
    """Parse Tree Node"""

    def __init__(self, content: str, start: int, end: int, children: List['PTNode'] = [], tag=None):
        self.content: str = content
        assert end >= start
        self.start: int = start
        self.end: int = end
        self.children: List['PTNode'] = children
        self.tag = tag

    def pformat(self, floor=0):
        header = '{}, {}'.format(self.start, self.end)
        header = header + ', {}'.format(self.tag) if self.tag else header
        body = '({}) {}'.format(header, self.content[:30])
        body = ('\t' * floor) + body
        return '\n'.join([body] + [pt.pformat(floor + 1) for pt in self.children])

    def __repr__(self):
        return self.pformat()

    def __bool__(self):
        return self.end > self.start

    @classmethod
    def merge(cls, pts):
        pts = [p for p in pts if p]
        assert pts
        if len(pts) == 1:
            return pts[0]
        content = ''.join([pt.content for pt in pts])
        pt = PTNode(content, pts[0].start, pts[-1].end, children=pts)
        return pt

    def fetch(self, tag):
        """Fetch those nodes whose tag == <tag>"""
        if self.tag == tag:
            yield self
        if self.children:
            for n in self.children:
                for nn in n.fetch(tag):
                    yield nn


class Pattern(object):
    def extract(self, text: str, start: int = 0) -> Iterable[PTNode]:
        """Extract pattern from <text>, start at <start>

        if extract fail, raise MatchFail
        """
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
        return PAny([self, self.make(pattern)])

    def __ror__(self, pattern) -> 'Pattern':
        return PAny([self.make(pattern), self])

    def search(self, text: str) -> List[PTNode]:
        """Search the pattern in text and return every match"""

        cur = 0
        ll = len(text)
        pts = []

        while cur < ll:
            try:
                for pt in self.extract(text, cur):
                    break
            except MatchFail:
                cur += 1
            else:
                pts.append(pt)
                if pt.end > cur:
                    cur = pt.end
                else:
                    cur += 1

        return pts

    def pformat(self):
        return str(self)

    def pp(self):
        print(self.pformat())


class PText(Pattern):
    """A plain pattern that just match text as it is"""

    def __init__(self, text: str):
        self.text = text

    def extract(self, text: str, start: int = 0) -> Iterable[PTNode]:
        if text[start: start + len(self.text)] == self.text:
            yield PTNode(self.text, start=start, end=start + len(self.text))
        else:
            raise MatchFail

    def __repr__(self):
        return repr(self.text)


class PAnyChar(Pattern):
    """A pattern that match any character"""

    def extract(self, text: str, start: int = 0) -> Iterable[PTNode]:
        if start < len(text):
            yield PTNode(text[start], start=start, end=start + 1)
        else:
            raise MatchFail

    def __repr__(self):
        return '.'


class PTag(Pattern):
    def __init__(self, pattern, tag):
        self.pattern = pattern
        self.tag = tag

    def extract(self, text, start=0) -> Iterable[PTNode]:
        for pt in self.pattern.extract(text, start):
            pt.tag = self.tag
            yield pt

    def __repr__(self):
        return '(?<{}>:{})'.format(self.tag, self.pattern)


class PNotInChars(Pattern):
    def __init__(self, chars: str):
        self.chars: str = chars

    def extract(self, text: str, start: int = 0) -> Iterable[PTNode]:
        if start >= len(text):
            raise MatchFail

        if text[start] not in self.chars:
            yield PTNode(text[start], start=start, end=start + 1)
        else:
            raise MatchFail

    def __repr__(self):
        return '[^{}]'.format(self.chars)


class PInChars(Pattern):
    def __init__(self, chars: str):
        self.chars: str = chars

    def extract(self, text: str, start: int = 0) -> Iterable[PTNode]:
        if start >= len(text):
            raise MatchFail
        
        if text[start] in self.chars:
            yield PTNode(text[start], start=start, end=start + 1)
        else:
            raise MatchFail

    def __repr__(self):
        return '[{}]'.format(self.chars)


class PAny(Pattern):
    def __init__(self, patterns):
        self.patterns: List[Pattern] = patterns

    def extract(self, text: str, start: int = 0) -> Iterable[PTNode]:
        matched = False
        for pattern in self.patterns:
            try:
                for pt in pattern.extract(text, start):
                    matched = True
                    yield pt
            except MatchFail:
                pass
        if not matched:
            raise MatchFail

    def __or__(self, pattern) -> Pattern:
        return PAny(list(self.patterns) + [self.make(pattern)])

    def __ror__(self, pattern) -> Pattern:
        return PAny([self.make(pattern)] + list(self.patterns))

    def __repr__(self):
        return '|'.join(['(' + str(p) + ')' for p in self.patterns])


class PRepeat(Pattern):
    def __init__(self, pattern: Pattern, _from: int, _to: int = None, greedy=False):
        self.pattern: Pattern = pattern
        self._from: int = _from
        self._to: Optional[int] = _to
        self.greedy: bool = greedy
        sub = self._prepare(pattern, _from, _to)
        self.sub = (PReversed(sub) if greedy else sub)

    def _prepare(self, pattern: Pattern, _from: int, _to: int = None) -> Pattern:
        if _to is None:
            tail = PRepeat0n(self.pattern)
        elif isinstance(_to, int):
            tail = PRepeat0n(self.pattern, _to - _from)

        cur: Pattern = tail
        for i in range(_from):
            cur = PTraverse(self.pattern, cur)

        return cur

    def extract(self, text: str, start: int = 0) -> Iterable[PTNode]:
        for pt in self.sub.extract(text, start):
            yield pt

    def __repr__(self):
        to = self._to if isinstance(self._to, int) else ''
        return '(%s){%s,%s}' % (self.pattern, self._from, to)


class PAdjacent(Pattern):
    def __init__(self, patterns: List[Pattern]):
        assert patterns
        assert len(patterns) >= 2
        self.patterns = patterns
        self.sub = self._prepare(patterns)

    def _prepare(self, patterns: List[Pattern]) -> Pattern:
        if len(patterns) == 2:
            return PTraverse(left=patterns[0], right=patterns[1])
        else:
            return PTraverse(left=patterns[0], right=self._prepare(patterns[1:]))

    def extract(self, text: str, start: int = 0) -> Iterable[PTNode]:
        for pt in self.sub.extract(text, start):
            yield pt

    def __add__(self, pattern) -> Pattern:
        return PAdjacent(list(self.patterns) + [self.make(pattern)])

    def __radd__(self, pattern) -> Pattern:
        return PAdjacent([self.make(pattern)] + list(self.patterns))

    def __repr__(self):
        return ''.join('({})'.format(p) for p in self.patterns)


class PRepeat0n(Pattern):
    def __init__(self, pattern: Pattern, _to: int = None):
        self.pattern: Pattern = pattern
        self._to: Optional[int] = _to

    def extract(self, text, start=0):
        
        def trydepth(depth: int, cur: int):
            if depth == 0:
                yield PTNode('', start=start, end=start)

            else:
                m = False
                for pt1 in self.pattern.extract(text, cur):
                    try:
                        for pt2 in trydepth(depth - 1, cur=pt1.end):
                            m = True
                            yield PTNode.merge([pt1, pt2])
                    except MatchFail:
                        pass
                if not m:
                    raise MatchFail

        d = 0
        stop = False
        matched = False
        while not stop:
            try:
                for pt in trydepth(d, start):
                    matched = True
                    yield pt
            except MatchFail:
                stop = True
            d += 1
            if isinstance(self._to, int) and d > self._to:
                break

        if not matched:
            raise MatchFail


class PTraverse(Pattern):
    def __init__(self, left: Pattern, right: Pattern):
        self.left: Pattern = left
        self.right: Pattern = right

    def extract(self, text, start=0) -> Iterable[PTNode]:
        matched = False
        for pt1 in self.left.extract(text, start):
            try:
                for pt2 in self.right.extract(text, pt1.end):
                    matched = True
                    yield PTNode.merge([pt1, pt2])
            except MatchFail:
                pass
        if not matched:
            raise MatchFail


class PReversed(Pattern):
    def __init__(self, pattern: Pattern):
        self.pattern: Pattern = pattern

    def extract(self, text, start=0):
        pts = []
        for pt in self.pattern.extract(text, start):
            pts.append(pt)
        for pt in reversed(pts):
            yield pt


class P(object):
    @staticmethod
    def anyc(chars: str) -> Pattern:
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
    def repeat(pattern, _from: int, _to: int = None, greedy=True) -> Pattern:
        """repeat a pattern some times

        if _to is None, repeat time upbound is not limited
        """
        return PRepeat(Pattern.make(pattern), _from=_from, _to=_to, greedy=greedy)

    n = repeat

    @classmethod
    def n01(cls, pattern, greedy=True) -> Pattern:
        """A pattern can be both match or not"""
        return cls.repeat(Pattern.make(pattern), 0, 1, greedy=greedy)

    ANYCHAR = PAnyChar()

    @staticmethod
    def any(*patterns) -> Pattern:
        """Try to match patterns in order, select the first one match"""
        return PAny([Pattern.make(p) for p in patterns])

    @staticmethod
    def pattern(pattern) -> Pattern:
        return Pattern.make(pattern)
