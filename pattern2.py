from typing import List, Optional


class MatchFail(Exception):
    pass


class InvalidPattern(Exception):
    pass


class PTNode(object):
    """Parse Tree Node"""

    def __init__(self, content: str, start: int, end: int, children: List['PTNode'] = [], tag=None):
        self.content: str = content
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


class Pattern(object):
    def extract(self, text: str, start: int = 0) -> PTNode:
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
                pt = self.extract(text, cur)
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

    def extract(self, text: str, start: int = 0) -> PTNode:
        print('text', start)
        if text[start: start + len(self.text)] == self.text:
            return PTNode(self.text, start=start, end=start + len(self.text))
        else:
            raise MatchFail

    def __repr__(self):
        return repr(self.text)


class PAnyChar(Pattern):
    """A pattern that match any character"""

    def extract(self, text: str, start: int = 0) -> PTNode:
        print('anychar', start)
        if start < len(text):
            return PTNode(text[start], start=start, end=start + 1)
        else:
            raise MatchFail

    def __repr__(self):
        return '.'


class PTag(Pattern):
    def __init__(self, pattern, tag):
        self.pattern = pattern
        self.tag = tag

    def extract(self, text, start=0):
        print('tag', start)
        pt = self.pattern.extract(text, start)
        pt.tag = self.tag
        return pt

    def __repr__(self):
        return '(?<{}>:{})'.format(self.tag, self.pattern)


class PNotInChars(Pattern):
    def __init__(self, chars: str):
        self.chars: str = chars

    def extract(self, text: str, start: int = 0):
        print('notinchars', start)
        if start >= len(text):
            raise MatchFail

        if text[start] not in self.chars:
            return PTNode(text[start], start=start, end=start + 1)
        else:
            raise MatchFail

    def __repr__(self):
        return '[^{}]'.format(self.chars)


class PInChars(Pattern):
    def __init__(self, chars: str):
        self.chars: str = chars

    def extract(self, text: str, start: int = 0):
        print('inchars', start)
        if start >= len(text):
            raise MatchFail
        
        if text[start] in self.chars:
            return PTNode(text[start], start=start, end=start + 1)
        else:
            raise MatchFail

    def __repr__(self):
        return '[{}]'.format(self.chars)


class PAny(Pattern):
    def __init__(self, patterns):
        self.patterns: List[Pattern] = patterns

    def extract(self, text: str, start: int = 0):
        print('any', start)
        pt = None
        for pattern in self.patterns:
            try:
                pt = pattern.extract(text, start)
            except MatchFail:
                pass
            if pt is not None:
                break
        if pt is None:
            raise MatchFail
        return pt

    def __or__(self, pattern) -> Pattern:
        return PAny(list(self.patterns) + [self.make(pattern)])

    def __ror__(self, pattern) -> Pattern:
        return PAny([self.make(pattern)] + list(self.patterns))

    def __repr__(self):
        return '|'.join(['(' + str(p) + ')' for p in self.patterns])


class PRepeat(Pattern):
    def __init__(self, pattern: Pattern, _from: int, _to: int = None):
        self.pattern: Pattern = pattern
        self._from: int = _from
        self._to: Optional[int] = _to

    def extract(self, text: str, start: int = 0):
        print('repeat', start)
        times = 0  # how many times has matched

        pts = []

        while True:
            cur = start
            try:
                pt = self.pattern.extract(text, cur)
            except MatchFail:
                break
            else:
                times += 1
                pts.append(pt)
                cur = pt.end
                if isinstance(self._to, int) and times >= self._to:
                    break

        if not pts and self._from == 0:
            return PTNode('', start=start, end=start)

        if self._from <= times and (isinstance(self._to, int) and times <= self._to):
            content = ''.join([pt.content for pt in pts])
            pt = PTNode(content, pts[0].start, pts[-1].end, children=pts)
            return pt

        raise MatchFail

    def __repr__(self):
        to = self._to if isinstance(self._to, int) else ''
        return '(%s){%s,%s}' % (self.pattern, self._from, to)


class PAdjacent(Pattern):
    def __init__(self, patterns: List[Pattern]):
        assert patterns
        self.patterns = patterns

    def extract(self, text: str, start: int = 0) -> PTNode:
        print('adjacent', start)
        pts = []
        cur = start
        for pattern in self.patterns:
            pt = pattern.extract(text, cur)
            pts.append(pt)
            cur = pt.end
        return PTNode(
            ''.join([pt.content for pt in pts]),
            pts[0].start,
            pts[-1].end,
            children=pts
        )

    def __add__(self, pattern) -> Pattern:
        return PAdjacent(list(self.patterns) + [self.make(pattern)])

    def __radd__(self, pattern) -> Pattern:
        return PAdjacent([self.make(pattern)] + list(self.patterns))

    def __repr__(self):
        return ''.join('({})'.format(p) for p in self.patterns)


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
    def repeat(pattern, _from: int, _to: int = None) -> Pattern:
        """repeat a pattern some times

        if _to is None, repeat time upbound is not limited
        """
        return PRepeat(Pattern.make(pattern), _from=_from, _to=_to)

    n = repeat

    @classmethod
    def n01(cls, pattern) -> Pattern:
        """A pattern can be both match or not"""
        return cls.repeat(Pattern.make(pattern), 0, 1)

    ANYCHAR = PAnyChar()

    @staticmethod
    def any(*patterns) -> Pattern:
        """Try to match patterns in order, select the first one match"""
        return PAny([Pattern.make(p) for p in patterns])
