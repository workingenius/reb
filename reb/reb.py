from typing import List, Optional, Iterator


class RebException(Exception):
    pass


class InvalidPattern(RebException):
    pass


class ExampleFail(RebException):
    pass


class PTNode(object):
    """Parse Tree Node"""

    def __init__(self, text: str, start: int, end: int, children: List['PTNode'] = [], tag=None):
        self.text: str = text
        assert end >= start >= 0
        self.index0: int = start
        self.index1: int = end
        self.children: List['PTNode'] = children
        self.tag = tag

    @property
    def content(self) -> str:
        return self.text[self.index0: self.index1]

    @property
    def string(self) -> str:
        return self.content

    def start(self):
        return self.index0

    def end(self):
        return self.index1

    def pformat(self, floor=0):
        header = '{}, {}'.format(self.index0, self.index1)
        header = header + ', {}'.format(self.tag) if self.tag else header
        body = '({}) {}'.format(header, self.content[:30])
        body = ('\t' * floor) + body
        return '\n'.join([body] + [pt.pformat(floor + 1) for pt in self.children])

    def __repr__(self):
        return self.pformat()

    def __bool__(self):
        return self.index1 > self.index0

    @classmethod
    def lead(cls, pts: List['PTNode']) -> 'PTNode':
        """Make a new PTNode as the common parent of nodes <pts> """
        
        for p1, p2 in zip(pts[:-1], pts[1:]):
            assert p1.index1 == p2.index0
            
        pt = PTNode(pts[0].text, pts[0].index0, pts[-1].index1, children=pts)
        return pt

    def fetch(self, tag):
        """Fetch those nodes whose tag == <tag>"""
        if self.tag == tag:
            yield self
        if self.children:
            for n in self.children:
                for nn in n.fetch(tag):
                    yield nn

    def drop(self) -> 'PTNode':
        """Copy the PTNode but without children"""
        return self.__class__(self.text, self.index0, self.index1, tag=self.tag)

    def __eq__(self, o):
        if isinstance(o, PTNode):
            return self.text == o.text \
                and self.index0 == o.index0 \
                and self.index1 == o.index1 \
                and self.children == o.children \
                and self.tag == o.tag
        return False

    def pp(self):
        try:
            from termcolor import colored
        except ImportError:
            print('Module termcolor is needed')

        else:
            # for i in tag_lst, tag_lst[i] is the tag most close to the leaf
            tag_lst = [None] * (self.index1 - self.index0)

            start0 = self.index0

            def set_tag(node, tl):
                """Traverse the parse tree and set tag_lst"""
                if node.tag is not None:
                    for i in range(node.index0 - start0, node.index1 - start0):
                        tl[i] = node.tag
                for cn in node.children:
                    set_tag(cn, tl)

            set_tag(self, tag_lst)

            colors = ['red', 'green', 'yellow', 'blue', 'magenta']
            white = 'white'

            def tag_color(tag):
                if tag is None:
                    return white
                return colors[sum(map(ord, tag)) % len(colors)]

            color_lst = [tag_color(tag) for tag in tag_lst]

            # extend several chars on both sides
            extend_n = 10
            left_i = max(0, self.index0 - extend_n)
            right_i = min(len(self.text), self.index1 + extend_n)

            left_str = self.text[left_i: self.index0]
            left_str = ('...' + left_str) if left_i > 0 else left_str

            right_str = self.text[self.index1: right_i]
            right_str = (right_str + '...') if right_i < len(self.text) else right_str

            # whole string to print
            ws = colored(left_str, attrs=['dark'])
            for offset, color in enumerate(color_lst):
                i = self.index0 + offset
                ws += colored(self.text[i], color)
            ws += colored(right_str, attrs=['dark'])

            print(ws)


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

        cur = 0
        ll = len(text)
        pts = []

        while cur < ll:
            m = False
            for pt in self.match(text, cur):
                if pt:
                    m = True
                    pts.append(pt)
                    if pt.index1 > cur:
                        cur = pt.index1
                    else:
                        cur += 1
                    break
            if not m:
                cur += 1

        return pts

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
        return PAny(list(self.patterns) + [self.make(pattern)])

    def __ror__(self, pattern) -> Pattern:
        return PAny([self.make(pattern)] + list(self.patterns))

    def __repr__(self):
        return '|'.join(['(' + str(p) + ')' for p in self.patterns])


class PClause(PAny):
    """A Pattern Clause
    
    A pattern with many clauses is basically the same as PAny, but with example checks.
      1. A clause must pass all example checks, that is, must has search results.
      2. # TODO Clauses should not conflict with each other.
         That is to say, search result of whole pattern should be the same as a single clause
    """
    def __init__(self, patterns: List[Pattern]):
        for p in patterns:
            self.check(p)
        super().__init__([self.real_pattern(p) for p in patterns])

    def __or__(self, pattern: Pattern) -> Pattern:
        self.check(pattern)
        return super().__or__(self.real_pattern(pattern))

    def __ror__(self, pattern: Pattern) -> Pattern:
        self.check(pattern)
        return super().__ror__(self.real_pattern(pattern))

    @classmethod
    def check(cls, pattern: Pattern):
        """Check pattern on the example, possibly raise ExampleFail"""
        assert isinstance(pattern, Pattern)
        assert isinstance(pattern, PExample) and pattern.has_example, \
            'Pattern as a clause must has at least one example'
        real_pattern = cls.real_pattern(pattern)
        for e in pattern.examples:
            if not real_pattern.extract(e):
                raise ExampleFail

    @staticmethod
    def real_pattern(p):
        if isinstance(p, PExample):
            return p.pattern
        elif isinstance(p, Pattern):
            return p
        else:
            raise TypeError


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
        if _to is None:
            tail = PRepeat0n(self.pattern)
        elif isinstance(_to, int):
            tail = PRepeat0n(self.pattern, _to - _from)
        return PAdjacent([pattern] * _from + [tail])

    def match(self, text: str, start: int = 0) -> Iterator[PTNode]:
        for pt in self.sub.match(text, start):
            yield pt.drop()

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
                assert res_stk[-1] != res_nxt
                res_stk[-1] = res_nxt
                idx_ptn += 1
                if idx_ptn < len(self.patterns):
                    idx_pos = res_nxt.index1
                    mtc_stk.append(self.patterns[idx_ptn].match(text, idx_pos))
                    res_stk.append(None)
                else:
                    yield PTNode.lead(res_stk)
                    idx_ptn -= 1
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
        # Node List NeXT
        nl_nxt = [PTNode(text, start, start)]
        yield PTNode.lead(nl_nxt)  # x* pattern can always match empty string
        nl_que = [nl_nxt]
        while nl_que:
            # Node List PREvious, which has already failed
            nl_pre = nl_que.pop(0)
            for n2 in self.pattern.match(text, nl_pre[-1].index1):
                if not n2:
                    # repeat expect it's sub pattern to proceed
                    continue
                nl_nxt = nl_pre + [n2]
                yield PTNode.lead(nl_nxt)
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


class PExample(Pattern):
    """A Pattern with example"""

    def __init__(self, pattern: Pattern, examples):
        self.pattern: Pattern = pattern
        self.examples: List[str] = [e for e in examples if e]

    @property
    def has_example(self) -> bool:
        return bool(self.examples)

    def __repr__(self):
        return '(' + repr(self.pattern) + ') **with example**'

    def match(self, text, start=0):
        # should not reach this line
        return self.pattern.match(text, start)


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
    def repeat(pattern, _from: int = 0, _to: int = None, greedy=True) -> Pattern:
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
