from itertools import permutations
from typing import List, Optional, Iterator

from .parse_tree import PTNode, VirtualPTNode
from .exceptions import RebException, InvalidPattern, ExampleFail


DEFAULT_ENGINE = 'plain'


class Finder(object):
    def finditer(self, text: str) -> Iterator[PTNode]:
        raise NotImplementedError


class Pattern(object):
    def __init__(self):
        self.finders = {}

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

    def extract(self, text: str, engine=DEFAULT_ENGINE) -> List[PTNode]:
        """Extract info from text by the pattern, and return every match, forming a parse tree"""
        return list(self.extractiter(text, engine=engine))

    extractall = extract

    def extractiter(self, text: str, engine=DEFAULT_ENGINE) -> Iterator[PTNode]:
        """Extract info from text by the pattern, and return every match, forming parse trees"""
        for n in self.finditer(text, engine=engine):
            if n:
                yield n

    def finditer(self, text: str, engine=DEFAULT_ENGINE) -> Iterator[PTNode]:
        """Find pattern in text, yield them one after another"""
        finder: Finder = self.finders.get(engine)
        if not finder:
            if engine == 'plain':
                from .plain import compile_pattern as _compile_pattern
                compile_pattern = _compile_pattern
            elif engine == 'vm':
                from .vm import compile_pattern as vm
                compile_pattern = vm
            else:
                raise ValueError('Invalid Engine')
            finder = compile_pattern(self)
            self.finders[engine] = finder
        return finder.finditer(text)

    def findall(self, text: str, engine=DEFAULT_ENGINE) -> List[str]:
        sl = []
        for n in self.finditer(text, engine=engine):
            sl.append(n.string)
        return sl

    def pformat(self):
        return str(self)

    def pp(self):
        print(self.pformat())

    @property
    def re(self):
        # circular dependency
        from .to_re import to_re
        return to_re(self)


class PText(Pattern):
    """A plain pattern that just match text as it is"""

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def __repr__(self):
        return repr(self.text)


class PAnyChar(Pattern):
    """A pattern that match any character"""

    def __repr__(self):
        return '.'


class PTag(Pattern):
    def __init__(self, pattern, tag):
        super().__init__()
        self.pattern = pattern
        assert tag is not None
        self.tag = tag

    def __repr__(self):
        return '(?<{}>:{})'.format(self.tag, self.pattern)


class PNotInChars(Pattern):
    def __init__(self, chars: str):
        super().__init__()
        self.chars: str = chars

    def __repr__(self):
        return '[^{}]'.format(self.chars)


class PInChars(Pattern):
    def __init__(self, chars: str):
        super().__init__()
        self.chars: str = chars

    def __repr__(self):
        return '[{}]'.format(self.chars)


class PAny(Pattern):
    def __init__(self, patterns):
        super().__init__()
        self.patterns: List[Pattern] = patterns

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
        super().__init__()
        self.pattern: Pattern = pattern
        if _to is not None:
            assert _to >= _from
        self._from: int = _from
        self._to: Optional[int] = _to
        self.greedy: bool = greedy

    def __repr__(self):
        to = self._to if isinstance(self._to, int) else ''
        return '(%s){%s,%s}' % (self.pattern, self._from, to)


class PAdjacent(Pattern):
    def __init__(self, patterns: List[Pattern]):
        super().__init__()
        assert patterns
        assert len(patterns) >= 1
        self.patterns = patterns

    def __add__(self, pattern) -> Pattern:
        return PAdjacent(list(self.patterns) + [self.make(pattern)])

    def __radd__(self, pattern) -> Pattern:
        return PAdjacent([self.make(pattern)] + list(self.patterns))

    def __repr__(self):
        return ''.join('({})'.format(p) for p in self.patterns)


# Several helper Patterns

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
        super().__init__()
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
    pass


class PEnding(Pattern):
    pass


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
