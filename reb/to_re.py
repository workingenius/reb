"""From a Pattern to a traditional re string"""


from functools import singledispatch
from re import escape as re_escape

from .pattern import Pattern, PText, PAnyChar, PNotInChars, PInChars, PAny, PTag, PRepeat, PAdjacent


@singledispatch
def to_re(pattern: Pattern) -> str:
    raise TypeError('The pattern can\'t rewrite in traditional re')


@to_re.register(PText)
def ptext_to_re(pattern: PText) -> str:
    return re_escape(pattern.text)


@to_re.register(PAnyChar)
def panychar_to_re(pattern: PAnyChar) -> str:
    return '.'


@to_re.register(PNotInChars)
def pnotinchars_to_re(pattern: PNotInChars) -> str:
    return '[^' + pattern.chars + ']'  # TODO consider "[", "]" and "^" in pattern.chars


@to_re.register(PInChars)
def pinchars_to_re(pattern: PInChars) -> str:
    return '[' + pattern.chars + ']'  # TODO consider "[" and "]" in pattern.chars


@to_re.register(PAny)
def pany_to_re(pattern: PAny) -> str:
    subs = [to_re(subp) for subp in pattern.patterns]
    return '|'.join([protect(s) for s in subs])


@to_re.register(PTag)
def ptag_to_re(pattern: PTag) -> str:
    return '(?P<{}>{})'.format(str(pattern.tag), to_re(pattern.pattern))


@to_re.register(PRepeat)
def prepeat_to_re(pattern: PRepeat) -> str:
    single = protect(to_re(pattern.pattern))
    whole = ''
    if pattern._from == 0 and pattern._to is None:
        whole = single + '*'
    elif pattern._from == 1 and pattern._to is None:
        whole = single + '+'
    elif pattern._from == 0 and pattern._to == 1:
        whole = single + '?'
    else:
        whole = single + '{%s,%s}' % (pattern._from, pattern._to or '')
    if not pattern.greedy:
        whole += '?'
    return whole


@to_re.register(PAdjacent)
def padjacent_to_re(pattern: PAdjacent) -> str:
    subs = [protect(to_re(p)) for p in pattern.patterns]
    return ''.join(subs)


def protect(restr: str) -> str:
    return '(?:' + restr + ')'
