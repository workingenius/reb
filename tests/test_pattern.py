import re

import pytest

from reb.pattern import P, PTNode, ExampleFail, InvalidPattern


def test_text_match1():
    pts = P.pattern('医院').extract('医院')
    assert pts[0].content == '医院'
    assert pts[0].start() == 0
    assert pts[0].end() == 2


def test_text_match2():
    pts = P.pattern('医院').extract('到某大医院就诊')
    assert pts[0].content == '医院'
    assert pts[0].start() == 3
    assert pts[0].end() == 5


def test_singlechar_match():
    digit = P.ic('0123456789')
    integer = P.n(digit, 1)

    pts = integer.extract('  129439  ')
    assert pts[0].content == '129439'
    assert pts[0].start() == 2
    assert pts[0].end() == 8


def test_anychar_match():
    ptn = P.ANYCHAR

    assert len(ptn.extract('a')) == 1
    assert len(ptn.extract('啊')) == 1
    assert len(ptn.extract('啊啊')) == 2
    assert not ptn.extract('')


def test_tag_match():
    ptn0 = P.ANYCHAR
    ptn1 = P.tag(P.ANYCHAR, tag='1')
    ptn2 = P.tag(P.ANYCHAR, tag='2')

    assert ptn0.extract('a')[0].tag is None
    assert ptn1.extract('a')[0].simplify().tag == '1'
    assert ptn2.extract('b')[0].simplify().tag == '2'


def test_in_chars():
    ptn = P.ic('abc')

    assert len(ptn.extract('a')) == 1
    assert len(ptn.extract('ab')) == 2
    assert len(ptn.extract('b')) == 1
    assert len(ptn.extract('c')) == 1
    assert len(ptn.extract('d')) == 0


def test_not_in_chars():
    ptn = P.nic('abc')

    assert len(ptn.extract('aaa')) == 0
    assert len(ptn.extract('aba')) == 0
    assert len(ptn.extract('abc')) == 0
    assert len(ptn.extract('add')) == 2
    assert len(ptn.extract('ant')) == 2
    assert len(ptn.extract('min')) == 3


def test_any_match():
    ptn = P.any(
        P.pattern('a'),
        P.pattern('b'),
        P.pattern('c'),
        P.pattern('d'),
    )

    assert len(ptn.extract('aaaaaaa')) == 7
    assert len(ptn.extract('aaaaabb')) == 7
    assert len(ptn.extract('aaadddd')) == 7
    assert len(ptn.extract('aaaaccc')) == 7
    assert len(ptn.extract('a a a a c c c')) == 7
    assert len(ptn.extract('azzzzzz')) == 1
    assert len(ptn.extract('bczzzzz')) == 2
    assert len(ptn.extract('llzzzzz')) == 0


def test_adjacent_match():
    ptn = P.ic('abc') + P.ic('123') + P.ic('xyz')

    assert len(ptn.extract('a1x b2z c3z')) == 3
    assert len(ptn.extract('a 1 x b 2 z c 3 z')) == 0
    assert len(ptn.extract('a10 b20 c30')) == 0


def test_repeat_match1():
    ptn = P.n('abc', greedy=True)

    assert len(ptn.extract('abc')) == 1
    assert len(ptn.extract('abcabc')) == 1
    assert len(ptn.extract('abcabcabc abc ab')) == 2
    assert len(ptn.extract('abd')) == 0


def test_repeat_match2():
    ptn = P.n('aba')
    assert len(ptn.extract('abababa')) == 2

    ptn2 = P.n('aaa')
    assert len(ptn2.extract('aaaaaaaaa')) == 1


def assert_extraction_eq(ex1, ex2):
    _ex2 = [pt.simplify() for pt in ex2]
    assert ex1 == _ex2


def test_prepeat_match3():
    ptn = P.n('a', 1, 3)
    assert_extraction_eq(ptn.extract(''), [])
    assert_extraction_eq(ptn.extract('a'), [
        PTNode('a', 0, 1, children=[
            PTNode('a', 0, 1)
        ])
    ])
    assert_extraction_eq(ptn.extract('aa'), [
        PTNode('aa', 0, 2, children=[
            PTNode('aa', 0, 1),
            PTNode('aa', 1, 2)
        ])
    ])
    assert_extraction_eq(ptn.extract('aaa'), [
        PTNode('aaa', 0, 3, children=[
            PTNode('aaa', 0, 1),
            PTNode('aaa', 1, 2),
            PTNode('aaa', 2, 3)
        ])
    ])
    assert_extraction_eq(ptn.extract('aaaa'), [
        PTNode('aaaa', 0, 3, children=[
            PTNode('aaaa', 0, 1),
            PTNode('aaaa', 1, 2),
            PTNode('aaaa', 2, 3)
        ]),
        PTNode('aaaa', 3, 4, children=[
            PTNode('aaaa', 3, 4)
        ])
    ])


def test_repeat_match4():
    ptn = P.n('a') + P.n('b') + P.n('c')
    # should not contain substring with zero length
    assert not ptn.extract('def')


def test_repeat_match5():
    ptn = P.n('a', greedy=False) + P.n('b')
    text = 'aabb'
    assert_extraction_eq(ptn.extract(text), [
        PTNode(text, 2, 4, children=[
            PTNode(text, 2, 4, children=[
                PTNode(text, 2, 3),
                PTNode(text, 3, 4),
            ]),
        ]),
    ])


def test_repeat_match6():
    ptn = P.n('a', 3, 3)
    assert ptn.extract('') == []

    text = 'aaa'
    assert_extraction_eq(ptn.extract(text), [PTNode(text, 0, 3, children=[
        PTNode(text, 0, 1),
        PTNode(text, 1, 2),
        PTNode(text, 2, 3),
    ])])

    text = 'aaaaa'
    assert_extraction_eq(ptn.extract(text), [PTNode(text, 0, 3, children=[
        PTNode(text, 0, 1),
        PTNode(text, 1, 2),
        PTNode(text, 2, 3),
    ])])


def test_repeat_7():
    p0 = P.n(P.ANYCHAR)
    assert p0._from == 0
    assert p0._to == None

    p1 = P.n(P.ANYCHAR, 1)
    assert p1._from == 1
    assert p1._to == None

    p2 = P.n(P.ANYCHAR, 1, 3)
    assert p2._from == 1
    assert p2._to == 3

    p3 = P.n(P.ANYCHAR, exact=4)
    assert p3._from == 4
    assert p3._to == 4

    with pytest.raises(InvalidPattern):
        P.n(P.ANYCHAR, 6, 1)


def test_dropped():
    """PTNodes make by helper patterns should not be in the parse tree"""

    ptn = P.n('aba')

    text = 'abababa'
    assert_extraction_eq(ptn.extract(text), [
        PTNode(text, 0, 3, children=[
            PTNode(text, 0, 3)
        ]),
        PTNode(text, 4, 7, children=[
            PTNode(text, 4, 7)
        ])
    ])

    text2 = 'a1x2'
    ptn2 = P.ic('abc') + P.ic('123') + P.ic('xyz') + P.ic('123')
    assert_extraction_eq(ptn2.extract(text2), [
        PTNode(text2, 0, 4, children=[
            PTNode(text2, 0, 1),
            PTNode(text2, 1, 2),
            PTNode(text2, 2, 3),
            PTNode(text2, 3, 4),
        ])
    ])


def test_match_recursion_max_limit():
    ptn = P.n('a', greedy=True) + P.n('b')
    text = 'a' * 2000 + 'b' * 2000
    # should not raise RecursionError
    assert ptn.extract(text)


def test_repeat_zero_spans():
    ptn = P.n(P.n('a'))

    assert_extraction_eq(ptn.extract(''), [])
    assert_extraction_eq(ptn.extract('a'), [PTNode('a', 0, 1, children=[
        PTNode('a', 0, 1, children=[PTNode('a', 0, 1)])])])
    assert_extraction_eq(ptn.extract('b'), [])
    assert_extraction_eq(ptn.extract('bba'), [
        PTNode('bba', 2, 3, children=[
            PTNode('bba', 2, 3, children=[
                PTNode('bba', 2, 3)
            ])
        ])
    ])
    assert_extraction_eq(ptn.extract('bbaaa'), [
        PTNode('bbaaa', 2, 5, children=[
            PTNode('bbaaa', 2, 3, children=[
                PTNode('bbaaa', 2, 3)
            ]),
            PTNode('bbaaa', 3, 4, children=[
                PTNode('bbaaa', 3, 4)
            ]),
            PTNode('bbaaa', 4, 5, children=[
                PTNode('bbaaa', 4, 5)
            ]),
        ])
    ])


def test_pstarting1():
    ptn = P.STARTING + 'abc'
    assert_extraction_eq(ptn.extract('abc'), [
        PTNode('abc', 0, 3, children=[
            PTNode('abc', 0, 0),
            PTNode('abc', 0, 3),
        ])
    ])
    assert ptn.extract(' abc') == []


def test_pending1():
    ptn = 'abc' + P.ENDING
    assert_extraction_eq(ptn.extract('abc'), [
        PTNode('abc', 0, 3, children=[
            PTNode('abc', 0, 3),
            PTNode('abc', 3, 3),
        ])
    ])
    assert_extraction_eq(ptn.extract('abc '), [])


def test_pnewline1():
    ptn = 'abc' + P.NEWLINE + 'abc'
    assert_extraction_eq(ptn.extract('abcabc'), [])

    text = 'abc\nabc'
    assert_extraction_eq(ptn.extract(text), [
        PTNode(text, 0, 7, children=[
            PTNode(text, 0, 3),
            PTNode(text, 3, 4),
            PTNode(text, 4, 7),
        ])
    ])

    text = 'abc\r\nabc'
    assert_extraction_eq(ptn.extract(text), [
        PTNode(text, 0, 8, children=[
            PTNode(text, 0, 3),
            PTNode(text, 3, 5),
            PTNode(text, 5, 8),
        ])
    ])


class TestRebBehaviourSameAsRE(object):
    def ensure_behaviour_same(self, pattern, text: str):
        l0 = list(re.compile(pattern.re).finditer(text))
        l1 = list(pattern.finditer(text))
        l0 = [m for m in l0 if m.end() - m.start()]
        l1 = [m for m in l1 if m.end() - m.start()]
        assert len(l0) == len(l1)
        for m0, n1 in zip(l0, l1):
            assert m0.start() == n1.start()
            assert m0.end() == n1.end()

    def test_ptext1(self):
        ptn = P.pattern('abc')
        self.ensure_behaviour_same(ptn, '  bca   ')
        self.ensure_behaviour_same(ptn, '  abc   ')
        self.ensure_behaviour_same(ptn, '  abc   abc  ')

    def test_panychar1(self):
        ptn = P.ANYCHAR
        self.ensure_behaviour_same(ptn, 'c')
        self.ensure_behaviour_same(ptn, '')
        self.ensure_behaviour_same(ptn, 'abcd  ')  # TODO Consider '\n'

    def test_pnotinchars1(self):
        ptn = P.nic('abcd')
        self.ensure_behaviour_same(ptn, 'abcd')
        self.ensure_behaviour_same(ptn, 'abce')
        self.ensure_behaviour_same(ptn, 'eeee')

    def test_pinchars1(self):
        ptn = P.ic('abcd')
        self.ensure_behaviour_same(ptn, 'abcd')
        self.ensure_behaviour_same(ptn, 'abee')
        self.ensure_behaviour_same(ptn, 'eeee')

    def test_pany1(self):
        ptn = P.any(
            'convention',
            'insertion',
            'creation',
            'modification',
            'sensation',
            'convertion',
        )

        self.ensure_behaviour_same(ptn, '')
        self.ensure_behaviour_same(ptn, 'abc')
        self.ensure_behaviour_same(ptn, ' creation')
        self.ensure_behaviour_same(ptn, ' creation satification convention')

    def test_prepeat1(self):
        ptn = P.n('a')
        self.ensure_behaviour_same(ptn, '')
        self.ensure_behaviour_same(ptn, 'a')
        self.ensure_behaviour_same(ptn, 'aa')
        self.ensure_behaviour_same(ptn, 'aaa')
        self.ensure_behaviour_same(ptn, ' aaa ')
        self.ensure_behaviour_same(ptn, ' ababa ')
        self.ensure_behaviour_same(ptn, ' ababa a')

    def test_prepeat2(self):
        ptn = P.n('a', 1)
        self.ensure_behaviour_same(ptn, '')
        self.ensure_behaviour_same(ptn, 'a')
        self.ensure_behaviour_same(ptn, 'aa')
        self.ensure_behaviour_same(ptn, 'aaa')
        self.ensure_behaviour_same(ptn, ' aaa ')
        self.ensure_behaviour_same(ptn, ' ababa ')
        self.ensure_behaviour_same(ptn, ' ababa a')

    def test_prepeat3(self):
        ptn = P.n('a', 0, 1)
        self.ensure_behaviour_same(ptn, '')
        self.ensure_behaviour_same(ptn, 'a')
        self.ensure_behaviour_same(ptn, 'aa')
        self.ensure_behaviour_same(ptn, 'aaa')
        self.ensure_behaviour_same(ptn, ' aaa ')
        self.ensure_behaviour_same(ptn, ' ababa ')
        self.ensure_behaviour_same(ptn, ' ababa a')

    def test_prepeat4(self):
        ptn = P.n('a', 1, 3)
        self.ensure_behaviour_same(ptn, '')
        self.ensure_behaviour_same(ptn, 'a')
        self.ensure_behaviour_same(ptn, 'aa')
        self.ensure_behaviour_same(ptn, 'aaa')
        self.ensure_behaviour_same(ptn, 'aaaa')
        self.ensure_behaviour_same(ptn, ' aaaa ')
        self.ensure_behaviour_same(ptn, ' aaa ')
        self.ensure_behaviour_same(ptn, ' ababa ')
        self.ensure_behaviour_same(ptn, ' ababa a')

    def test_padjacent1(self):
        ptn = P.n(P.ANYCHAR + '=' + P.n(' ') + P.n(P.ANYCHAR) + ';', 1)
        self.ensure_behaviour_same(ptn, '')
        self.ensure_behaviour_same(ptn, 'y= 2;')
        # following cases do not pass, as search order differs.
        # self.ensure_behaviour_same(ptn, 'x=1; y= 2;')
        # self.ensure_behaviour_same(ptn, 'x=1; y= 2; zzz==xxx;;')


class TestExample(object):
    def test_example1(self):
        ptn0 = P.pattern('abc')
        ptn = P.example(
            ptn0,
            'abc',
            ' abc '
        )

        assert ptn.extract('abc') == ptn0.extract('abc')

    def test_example2(self):
        ptn0 = P.pattern('def')

        with pytest.raises(ExampleFail):
            P.example(
                ptn0,
                'alala'
            )

    def test_example_conflict1(self):
        ptn1 = P.example(P.pattern('def'), 'def')
        ptn2 = P.example(P.pattern('de'), 'de')

        with pytest.raises(ExampleFail):
            ptn2 | ptn1
