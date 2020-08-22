from reb import P, PTNode


def test_text_match1():
    pts = P.pattern('医院').extract('医院')
    assert pts[0].content == '医院'
    assert pts[0].start == 0
    assert pts[0].end == 2


def test_text_match2():
    pts = P.pattern('医院').extract('到某大医院就诊')
    assert pts[0].content == '医院'
    assert pts[0].start == 3
    assert pts[0].end == 5


def test_singlechar_match():
    digit = P.ic('0123456789')
    integer = P.n(digit, 1)

    pts = integer.extract('  129439  ')
    assert pts[0].content == '129439'
    assert pts[0].start == 2
    assert pts[0].end == 8


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
    assert ptn1.extract('a')[0].tag == '1'
    assert ptn2.extract('b')[0].tag == '2'


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


def test_dropped():
    """PTNodes make by helper patterns should not be in the parse tree"""

    ptn = P.n('aba')
    assert ptn.extract('abababa')[0] == PTNode('abababa', start=0, end=3, children=[])
    assert ptn.extract('abababa')[1] == PTNode('abababa', start=4, end=7, children=[])

    text2 = 'a1x2'
    ptn2 = P.ic('abc') + P.ic('123') + P.ic('xyz') + P.ic('123')
    assert ptn2.extract(text2)[0] == PTNode(text2, start=0, end=4, children=[
        PTNode(text2, start=0, end=1),
        PTNode(text2, start=1, end=2),
        PTNode(text2, start=2, end=3),
        PTNode(text2, start=3, end=4),
    ])


def test_repeat_match3():
    ptn = P.n('a') + P.n('b') + P.n('c')
    # should not contain substring with zero length
    assert not ptn.extract('def')


def test_repeat_match4():
    ptn = P.n('a', greedy=False) + P.n('b')
    text = 'aabb'
    assert ptn.extract(text) == [
        PTNode(text, 0, 1, children=[
            PTNode(text, 0, 1, children=[]),
            PTNode(text, 1, 1, children=[]),
        ]),
        PTNode(text, 1, 4, children=[
            PTNode(text, 1, 2, children=[]),
            PTNode(text, 2, 4, children=[])
        ])
    ]


def test_match_recursion_max_limit():
    ptn = P.n('a', greedy=True) + P.n('b')
    text = 'a' * 2000 + 'b' * 2000
    # should not raise RecursionError
    assert ptn.extract(text)


def test_repeat_zero_spans():
    ptn = P.n(P.n('a'))

    assert ptn.extract('') == []
    assert ptn.extract('a') == [PTNode('a', 0, 1)]
    assert ptn.extract('b') == []
    assert ptn.extract('bba') == [PTNode('bba', 2, 3)]
    assert ptn.extract('bbaaa') == [PTNode('bbaaa', 2, 5)]