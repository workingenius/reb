from pattern2 import P, PTNode


def test_text_match1():
    pts = P.pattern('医院').search('医院')
    assert pts[0].content == '医院'
    assert pts[0].start == 0
    assert pts[0].end == 2


def test_text_match2():
    pts = P.pattern('医院').search('到某大医院就诊')
    assert pts[0].content == '医院'
    assert pts[0].start == 3
    assert pts[0].end == 5


def test_singlechar_match():
    digit = P.anyc('0123456789')
    integer = P.n(digit, 1)

    pts = integer.search('  129439  ')
    assert pts[0].content == '129439'
    assert pts[0].start == 2
    assert pts[0].end == 8


def test_anychar_match():
    ptn = P.ANYCHAR

    assert len(ptn.search('a')) == 1
    assert len(ptn.search('啊')) == 1
    assert len(ptn.search('啊啊')) == 2
    assert not ptn.search('')


def test_tag_match():
    ptn0 = P.ANYCHAR
    ptn1 = P.tag(P.ANYCHAR, tag='1')
    ptn2 = P.tag(P.ANYCHAR, tag='2')

    assert ptn0.search('a')[0].tag is None
    assert ptn1.search('a')[0].tag == '1'
    assert ptn2.search('b')[0].tag == '2'


def test_in_chars():
    ptn = P.ic('abc')

    assert len(ptn.search('a')) == 1
    assert len(ptn.search('ab')) == 2
    assert len(ptn.search('b')) == 1
    assert len(ptn.search('c')) == 1
    assert len(ptn.search('d')) == 0


def test_not_in_chars():
    ptn = P.nic('abc')

    assert len(ptn.search('aaa')) == 0
    assert len(ptn.search('aba')) == 0
    assert len(ptn.search('abc')) == 0
    assert len(ptn.search('add')) == 2
    assert len(ptn.search('ant')) == 2
    assert len(ptn.search('min')) == 3


def test_any_match():
    ptn = P.any(
        P.pattern('a'),
        P.pattern('b'),
        P.pattern('c'),
        P.pattern('d'),
    )

    assert len(ptn.search('aaaaaaa')) == 7
    assert len(ptn.search('aaaaabb')) == 7
    assert len(ptn.search('aaadddd')) == 7
    assert len(ptn.search('aaaaccc')) == 7
    assert len(ptn.search('a a a a c c c')) == 7
    assert len(ptn.search('azzzzzz')) == 1
    assert len(ptn.search('bczzzzz')) == 2
    assert len(ptn.search('llzzzzz')) == 0


def test_adjacent_match():
    ptn = P.ic('abc') + P.ic('123') + P.ic('xyz')

    assert len(ptn.search('a1x b2z c3z')) == 3
    assert len(ptn.search('a 1 x b 2 z c 3 z')) == 0
    assert len(ptn.search('a10 b20 c30')) == 0


def test_repeat_match1():
    ptn = P.n('abc', greedy=True)

    assert len(ptn.search('abc')) == 1
    assert len(ptn.search('abcabc')) == 1
    assert len(ptn.search('abcabcabc abc ab')) == 2
    assert len(ptn.search('abd')) == 0


def test_repeat_match2():
    ptn = P.n('aba')
    assert len(ptn.search('abababa')) == 2

    ptn2 = P.n('aaa')
    assert len(ptn2.search('aaaaaaaaa')) == 1


def test_dropped():
    """PTNodes make by helper patterns should not be in the parse tree"""

    ptn = P.n('aba')
    assert ptn.search('abababa')[0] == PTNode('abababa', start=0, end=3, children=[])
    assert ptn.search('abababa')[1] == PTNode('abababa', start=4, end=7, children=[])

    text2 = 'a1x b2z c3z'
    ptn2 = P.ic('abc') + P.ic('123') + P.ic('xyz')
    assert ptn2.search(text2)[0] == PTNode(text2, start=0, end=3, children=[])
    assert ptn2.search(text2)[1] == PTNode(text2, start=4, end=7, children=[])
    assert ptn2.search(text2)[2] == PTNode(text2, start=8, end=11, children=[])
