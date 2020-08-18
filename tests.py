from pattern2 import P


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
