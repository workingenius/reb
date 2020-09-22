from reb import P, PTNode


def same(extraction, expected):
    """Assert that extraction result has the same effect with expected"""
    assert isinstance(extraction, list)
    assert isinstance(expected, list)
    assert len(extraction) == len(expected)

    for ext, exp in zip(extraction, expected):
        assert isinstance(ext, PTNode)

        if isinstance(exp, str):
            assert ext.content == exp
        elif isinstance(exp, PTNode):
            assert pt_for_user(ext) == pt_for_user(exp)


def pt_for_user(ptnode: PTNode) -> PTNode:
    def without_children(ptn):
        return PTNode(text=ptn.text, start=ptn.start(), end=ptn.end(), tag=ptn.tag)

    def node_with_tag(ptn):
        for n in ptn.children:
            if n.tag is not None:
                yield without_children(n)
            yield from node_with_tag(n)
    
    return [without_children(ptnode)] + sorted(node_with_tag(ptnode),
                                               key=lambda n: (n.start, n.end, n.tag))


def case(pattern, text, expect_pt):
    same(pattern.extractall(text, engine='plain'), expect_pt)
    same(pattern.extractall(text, engine='vm'), expect_pt)


def test_ptext():
    case(P.pattern('a'), 'a', ['a'])
    case(P.pattern('a'), 'aa', ['a', 'a'])
    case(P.pattern('a'), 'aba', ['a', 'a'])
    case(P.pattern('a'), 'b', [])


def test_panychar():
    case(P.ANYCHAR, 'a', ['a'])
    case(P.ANYCHAR, 'b', ['b'])
    case(P.ANYCHAR, ' ', [' '])
    case(P.ANYCHAR, '\n', ['\n'])
    case(P.ANYCHAR, '', [])
    case(P.ANYCHAR, 'abc', ['a', 'b', 'c'])


def test_pinchars():
    case(P.ic('bcd'), 'a', [])
    case(P.ic('abc'), 'a', ['a'])
    case(P.ic('abc'), 'b', ['b'])
    case(P.ic('abc'), 'c', ['c'])
    case(P.ic('abc'), 'abcdef', ['a', 'b', 'c'])


def test_pnotinchars():
    case(P.nic('bcd'), 'a', ['a'])
    case(P.nic('abc'), 'a', [])
    case(P.nic('abc'), 'b', [])
    case(P.nic('abc'), 'c', [])
    case(P.nic('abc'), 'abcdef', ['d', 'e', 'f'])


def test_pany():
    case(P.any('ab', 'abc', 'cd'), 'abcdef', ['ab', 'cd'])
    case(P.any('aa', 'ab', 'ac'), 'aaaaaa', ['aa', 'aa', 'aa'])


def test_prepeat():
    case(P.n('a'), 'aaa', ['aaa'])

    case(P.n('a', 4), 'a' * 3, [])
    case(P.n('a', 4), 'a' * 4, ['a' * 4])
    case(P.n('a', 4), 'a' * 5, ['a' * 5])
    case(P.n('a', 4), 'a' * 20, ['a' * 20])

    case(P.n('a', 0, 5), 'a' * 6, ['aaaaa', 'a'])
    case(P.n('a', 0, 5), 'a' * 10, ['aaaaa', 'aaaaa'])

    case(P.n('a', 2, 3), 'a' * 6, ['aaa', 'aaa'])
    case(P.n('a', 3, 5), 'a' * 9, ['aaaaa', 'aaaa'])
    case(P.n('a', exact=2), 'a' * 5, ['aa', 'aa'])

    case(P.n('a', greedy=False), 'aaa', [])


def test_padjacent():
    case(P.pattern('a') + P.ic('abcde'), 'ab', ['ab'])
    case(P.pattern('a') + P.ic('abcde'), 'ac', ['ac'])
    case(P.pattern('a') + P.ic('abcde'), 'ad', ['ad'])
    case(P.pattern('a') + P.ic('abcde'), 'af', [])
    case(P.pattern('a') + P.ic('abcde'), 'ba', [])

    case(P.ic('ab') + P.ic('cd') + P.ic('ef'), 'aacee', ['ace'])
    case(P.ic('ab') + P.ic('cd') + P.ic('ef'), 'abdfe', ['bdf'])
    case(P.ic('ab') + P.ic('cd') + P.ic('ef'), 'acdfe', [])
    case(P.ic('ab') + P.ic('cd') + P.ic('ef'), 'aaafe', [])
    case(P.ic('ab') + P.ic('cd') + P.ic('ef'), 'aacae', [])


def test_overall1():
    text = 'a' * 10 + 'b'
    case(P.tag(P.n('a'), tag='A') + 'b', text, [
        PTNode(text=text, start=0, end=11, children=[
            PTNode(text=text, start=0, end=10, tag='A')
        ])
    ])


def test_overall2():
    text = 'a' * 30 + 'c'
    case(P.n('a') + 'b', text, [])
