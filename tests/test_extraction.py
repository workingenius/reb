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


class ExtractionTestCases(object):
    def test_ptext(self):
        self.case(P.pattern('a'), 'a', ['a'])
        self.case(P.pattern('a'), 'aa', ['a', 'a'])
        self.case(P.pattern('a'), 'aba', ['a', 'a'])
        self.case(P.pattern('a'), 'b', [])

    def test_panychar(self):
        self.case(P.ANYCHAR, 'a', ['a'])
        self.case(P.ANYCHAR, 'b', ['b'])
        self.case(P.ANYCHAR, ' ', [' '])
        self.case(P.ANYCHAR, '\n', ['\n'])
        self.case(P.ANYCHAR, '', [])
        self.case(P.ANYCHAR, 'abc', ['a', 'b', 'c'])

    def test_pinchars(self):
        self.case(P.ic('bcd'), 'a', [])
        self.case(P.ic('abc'), 'a', ['a'])
        self.case(P.ic('abc'), 'b', ['b'])
        self.case(P.ic('abc'), 'c', ['c'])
        self.case(P.ic('abc'), 'abcdef', ['a', 'b', 'c'])

    def test_pnotinchars(self):
        self.case(P.nic('bcd'), 'a', ['a'])
        self.case(P.nic('abc'), 'a', [])
        self.case(P.nic('abc'), 'b', [])
        self.case(P.nic('abc'), 'c', [])
        self.case(P.nic('abc'), 'abcdef', ['d', 'e', 'f'])

    def test_pany(self):
        self.case(P.any('ab', 'abc', 'cd'), 'abcdef', ['ab', 'cd'])
        self.case(P.any('aa', 'ab', 'ac'), 'aaaaaa', ['aa', 'aa', 'aa'])

    def test_prepeat(self):
        self.case(P.n('a'), 'aaa', ['aaa'])

        self.case(P.n('a', 0, 1), 'aaa', ['a', 'a', 'a'])
        self.case(P.n('a', 0, 1), '', [])

        self.case(P.n('a', 4), 'a' * 3, [])
        self.case(P.n('a', 4), 'a' * 4, ['a' * 4])
        self.case(P.n('a', 4), 'a' * 5, ['a' * 5])
        self.case(P.n('a', 4), 'a' * 20, ['a' * 20])

        self.case(P.n('a', 0, 5), 'a' * 6, ['aaaaa', 'a'])
        self.case(P.n('a', 0, 5), 'a' * 10, ['aaaaa', 'aaaaa'])

        self.case(P.n('a', 2, 3), 'a' * 6, ['aaa', 'aaa'])
        self.case(P.n('a', 3, 5), 'a' * 9, ['aaaaa', 'aaaa'])
        self.case(P.n('a', exact=2), 'a' * 5, ['aa', 'aa'])

        self.case(P.n('a', greedy=False), 'aaa', [])

    def test_padjacent(self):
        self.case(P.pattern('a') + P.ic('abcde'), 'ab', ['ab'])
        self.case(P.pattern('a') + P.ic('abcde'), 'ac', ['ac'])
        self.case(P.pattern('a') + P.ic('abcde'), 'ad', ['ad'])
        self.case(P.pattern('a') + P.ic('abcde'), 'af', [])
        self.case(P.pattern('a') + P.ic('abcde'), 'ba', [])

        self.case(P.ic('ab') + P.ic('cd') + P.ic('ef'), 'aacee', ['ace'])
        self.case(P.ic('ab') + P.ic('cd') + P.ic('ef'), 'abdfe', ['bdf'])
        self.case(P.ic('ab') + P.ic('cd') + P.ic('ef'), 'acdfe', [])
        self.case(P.ic('ab') + P.ic('cd') + P.ic('ef'), 'aaafe', [])
        self.case(P.ic('ab') + P.ic('cd') + P.ic('ef'), 'aacae', [])

    def test_pstarting(self):
        self.case(P.STARTING + 'a', 'aaa', [PTNode('aaa', start=0, end=1)])
        self.case(P.STARTING + 'a', 'baa', [])

    def test_pending(self):
        self.case('a' + P.ENDING, 'aaa', [PTNode('aaa', start=2, end=3)])
        self.case('a' + P.ENDING, 'aab', [])

    def test_overall1(self):
        text = 'a' * 10 + 'b'
        self.case(P.tag(P.n('a'), tag='A') + 'b', text, [
            PTNode(text=text, start=0, end=11, children=[
                PTNode(text=text, start=0, end=10, tag='A')
            ])
        ])

    def test_overall2(self):
        text = 'a' * 30 + 'c'
        self.case(P.n('a') + 'b', text, [])

    def test_overall3(self):
        text = 'a' * 6
        self.case(P.n(P.n('a', exact=3)), text, ['a' * 6])

        text = 'a' * 8
        self.case(P.n(P.n('a', exact=3)), text, ['a' * 6])


class TestExtractionPlain(ExtractionTestCases):
    def case(self, pattern, text, expect_pt):
        same(pattern.extractall(text, engine='plain'), expect_pt)


class TestExtractionVM(ExtractionTestCases):
    def case(self, pattern, text, expect_pt):
        same(pattern.extractall(text, engine='vm'), expect_pt)
