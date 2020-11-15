from reb.parse_tree import PTNode, VirtualPTNode


def test_parse_tree():
    text = 'abcdef'

    node = PTNode(text, 0, 3, children=[
        VirtualPTNode(text, 0, 2, children=[
            PTNode(text, 0, 1),
            PTNode(text, 1, 2),
        ]),
    ])

    assert node.children == [
        PTNode(text, 0, 1),
        PTNode(text, 1, 2)
    ]


def test_parse_tree2():
    """Empty nodes are hidden"""
    text = 'abcdef'

    node = PTNode(text, 0, 3, children=[
        VirtualPTNode(text, 0, 2, children=[
            PTNode(text, 0, 1),
            PTNode(text, 1, 1),
            PTNode(text, 1, 2),
        ]),
    ])

    assert node.children == [
        PTNode(text, 0, 1),
        PTNode(text, 1, 2)
    ]


def test_parse_tree3():
    text = 'abcdef'

    node = PTNode(text, 0, 5, children=[
        PTNode(text, 0, 3, children=[
            VirtualPTNode(text, 0, 2, children=[
                PTNode(text, 0, 1),
                PTNode(text, 1, 2),
            ])
        ]),
    ])

    assert node.children == [
        PTNode(text, 0, 3, children=[
            PTNode(text, 0, 1),
            PTNode(text, 1, 2),
        ]),
    ]


def assert_simplify_eq(pt, simplified):
    spt = pt.simplify()
    assert spt == simplified
    assert spt == spt.simplify()


def test_parse_tree_simplify1():
    """If a ptnode does not have children, it is simplified"""
    text = 'abcdef'

    assert_simplify_eq(PTNode(text, 0, 5), PTNode(text, 0, 5))
    assert_simplify_eq(PTNode(text, 0, 3), PTNode(text, 0, 3))
    assert_simplify_eq(PTNode(text, 0, 0), PTNode(text, 0, 0))


def test_parse_tree_simplify2():
    """Unnecessary sub nodes (without any tag) should be removed"""
    text = 'abcdef'

    assert_simplify_eq(
        PTNode(text, 0, 5, children=[
            PTNode(text, 0, 3)
        ]),
        PTNode(text, 0, 5)
    )

    assert_simplify_eq(
        PTNode(text, 0, 0, children=[
            PTNode(text, 0, 0, children=[
                PTNode(text, 0, 0)
            ])
        ]),
        PTNode(text, 0, 0)
    )


def test_parse_tree_simplify3():
    """Sub nodes with tag should be preserved"""
    text = 'abcdef'

    assert_simplify_eq(
        PTNode(text, 0, 6, children=[
            PTNode(text, 0, 1),
            PTNode(text, 1, 2, tag='t1'),
            PTNode(text, 2, 3, tag='t2'),
            PTNode(text, 3, 4),
            PTNode(text, 4, 5, tag='t3')
        ]),
        PTNode(text, 0, 6, children=[
            PTNode(text, 1, 2, tag='t1'),
            PTNode(text, 2, 3, tag='t2'),
            PTNode(text, 4, 5, tag='t3')
        ])
    )


def test_parse_tree_simplify4():
    """Subnode with tag should be "promoted" if possible"""
    text = 'abcdef'

    assert_simplify_eq(
        PTNode(text, 0, 6, children=[
            PTNode(text, 0, 5, children=[
                PTNode(text, 0, 4, children=[
                    PTNode(text, 0, 3, tag='t')
                ])
            ])
        ]),
        PTNode(text, 0, 6, children=[
            PTNode(text, 0, 3, tag='t')
        ])
    )

    assert_simplify_eq(
        PTNode(text, 0, 6, children=[
            PTNode(text, 0, 5, children=[
                PTNode(text, 0, 4, children=[
                    PTNode(text, 0, 3, tag='t1', children=[
                        PTNode(text, 0, 2, children=[
                            PTNode(text, 0, 1, tag='t2')
                        ])
                    ])
                ])
            ])
        ]),
        PTNode(text, 0, 6, children=[
            PTNode(text, 0, 3, tag='t1', children=[
                PTNode(text, 0, 1, tag='t2')
            ])
        ])
    )

    assert_simplify_eq(
        PTNode(text, 0, 6, children=[
            PTNode(text, 0, 5, children=[
                PTNode(text, 0, 4, children=[
                    PTNode(text, 0, 3, tag='t1', children=[
                        PTNode(text, 0, 2, children=[
                            PTNode(text, 0, 1, tag='t2')
                        ]),
                        PTNode(text, 2, 3, tag='t3')
                    ])
                ])
            ])
        ]),
        PTNode(text, 0, 6, children=[
            PTNode(text, 0, 3, tag='t1', children=[
                PTNode(text, 0, 1, tag='t2'),
                PTNode(text, 2, 3, tag='t3'),
            ])
        ])
    )


def test_parse_tree_simplify5():
    """Subnode promote to replace the root"""
    text = 'abcdef'

    assert_simplify_eq(
        PTNode(text, 0, 5, children=[
            PTNode(text, 0, 5, children=[
                PTNode(text, 0, 5, tag='t1')
            ])
        ]),
        PTNode(text, 0, 5, tag='t1')
    )
