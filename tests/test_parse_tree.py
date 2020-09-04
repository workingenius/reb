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
