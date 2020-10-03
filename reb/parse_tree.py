from typing import List, Optional


__all__ = [
    'PTNode', 'VirtualPTNode'
]


class PTNode(object):
    """Parse Tree Node"""

    def __init__(self, text: str, start: int, end: int, children: List['PTNode'] = [], tag=None):
        self.text: str = text
        assert end >= start >= 0
        self.index0: int = start
        self.index1: int = end
        self._children: List['PTNode'] = children
        self.tag = tag

    @property
    def string(self) -> str:
        return self.text

    @property
    def content(self) -> str:
        return self.text[self.index0: self.index1]

    # start(), end() as methods, simulating re MathcObject behaviour
    def start(self):
        return self.index0

    def end(self):
        return self.index1

    @property
    def children(self):
        return node_children(self)

    def __repr__(self):
        c = '{}, {}, content={}'.format(self.index0, self.index1, repr(self.content))
        if self.tag:
            c += ', tag={}'.format(repr(self.tag))
        elif self.children:
            c += ', children=[{}]'.format(', '.join([repr(n) for n in self.children]))
        return '{}('.format(self.__class__.__name__) + c + ')'

    def __bool__(self):
        return self.index1 > self.index0

    @classmethod
    def lead(cls, pts: List['PTNode']) -> 'PTNode':
        """Make a new PTNode as the common parent of nodes <pts> """
        
        for p1, p2 in zip(pts[:-1], pts[1:]):
            assert p1.index1 == p2.index0
            
        pt = cls(pts[0].text, pts[0].index0, pts[-1].index1, children=list(pts))
        return pt

    def fetch(self, tag):
        """Fetch those nodes whose tag == <tag>"""
        if self.tag == tag:
            yield self
        if self.children:
            for n in self.children:
                for nn in n.fetch(tag):
                    yield nn

    def drop(self) -> 'PTNode':
        """Copy the PTNode but without children"""
        return self.__class__(self.text, self.index0, self.index1, tag=self.tag)

    def __eq__(self, o):
        if isinstance(o, PTNode):
            return self.text == o.text \
                and self.index0 == o.index0 \
                and self.index1 == o.index1 \
                and self.children == o.children \
                and self.tag == o.tag
        return False

    def pp(self):
        """Pretty Print in terminals, designed for terminal users"""
        pretty_print_tree(self)

    def show(self):
        """Show tree structure in detail, designed for pdb debugging"""
        show_tree(self)


class VirtualPTNode(PTNode):
    """A class of nodes that is transparent to callers"""


def node_children(n: PTNode):
    return list(iter_node_children(n))


def iter_node_children(n: PTNode):
    for c in n._children:
        if not isinstance(c, VirtualPTNode):
            if c:
                yield c
        else:
            for cc in iter_node_children(c):
                yield cc


def show_tree(tree: PTNode):
    
    def _show_tree(node: PTNode, depth=0) -> str:
        assert isinstance(node, PTNode)
        mark = '-' if isinstance(node, VirtualPTNode) else '+'
        text = '{}{} ({}, {})'.format('  ' * depth, mark, node.start(), node.end())
        subs = [_show_tree(sub, depth+1) for sub in node._children]
        return '\n'.join([text] + subs)

    print(_show_tree(tree))


def pretty_print_tree(tree: PTNode):
    try:
        from termcolor import colored
    except ImportError:
        print('Module termcolor is needed')

    else:
        # for i in tag_lst, tag_lst[i] is the tag most close to the leaf
        tag_lst = [None] * (tree.index1 - tree.index0)

        start0 = tree.index0

        def set_tag(node, tl):
            """Traverse the parse tree and set tag_lst"""
            if node.tag is not None:
                for i in range(node.index0 - start0, node.index1 - start0):
                    tl[i] = node.tag
            for cn in node.children:
                set_tag(cn, tl)

        set_tag(tree, tag_lst)

        colors = ['red', 'green', 'yellow', 'blue', 'magenta']
        white = 'white'

        def tag_color(tag):
            if tag is None:
                return white
            return colors[sum(map(ord, tag)) % len(colors)]

        color_lst = [tag_color(tag) for tag in tag_lst]

        # extend several chars on both sides
        extend_n = 10
        left_i = max(0, tree.index0 - extend_n)
        right_i = min(len(tree.text), tree.index1 + extend_n)

        left_str = tree.text[left_i: tree.index0]
        left_str = ('...' + left_str) if left_i > 0 else left_str

        right_str = tree.text[tree.index1: right_i]
        right_str = (right_str + '...') if right_i < len(tree.text) else right_str

        # whole string to print
        ws = colored(left_str, attrs=['dark'])
        for offset, color in enumerate(color_lst):
            i = tree.index0 + offset
            ws += colored(tree.text[i], color)
        ws += colored(right_str, attrs=['dark'])

        print(ws)
