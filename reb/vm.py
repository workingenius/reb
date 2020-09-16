"""Implement Regular Expression with vm"""


from typing import Iterator, List

from .parse_tree import PTNode
from .pattern import Pattern


class Instruction(object):
    """Instruction of the VM"""


class InsStart(Instruction):
    """Start a re match"""


class InsSuccess(Instruction):
    """A match succeed in current thread"""


class InsCompare(Instruction):
    """Step to next instruction only if current character equals <char>, otherwise current thread fails"""
    def __init__(self, char: str):
        self.char: char = char


class InsForkHigher(Instruction):
    """Step on, create a new thread with higher priority, and put it to instruction at <to>"""
    def __init__(self, to: int):
        self.to: int = to


class InsForkLower(Instruction):
    """Step on, create a new thread with lower priority, and put it to instruction at <to>"""
    def __init__(self, to: int):
        self.to: int = to


class InsJump(Instruction):
    """Goto instruction at <to>"""
    def __init__(self, to: int):
        self.to: int = to


class InsGroupStart(Instruction):
    """Mark a group with <group_id> as start at string <index>"""
    def __init__(self, index: int, group_id):
        self.index: int = index
        self.group_id = group_id


class InsGroupEnd(Instruction):
    """Mark a group with <group_id> as end at string <index>"""
    def __init__(self, index: int, group_id):
        self.index: int = index
        self.group_id = group_id


class InsPredicate(Instruction):
    """Step on only if <pred> get True, else fail current thread"""
    def __init__(self, pred):
        self.pred: callable = pred


class Thread(object):
    def __init__(self, pc: int, sp: int, starter: int, groups=[]):
        self.pc: int = pc  # Program Counter
        self.sp: int = sp  # String pointer
        self.starter: int = starter  # where does the match started in the current string
        self.groups = groups  # group start end end marks

        # Priority double ended link list
        self.prio_former: 'Thread' = None
        self.prio_later: 'Thread' = None


class Finder(object):
    def finditer(self, program: List[Instruction], text: str) -> Iterator[PTNode]:
        # TODO execute instructions on text and generate PTNodes
        raise NotImplementedError


def compile_pattern(pattern: Pattern) -> Finder:
    # TODO Turn a pattern into a series of instructions
    pass
