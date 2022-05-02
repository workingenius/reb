"""Building instructions as program for vms"""


from typing import List, Optional, Union, Type
from functools import singledispatch
from collections import defaultdict

from ..pattern import (
    Pattern, PText, PAnyChar, PTag, PInChars, PNotInChars,
    PAny, PRepeat, PAdjacent,
    PStarting, PEnding,
    PExample)


class Instruction(object):
    """Instruction of the VM"""
    name: Optional[str] = None

    def __str__(self):
        return str(self.name)


class Program(object):
    """A Tree style instruction series"""

    def __init__(self, sub: List['SubProgram']):
        self.sub: List['SubProgram'] = sub
        self.inst_count: Optional[int] = None
        self._offset: Optional[int] = None

    @property
    def offset(self) -> int:
        return self._offset if self._offset is not None else -1

    @offset.setter
    def offset(self, val: int):
        assert self._offset is None, 'Once program has an offset, it can\'t be modified'
        assert val >= 0, 'An program offset should be gte zero'
        self._offset = val

    def dump(self, offset: int = 0) -> List[Instruction]:
        """Dump to a list of instructions for execution"""
        return dump_prog(self)

    def prepend(self, prog: 'SubProgram'):
        self.sub = [prog] + self.sub

    def append(self, prog: 'SubProgram'):
        self.sub.append(prog)


SubProgram = Union[Instruction, Program]


def dump_prog(prog: SubProgram) -> List[Instruction]:
    ins_lst : List[Instruction] = []
    ptr_lst_dct = defaultdict(list)
    dumping_prog = []

    def _dump_prog(sp):
        if isinstance(sp, InsPointer):
            ptr_lst_dct[sp.program].append(sp)
            ins_lst.append(sp)
        if isinstance(sp, Instruction):
            ins_lst.append(sp)
        elif isinstance(sp, Program):
            if sp in dumping_prog:
                raise Exception("Infinite loop")
            else:
                dumping_prog.append(sp)
            start = len(ins_lst)
            for s in sp.sub:
                _dump_prog(s)
            end = len(ins_lst)
            for ptr in ptr_lst_dct.get(sp, []):
                if ptr.to_ending:
                    ptr.to = end
                else:
                    ptr.to = start
            dumping_prog.pop()

    for ins in ins_lst:
        if isinstance(ins, InsPointer):
            assert ins.to != -1

    _dump_prog(Program(
        [
            InsStart(),
            InsGroupStart(group_id=None),
            prog,
            InsGroupEnd(group_id=None),
            InsSuccess()
        ]
    ))

    return ins_lst


class InsPointer(Instruction):
    """An kind of instruction that has link to a program"""

    def __init__(self, program: Program, to_ending: bool = False):
        self.program = program
        self.to_ending: bool = to_ending
        self.to: int = -1

    def __str__(self):
        return '{} {}'.format(self.name, self.to)


class InsStart(Instruction):
    """Start a re match"""
    name = 'START'


class InsSuccess(Instruction):
    """A match succeed in current thread"""
    name = 'SUCCESS'


class InsCompare(Instruction):
    """Step to next instruction only if current character equals <ch>, otherwise current thread fails"""
    name = 'CMP'

    def __init__(self, ch: str):
        assert len(ch) <= 1
        self.ch: str = ch

    def __str__(self):
        return '{} {}'.format(self.name, repr(self.ch))


class InsForkHigher(InsPointer):
    """Step on, create a new thread with higher priority, and put it to instruction at <to>"""
    name = 'FORKH'


class InsForkLower(InsPointer):
    """Step on, create a new thread with lower priority, and put it to instruction at <to>"""
    name = 'FORKL'


class InsJump(InsPointer):
    """Goto instruction at <to>"""
    name = 'JMP'


class InsGroupStart(Instruction):
    """Mark a group with <group_id> as start"""
    name = 'GROUPSTART'

    def __init__(self, group_id):
        self.group_id = group_id

    def __str__(self):
        return '{} {}'.format(self.name, repr(self.group_id))


class InsGroupEnd(Instruction):
    """Mark a group with <group_id> as ending"""
    name = 'GROUPEND'

    def __init__(self, group_id):
        self.group_id = group_id

    def __str__(self):
        return '{} {}'.format(self.name, repr(self.group_id))


class InsInChars(Instruction):
    """Step on only if in "chars", else fail current thread"""
    name = 'INCHARS'

    def __init__(self, chars: str):
        self.chars = chars

    def __str__(self):
        return '{} {}'.format(self.name, self.chars)


class InsNotInChars(Instruction):
    """Step on only if not in "chars", else fail current thread"""
    name = 'NINCHARS'

    def __init__(self, chars: str):
        self.chars = chars

    def __str__(self):
        return '{} {}'.format(self.name, self.chars)


class InsAny(Instruction):
    """Step on unconditionally"""
    name = 'ANY'


class InsStarting(Instruction):
    """If it is not at the starting, fail the thread"""
    name = 'STARTING'


class InsEnding(Instruction):
    """If it is not at the ending, fail the thread"""
    name = 'ENDING'


class InsResetAdvanced(Instruction):
    """Reset thread <moved> to false"""
    name = 'RESET_ADVANCED'


class InsAssertAdvanced(Instruction):
    """Fail if thread is not moved"""
    name = 'ASSERT_ADVANCED'


class Mark(object):
    def __init__(self, index: int, name, is_open: bool):
        self.index: int = index
        self.name = name
        self.is_open: bool = bool(is_open)
        self.is_close: bool = not self.is_open

    def is_pair(self, other: 'Mark') -> bool:
        return isinstance(other, Mark) and self.name == other.name and (
            (self.is_open and other.is_close)
            or (self.is_close and other.is_open)
        )


@singledispatch
def pattern_to_program(pattern: Pattern) -> Program:
    raise TypeError(
        'Pattern {} can\'t compiled to vm instructions'.format(pattern.__class__))


@pattern_to_program.register(PText)
def _ptext_to_program(pattern: PText) -> Program:
    assert len(pattern.text) > 0
    return Program([InsCompare(c) for c in pattern.text])


@pattern_to_program.register(PAnyChar)
def _panychar_to_program(pattern: PAny) -> Program:
    return Program([InsAny()])


@pattern_to_program.register(PTag)
def _ptag_to_program(pattern: PTag) -> Program:
    return Program([
        InsGroupStart(group_id=pattern.tag),
        pattern_to_program(pattern.pattern),
        InsGroupEnd(group_id=pattern.tag),
    ])


@pattern_to_program.register(PInChars)
def _pinchars_to_program(pattern: PInChars) -> Program:
    return Program([InsInChars(pattern.chars)])


@pattern_to_program.register(PNotInChars)
def _pnotinchars_to_program(pattern: PNotInChars) -> Program:
    return Program([InsNotInChars(pattern.chars)])


@pattern_to_program.register(PAny)
def _pany_to_program(pattern: PAny) -> Program:
    assert len(pattern.patterns) > 0
    if len(pattern.patterns) == 1:
        return pattern_to_program(pattern.patterns[0])
    else:
        prog0 = pattern_to_program(pattern.patterns[0])
        prog1 = pattern_to_program(PAny(pattern.patterns[1:]))

        return Program([
            InsForkLower(program=prog1),
            prog0,
            InsJump(program=prog1, to_ending=True),
            prog1,
        ])


@pattern_to_program.register(PRepeat)
def _prepeat_to_program(pattern: PRepeat) -> Program:
    # 0. preparation
    fr, to = pattern._from, pattern._to

    if fr is None:
        fr = 0

    subprog = pattern_to_program(pattern.pattern)

    fork_cls: Type[Instruction]
    if pattern.greedy:
        fork_cls = InsForkLower
    else:
        fork_cls = InsForkHigher

    # 1. 0-fr as prog0
    prog0 = Program([subprog for i in range(fr)])

    # 2. fr-to as prog1

    # infinite loop
    if to is None:
        prog1 = Program([
            # fork over
            InsResetAdvanced(),
            subprog,
            InsAssertAdvanced(),
            # jump back
        ])
        prog1.prepend(fork_cls(program=prog1, to_ending=True))
        prog1.append(InsJump(program=prog1))

    # exact
    elif to == fr:
        prog1 = Program([])

    elif to > fr:
        prog1 = Program([
            # fork over
            subprog,
        ])
        prog1.prepend(fork_cls(program=prog1, to_ending=True))
        for i in range(to - fr - 1):
            prog1 = Program([
                # fork over
                subprog,
                prog1,
            ])
            prog1.prepend(fork_cls(program=prog1, to_ending=True))

    else:
        raise ValueError

    # 3. combine the two
    return Program([prog0, prog1])


@pattern_to_program.register(PAdjacent)
def _padjacent_to_program(pattern: PAdjacent) -> Program:
    return Program([pattern_to_program(p) for p in pattern.patterns])


@pattern_to_program.register(PExample)
def _pexample_to_program(pattern: PExample) -> Program:
    return pattern_to_program(pattern.pattern)


@pattern_to_program.register(PStarting)
def _pstarting_to_program(pattern: PStarting) -> Program:
    return Program([InsStarting()])


@pattern_to_program.register(PEnding)
def _pending_to_program(pattern: PEnding) -> Program:
    return Program([InsEnding()])
