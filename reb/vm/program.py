"""Building instructions as program for vms"""


from typing import List, Optional, Union, Type
from functools import singledispatch

from ..pattern import (
    Pattern, PText, PAnyChar, PTag, PInChars, PNotInChars,
    PAny, PRepeat, PAdjacent,
    PStarting, PEnding,
    PExample)


class Instruction(object):
    """Instruction of the VM"""
    name: Optional[str] = None

    def ready(self):
        """Mark instruction as ready and should not be modified any more"""

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
        inst_lst = [
            InsStart(),
            InsGroupStart(group_id=None)
        ] + self._dump(offset=2) + [
            InsGroupEnd(group_id=None),
            InsSuccess()
        ]
        for inst in inst_lst:
            inst.ready()
        return inst_lst

    def _dump(self, offset: int = 0) -> List[Instruction]:
        """Dump to a list of instructions for execution"""
        self.offset = offset
        inst_lst: List[Instruction] = []
        for s in self.sub:
            if isinstance(s, Instruction):
                inst_lst.append(s)
            elif isinstance(s, Program):
                ilst = s._dump(offset=offset + len(inst_lst))
                inst_lst.extend(ilst)
        self.inst_count = len(inst_lst)
        return inst_lst

    def prepend(self, prog: 'SubProgram'):
        self.sub = [prog] + self.sub

    def append(self, prog: 'SubProgram'):
        self.sub.append(prog)

    def copy(self):
        return Program(list(self.sub))


SubProgram = Union[Instruction, Program]


class InsPointer(Instruction):
    """An kind of instruction that has link to a program"""

    def __init__(self, program: Program, to_ending: bool = False):
        self.program = program
        self.to_ending: bool = to_ending
        self.to: int = -1

    def ready(self):
        assert self.program.offset >= 0
        if self.to_ending:
            self.to = self.program.offset + self.program.inst_count
        else:
            self.to = self.program.offset
        assert self.to >= 0

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
    prog0 = Program([subprog.copy() for i in range(fr)])

    # 2. fr-to as prog1

    # infinite loop
    if to is None:
        prog1 = Program([
            # fork over
            subprog,
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
            subprog.copy(),
        ])
        prog1.prepend(fork_cls(program=prog1, to_ending=True))
        for i in range(to - fr - 1):
            prog1 = Program([
                # fork over
                subprog.copy(),
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
