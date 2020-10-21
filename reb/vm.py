"""Implement Regular Expression with vm

Basic ideas borrowed mainly from https://swtch.com/~rsc/regexp/regexp2.html
"""


# TODO able to choose match engine (plain / vm) on doing pattern extractions
# TODO make overall integrated tests and ensure the vm solves every conner case
# TODO speed up vm


import os
from typing import Iterator, List, Optional, Union, Dict, Callable, Sequence, Type
from functools import singledispatch
from itertools import chain
from collections import defaultdict

from .parse_tree import PTNode
from .pattern import (
    Finder as BaseFinder,
    Pattern, PText, PAnyChar, PTag, PInChars, PNotInChars,
    PAny, PClause, PRepeat, PAdjacent,
    PStarting, PEnding,
    PExample)


DEBUG = bool(os.environ.get('REB_DEBUG'))


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
    """Step to next instruction only if current character equals <char>, otherwise current thread fails"""
    name = 'CMP'

    def __init__(self, char: str):
        assert len(char) <= 1
        self.char: str = char

    def __str__(self):
        return '{} {}'.format(self.name, repr(self.char))


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
    def __init__(self, index: int, name, is_open: bool, depth: int):
        self.index: int = index
        self.name = name
        self.is_open: bool = bool(is_open)
        self.is_close: bool = not self.is_open
        self.depth: int = depth

    def is_pair(self, other: 'Mark') -> bool:
        return isinstance(other, Mark) and self.name == other.name and (
            (self.is_open and other.is_close)
            or (self.is_close and other.is_open)
        )


_thread_id = [1]


class Thread(object):
    def __init__(self, pc: int, sp: int, starter: int, marks=None):
        self.pc: int = pc  # Program Counter
        self.sp: int = sp  # String pointer
        self.starter: int = starter  # where does the match started in the current string
        self.marks: List[Mark] = list(marks) if marks else []  # group start end end marks

        # Priority double ended link list
        self.prio_former: Optional['Thread'] = None
        self.prio_later: Optional['Thread'] = None

        self.succeed_at: int = -1
        self.moved: bool = False

        self.id = _thread_id[0]
        _thread_id[0] += 1

    def to_ptnode(self, text: str) -> Optional[PTNode]:
        if not self.marks:
            return None

        mark_stk: List[Mark] = []
        node_stk: List[PTNode] = []

        for m in self.marks:
            if m.is_open:
                mark_stk.append(m)
            else:
                assert mark_stk
                m0 = mark_stk.pop()
                assert m.is_pair(m0)

                start = m0.index
                end = m.index
                chl = []
                while node_stk and node_stk[-1].index0 >= start:
                    chl.append(node_stk.pop())
                node_stk.append(PTNode(text, start=start, end=end, children=chl, tag=m.name))

        assert not mark_stk
        assert len(node_stk) == 1
        return node_stk[0]

    def __str__(self):
        return '<reb Thread {}>'.format(self.id)


class Finder(BaseFinder):
    def __init__(self, program: List[Instruction]):
        self.program: List[Instruction] = program

    def finditer(self, text: str) -> Iterator[PTNode]:
        return FinderState(self.program, text)


class FinderState(object):
    def __init__(self, program: List[Instruction], text: str):
        self.text: str = text

        self.program: List[Instruction] = program

        # mapping threads pc to threads
        self.thread_map: List[Optional[Thread]] = [None] * len(program)

        # current thread linked list
        self.cur_hi = Thread(pc=-1, sp=-1, starter=-1)  # helper node with highest priority
        self.cur_lo = Thread(pc=-1, sp=-1, starter=-1)  # helper node with lowest priority
        self.cur_hi.prio_later = self.cur_lo
        self.cur_lo.prio_former = self.cur_hi

        # next thread linked list
        self.nxt_hi = Thread(pc=-1, sp=-1, starter=-1)  # helper node with highest priority
        self.nxt_lo = Thread(pc=-1, sp=-1, starter=-1)  # helper node with lowest priority
        self.nxt_hi.prio_later = self.nxt_lo
        self.nxt_lo.prio_former = self.nxt_hi

        # __next__ related vars
        self.index: int = 0
        self.step_on: bool = True

    def _print_state(self):
        for i in range(len(self.program)):
            line = '{}.\t{}'.format(i, str(self.program[i]))
            if self.thread_map[i]:
                line += '\t<- {}'.format(str(self.thread_map[i]))
            print(line)
        
        segs = ['cur lst: head']
        ptr = self.cur_hi.prio_later
        while ptr is not self.cur_lo:
            segs.append(str(ptr))
            ptr = ptr.prio_later
        segs.append('tail')
        line = ' => '.join(segs)
        print(line)

        segs = ['nxt lst: head']
        ptr = self.nxt_hi.prio_later
        while ptr is not self.nxt_lo:
            segs.append(str(ptr))
            ptr = ptr.prio_later
        segs.append('tail')
        line = ' => '.join(segs)
        print(line)

        print()

    @staticmethod
    def _move_thread_off(thread: Thread) -> None:
        """Put a thread off from its linked list, nothing happens if it is not in a linked list"""
        if thread.prio_former is not None:
            thread.prio_former.prio_later = thread.prio_later
        if thread.prio_later is not None:
            thread.prio_later.prio_former = thread.prio_former
        thread.prio_former = None
        thread.prio_later = None

    def move_thread_higher(self, thread: Thread, than: Thread) -> None:
        self._move_thread_off(thread)
        assert than.prio_former is not None
        thread.prio_later = than
        thread.prio_former = than.prio_former
        thread.prio_former.prio_later = thread
        thread.prio_later.prio_former = thread

    def move_thread_lower(self, thread: Thread, than: Thread) -> None:
        self._move_thread_off(thread)
        assert than.prio_later is not None
        thread.prio_former = than
        thread.prio_later = than.prio_later
        thread.prio_former.prio_later = thread
        thread.prio_later.prio_former = thread

    def del_thread(self, thread: Thread) -> None:
        self._move_thread_off(thread)
        self.thread_map[thread.pc] = None

    def put_thread(self, thread: Thread, pc: int, expel: bool = True) -> Optional[Thread]:
        assert thread is not None
        if self.thread_map[thread.pc] is thread:
            self.thread_map[thread.pc] = None

        to = pc
        thread0 = self.thread_map[to]
        if thread0:
            # Should thread replace original thread0 or not
            replace = True
            if thread.starter > thread0.starter:
                replace = False
            elif thread.starter == thread0.starter:
                if thread0.moved:
                    replace = False
                else:
                    replace = expel

            if not replace:
                self._move_thread_off(thread)
                return None
            else:
                self.del_thread(thread0)
        
        self.thread_map[to] = thread
        thread.pc = to
        return thread

    def thread_for_new_char(self, index: int):
        # for every new char, create a new thread
        # it should be the lowest priority in current linked list
        th = Thread(pc=0, sp=index, starter=index)
        th1 = self.put_thread(th, pc=th.pc)
        if th1:
            self.move_thread_higher(th1, than=self.cur_lo)

    def match_done(self, thread: Thread, text: str) -> Optional[PTNode]:
        # found a match, arrange it to be PTNodes, and re-init threads
        node = thread.to_ptnode(text)
        self.del_thread(thread)
        # clear all earlier started threads
        ct = thread.succeed_at  # clear_till
        _th1 = self.cur_hi.prio_later
        assert _th1 is not None
        _th2 = _th1.prio_later
        while _th2 is not None:
            if _th1.starter < ct or _th1.starter == thread.starter:
                self.del_thread(_th1)
            _th1 = _th2
            _th2 = _th2.prio_later
        _th1 = self.nxt_hi.prio_later
        assert _th1 is not None
        _th2 = _th1.prio_later
        while _th2 is not None:
            if _th1.starter < ct or _th1.starter == thread.starter:
                self.del_thread(_th1)
            _th1 = _th2
            _th2 = _th2.prio_later
        return node

    def __iter__(self) -> Iterator[PTNode]:
        return self

    def __next__(self) -> PTNode:
        node: Optional[PTNode] = None

        text = self.text
        program = self.program
        thread_map = self.thread_map

        th: Optional[Thread]
        ins: Instruction

        while node is None:
            index = self.index
            if index == len(text):
                char = ''
            elif index > len(text):
                raise StopIteration
            else:
                char = text[index]

            if self.step_on:
                self.thread_for_new_char(index)
            self.step_on = True

            # as long as the ready ll is not empty
            while self.cur_hi.prio_later is not self.cur_lo:
                if DEBUG:
                    self._print_state()

                # pick the ready thread with highest prioity and run it
                th = self.cur_hi.prio_later
                assert th is not None
                ins = program[th.pc]

                if isinstance(ins, InsStart):
                    self.put_thread(th, pc=th.pc + 1)
                elif isinstance(ins, InsSuccess):
                    if th.succeed_at < 0:
                        th.succeed_at = index
                    if self.nxt_hi.prio_later is self.nxt_lo:
                        assert node is None
                        node = self.match_done(th, text)
                        assert node is not None
                    else:
                        self.move_thread_higher(th, than=self.nxt_lo)
                elif isinstance(ins, InsCompare):
                    self.move_thread_higher(th, than=self.nxt_lo)
                    th.moved = True
                elif isinstance(ins, InsForkHigher):
                    th1 = Thread(pc=ins.to, sp=index, starter=th.starter, marks=th.marks)
                    if self.put_thread(th1, pc=th1.pc):
                        self.move_thread_higher(th1, than=th)
                    self.put_thread(th, pc=th.pc + 1)
                elif isinstance(ins, InsForkLower):
                    th1 = Thread(pc=ins.to, sp=index, starter=th.starter, marks=th.marks)
                    if self.put_thread(th1, pc=th1.pc):
                        self.move_thread_lower(th1, than=th)
                    self.put_thread(th, pc=th.pc + 1)
                elif isinstance(ins, InsJump):
                    self.put_thread(th, pc=ins.to)
                elif isinstance(ins, InsGroupStart):
                    th.marks.append(Mark(index=index, name=ins.group_id, is_open=True, depth=0))  # TODO depth
                    self.put_thread(th, pc=th.pc + 1)
                elif isinstance(ins, InsGroupEnd):
                    th.marks.append(Mark(index=index, name=ins.group_id, is_open=False, depth=0))  # TODO depth
                    self.put_thread(th, pc=th.pc + 1)
                elif isinstance(ins, InsInChars):
                    self.move_thread_higher(th, than=self.nxt_lo)
                    th.moved = True
                elif isinstance(ins, InsNotInChars):
                    self.move_thread_higher(th, than=self.nxt_lo)
                    th.moved = True
                elif isinstance(ins, InsAny):
                    self.move_thread_higher(th, than=self.nxt_lo)
                    th.moved = True
                elif isinstance(ins, InsStarting):
                    if index == 0:
                        if self.put_thread(th, pc=th.pc + 1):
                            self.move_thread_higher(th, than=self.cur_lo)
                    else:
                        self.del_thread(th)
                elif isinstance(ins, InsEnding):
                    if index == len(text):
                        if self.put_thread(th, pc=th.pc + 1):
                            self.move_thread_higher(th, than=self.cur_lo)
                    else:
                        self.del_thread(th)
                else:
                    raise TypeError('Invalid Instruction Type')

                if node is not None:
                    self.step_on = False
                    return node

            if DEBUG:
                self._print_state()

            # for each alive thread, step on
            # and reset moved flag to "not moved"
            for i in range(len(thread_map)-1, -1, -1):
                th = thread_map[i]
                if th is None:
                    continue
                assert th is not None
                th.moved = False
                ins = program[th.pc]
                if isinstance(ins, InsSuccess):
                    pass
                elif isinstance(ins, InsCompare):
                    if ins.char == char:
                        self.put_thread(th, pc=th.pc + 1, expel=False)
                    else:
                        self.del_thread(th)
                elif isinstance(ins, InsInChars):
                    if char in ins.chars:
                        self.put_thread(th, pc=th.pc + 1, expel=False)
                    else:
                        self.del_thread(th)
                elif isinstance(ins, InsNotInChars):
                    if char not in ins.chars:
                        self.put_thread(th, pc=th.pc + 1, expel=False)
                    else:
                        self.del_thread(th)
                elif isinstance(ins, InsAny):
                    self.put_thread(th, pc=th.pc + 1, expel=False)
                else:
                    raise TypeError

            if DEBUG:
                self._print_state()

            # swap cur list and next list
            self.cur_hi, self.cur_lo, self.nxt_hi, self.nxt_lo = self.nxt_hi, self.nxt_lo, self.cur_hi, self.cur_lo

            # ending
            if index == len(text):
                th = thread_map[-1]
                if th is not None:
                    ins = program[th.pc]
                    assert isinstance(ins, InsSuccess)
                    assert node is None
                    node = self.match_done(th, text)
                    assert node is not None
                
                if node is None:
                    raise StopIteration
        
            self.index += 1

        return node


def compile_pattern(pattern: Pattern) -> Finder:
    return Finder(pattern_to_program(pattern).dump())


@singledispatch
def pattern_to_program(pattern: Pattern) -> Program:
    raise TypeError('Pattern {} can\'t compiled to vm instructions'.format(pattern.__class__))


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
