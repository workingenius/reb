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

    def __repr__(self):
        return 'Mark(index={}, name={}, is_open={})'.format(
            self.index, self.name, self.is_open)


cdef enum INS:
    NONE,
    START,
    SUCCESS,
    CMP,
    FORKH,
    FORKL,
    JMP,
    GROUPSTART,
    GROUPEND,
    ANY,
    INCHARS,
    NINCHARS,
    STARTING,
    ENDING


cdef class Inst:
    cdef:
        INS _type
        int to
        str chars
        object group_id

    @staticmethod
    cdef Inst create(INS _type = INS.NONE, int to = -1, str chars = None, group_id = None):
        cdef Inst inst = Inst()
        inst._type = _type
        inst.to = to
        inst.chars = chars
        inst.group_id = group_id
        return inst


cdef list inst2inst(list inst_lst):
    cdef Inst inst
    ilst = []
    for i in inst_lst:
        inst = Inst.create()
        ins = i
        if isinstance(ins, InsStart):
            inst._type = INS.START
        elif isinstance(ins, InsSuccess):
            inst._type = INS.SUCCESS
        elif isinstance(ins, InsCompare):
            inst._type = INS.CMP
            inst.chars = ins.ch
        elif isinstance(ins, InsForkHigher):
            inst._type = INS.FORKH
            inst.to = ins.to
        elif isinstance(ins, InsForkLower):
            inst._type = INS.FORKL
            inst.to = ins.to
        elif isinstance(ins, InsJump):
            inst._type = INS.JMP
            inst.to = ins.to
        elif isinstance(ins, InsGroupStart):
            inst._type = INS.GROUPSTART
            inst.group_id = ins.group_id
        elif isinstance(ins, InsGroupEnd):
            inst._type = INS.GROUPEND
            inst.group_id = ins.group_id
        elif isinstance(ins, InsInChars):
            inst._type = INS.INCHARS
            inst.chars = ins.chars
        elif isinstance(ins, InsNotInChars):
            inst._type = INS.NINCHARS
            inst.chars = ins.chars
        elif isinstance(ins, InsAny):
            inst._type = INS.ANY
        elif isinstance(ins, InsStarting):
            inst._type = INS.STARTING
        elif isinstance(ins, InsEnding):
            inst._type = INS.ENDING
        else:
            raise TypeError('Invalid Instruction Type')
        ilst.append(inst)
    return ilst


cdef int _thread_id = 1


cdef class Thread:
    cdef:
        int pc
        int sp
        int starter
        list marks

        Thread prio_former
        Thread prio_later

        int succeed_at
        bint moved
        
        int id

    @staticmethod
    cdef Thread create(int pc, int sp, int starter, list marks = None):
        global _thread_id

        cdef Thread th;

        th = Thread()
        th.pc = pc
        th.sp = sp
        th.starter = starter
        th.marks = list(marks) if marks else []

        th.prio_former = None
        th.prio_later = None

        th.succeed_at = -1
        th.moved = False

        th.id = _thread_id
        _thread_id += 1

        return th

    cdef object to_ptnode(self, str text):
        if not self.marks:
            return None

        mark_stk = []
        node_stk = []

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
        self.program: List[Instruction] = inst2inst(program)

    def finditer(self, text: str) -> Iterator[PTNode]:
        return FinderState.create(self.program, text)


cdef class FinderState:
    cdef:
        str text

        list program

        list thread_map

        Thread cur_hi
        Thread cur_lo
        Thread nxt_hi
        Thread nxt_lo

        int index
        bint step_on

    @staticmethod
    cdef FinderState create(list program, str text):
        cdef FinderState fs = FinderState()

        fs.text = text
        fs.program = program

        # mapping threads pc to threads
        fs.thread_map = [None] * len(program)

        # current thread linked list
        fs.cur_hi = Thread.create(pc=-1, sp=-1, starter=-1)
        fs.cur_lo = Thread.create(pc=-1, sp=-1, starter=-1)
        fs.cur_hi.prio_later = fs.cur_lo
        fs.cur_lo.prio_former = fs.cur_hi

        # next thread linked list
        fs.nxt_hi = Thread.create(pc=-1, sp=-1, starter=-1)  # helper node with highest priority
        fs.nxt_lo = Thread.create(pc=-1, sp=-1, starter=-1)  # helper node with lowest priority
        fs.nxt_hi.prio_later = fs.nxt_lo
        fs.nxt_lo.prio_former = fs.nxt_hi

        # __next__ related vars
        fs.index = 0
        fs.step_on = True

        return fs

    cdef _print_state(self):
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

    cdef void _move_thread_off(self, Thread thread):
        """Put a thread off from its linked list, nothing happens if it is not in a linked list"""
        if thread.prio_former is not None:
            thread.prio_former.prio_later = thread.prio_later
        if thread.prio_later is not None:
            thread.prio_later.prio_former = thread.prio_former
        thread.prio_former = None
        thread.prio_later = None

    cdef void move_thread_higher(self, Thread thread, Thread than):
        self._move_thread_off(thread)
        assert than.prio_former is not None
        thread.prio_later = than
        thread.prio_former = than.prio_former
        thread.prio_former.prio_later = thread
        thread.prio_later.prio_former = thread

    cdef void move_thread_lower(self, Thread thread, Thread than):
        self._move_thread_off(thread)
        assert than.prio_later is not None
        thread.prio_former = than
        thread.prio_later = than.prio_later
        thread.prio_former.prio_later = thread
        thread.prio_later.prio_former = thread

    cdef void del_thread(self, Thread thread):
        self._move_thread_off(thread)
        self.thread_map[thread.pc] = None

    cdef Thread put_thread(self, Thread thread, int pc, bint expel = True):
        assert thread is not None
        if self.thread_map[thread.pc] is thread:
            self.thread_map[thread.pc] = None

        cdef bint replace
        cdef int to = pc
        cdef Thread thread0 = self.thread_map[to]
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

    cdef void thread_for_new_char(self, int index):
        cdef Thread th, th1
        # for every new ch, create a new thread
        # it should be the lowest priority in current linked list
        th = Thread.create(pc=0, sp=index, starter=index)
        th1 = self.put_thread(th, pc=th.pc)
        if th1:
            self.move_thread_higher(th1, than=self.cur_lo)

    cdef object match_done(self, Thread thread, str text):
        cdef:
            int ct
            Thread _th1, _th2

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

    def __next__(self):
        node: Optional[PTNode] = None

        cdef int index
        cdef str ch

        cdef str text = self.text
        cdef list program = self.program
        cdef list thread_map = self.thread_map

        cdef Thread th
        cdef Inst ins
        # cdef Instruction ins

        while node is None:
            index = self.index
            if index == len(text):
                ch = ''
            elif index > len(text):
                raise StopIteration
            else:
                ch = text[index]

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

                if ins._type == INS.START:
                    self.put_thread(th, pc=th.pc + 1)
                elif ins._type == INS.SUCCESS:
                    if th.succeed_at < 0:
                        th.succeed_at = index
                    if self.nxt_hi.prio_later is self.nxt_lo:
                        assert node is None
                        node = self.match_done(th, text)
                        assert node is not None
                    else:
                        self.move_thread_higher(th, than=self.nxt_lo)
                elif ins._type == INS.CMP:
                    self.move_thread_higher(th, than=self.nxt_lo)
                    th.moved = True
                elif ins._type == INS.FORKH:
                    th1 = Thread.create(pc=ins.to, sp=index, starter=th.starter, marks=th.marks)
                    if self.put_thread(th1, pc=th1.pc):
                        self.move_thread_higher(th1, than=th)
                    self.put_thread(th, pc=th.pc + 1)
                elif ins._type == INS.FORKL:
                    th1 = Thread.create(pc=ins.to, sp=index, starter=th.starter, marks=th.marks)
                    if self.put_thread(th1, pc=th1.pc):
                        self.move_thread_lower(th1, than=th)
                    self.put_thread(th, pc=th.pc + 1)
                elif ins._type == INS.JMP:
                    self.put_thread(th, pc=ins.to)
                elif ins._type == INS.GROUPSTART:
                    th.marks.append(Mark(index=index, name=ins.group_id, is_open=True, depth=0))  # TODO depth
                    self.put_thread(th, pc=th.pc + 1)
                elif ins._type == INS.GROUPEND:
                    th.marks.append(Mark(index=index, name=ins.group_id, is_open=False, depth=0))  # TODO depth
                    self.put_thread(th, pc=th.pc + 1)
                elif ins._type == INS.INCHARS:
                    self.move_thread_higher(th, than=self.nxt_lo)
                    th.moved = True
                elif ins._type == INS.NINCHARS:
                    self.move_thread_higher(th, than=self.nxt_lo)
                    th.moved = True
                elif ins._type == INS.ANY:
                    self.move_thread_higher(th, than=self.nxt_lo)
                    th.moved = True
                elif ins._type == INS.STARTING:
                    if index == 0:
                        if self.put_thread(th, pc=th.pc + 1):
                            self.move_thread_higher(th, than=self.cur_lo)
                    else:
                        self.del_thread(th)
                elif ins._type == INS.ENDING:
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
                if ins._type == INS.SUCCESS:
                    pass
                elif ins._type == INS.CMP:
                    if ins.chars == ch:
                        self.put_thread(th, pc=th.pc + 1, expel=False)
                    else:
                        self.del_thread(th)
                elif ins._type == INS.INCHARS:
                    if ch in ins.chars:
                        self.put_thread(th, pc=th.pc + 1, expel=False)
                    else:
                        self.del_thread(th)
                elif ins._type == INS.NINCHARS:
                    if ch not in ins.chars:
                        self.put_thread(th, pc=th.pc + 1, expel=False)
                    else:
                        self.del_thread(th)
                elif ins._type == INS.ANY:
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
                    assert ins._type == INS.SUCCESS
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
