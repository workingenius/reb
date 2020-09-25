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


class InsPredicate(Instruction):
    """Step on only if <pred> get True, else fail current thread"""
    name = 'PRED'

    def __init__(self, pred):
        self.pred: Callable[[str, int, str], bool] = pred

    def __str__(self):
        return '{} {}'.format(self.name, repr(self.pred))


class InsAny(Instruction):
    """Step on unconditionally"""
    name = 'ANY'


class InsCount(Instruction):
    """Count how many times a thread has passed this instruction"""
    name = 'COUNT'


class InsCountGTE(Instruction):
    """Pass only if the thread has passed a given InsCount for more than or equals to  <bounder> times"""
    name = 'COUNTGTE'

    def __init__(self, counter: InsCount, bounder: int):
        self.counter: InsCount = counter
        self.bounder: int = bounder

    def __str__(self):
        return '{} {}'.format(self.name, self.bounder)


class InsCountLTE(Instruction):
    """Pass only if the thread has passed a given InsCount for less than or equals to <bounder> times"""
    name = 'COUNTLTE'

    def __init__(self, counter: InsCount, bounder: int):
        self.counter: InsCount = counter
        self.bounder: int = bounder

    def __str__(self):
        return '{} {}'.format(self.name, self.bounder)


class InsAssert(Instruction):
    """If pred is not satisfied, fail the thread"""
    name = 'ASSERT'

    def __init__(self, pred):
        self.pred: Callable[[str, int, str], bool] = pred

    def __str__(self):
        return '{} {}'.format(self.name, repr(self.pred))


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


def _thread_id_generator():
    i = 0
    while True:
        i += 1
        yield i
thread_id_generator = _thread_id_generator()


class Thread(object):
    def __init__(self, pc: int, sp: int, starter: int, marks=None,
                 counter: Dict[InsCount, int] = None):
        self.pc: int = pc  # Program Counter
        self.sp: int = sp  # String pointer
        self.starter: int = starter  # where does the match started in the current string
        self.marks: List[Mark] = list(marks) if marks else []  # group start end end marks

        # Priority double ended link list
        self.prio_former: Optional['Thread'] = None
        self.prio_later: Optional['Thread'] = None

        self.counter: Dict[InsCount, int] = defaultdict(int)
        if counter:
            self.counter.update(counter)

        self.id = next(thread_id_generator)

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
        program = self.program

        # mapping threads pc to threads
        thread_map: Dict[int, Optional[Thread]] = {}

        # current thread linked list
        cur_hi = Thread(pc=-1, sp=-1, starter=-1)  # helper node with highest priority
        cur_lo = Thread(pc=-1, sp=-1, starter=-1)  # helper node with lowest priority
        cur_hi.prio_later = cur_lo
        cur_lo.prio_former = cur_hi

        # next thread linked list
        nxt_hi = Thread(pc=-1, sp=-1, starter=-1)  # helper node with highest priority
        nxt_lo = Thread(pc=-1, sp=-1, starter=-1)  # helper node with lowest priority
        nxt_hi.prio_later = nxt_lo
        nxt_lo.prio_former = nxt_hi

        def _print_state():
            for i in range(len(self.program)):
                line = '{}.\t{}'.format(i, str(self.program[i]))
                if thread_map.get(i):
                    line += '\t<- {}'.format(str(thread_map[i]))
                print(line)
            
            segs = ['cur lst: head']
            ptr = cur_hi.prio_later
            while ptr is not cur_lo:
                segs.append(str(ptr))
                ptr = ptr.prio_later
            segs.append('tail')
            line = ' => '.join(segs)
            print(line)

            segs = ['nxt lst: head']
            ptr = nxt_hi.prio_later
            while ptr is not nxt_lo:
                segs.append(str(ptr))
                ptr = ptr.prio_later
            segs.append('tail')
            line = ' => '.join(segs)
            print(line)

            print()

        def _move_thread_off(thread: Thread) -> None:
            """Put a thread off from its linked list, nothing happens if it is not in a linked list"""
            if thread.prio_former is not None:
                thread.prio_former.prio_later = thread.prio_later
            if thread.prio_later is not None:
                thread.prio_later.prio_former = thread.prio_former
            thread.prio_former = None
            thread.prio_later = None

        def move_thread_higher(thread: Thread, than: Thread) -> None:
            _move_thread_off(thread)
            assert than.prio_former is not None
            thread.prio_later = than
            thread.prio_former = than.prio_former
            thread.prio_former.prio_later = thread
            thread.prio_later.prio_former = thread

        def move_thread_lower(thread: Thread, than: Thread) -> None:
            _move_thread_off(thread)
            assert than.prio_later is not None
            thread.prio_former = than
            thread.prio_later = than.prio_later
            thread.prio_former.prio_later = thread
            thread.prio_later.prio_former = thread

        def del_thread(thread: Thread) -> None:
            _move_thread_off(thread)
            thread_map[thread.pc] = None

        def put_thread(thread: Thread, pc: int, expel: bool) -> Optional[Thread]:
            if thread_map.get(thread.pc) is thread:
                thread_map.pop(thread.pc)

            to = pc
            thread0 = thread_map.get(to)
            if thread0:
                # Should thread replace original thread0 or not
                replace = True
                if thread.starter > thread0.starter:
                    replace = False
                elif thread.starter == thread0.starter:
                    replace = expel

                if not replace:
                    _move_thread_off(thread)
                    return None
                else:
                    del_thread(thread0)
            
            thread_map[to] = thread
            thread.pc = to
            return thread

        def thread_for_new_char(index: int):
            # for every new char, create a new thread
            # it should be the lowest priority in current linked list
            th = Thread(pc=0, sp=index, starter=index)
            th1 = put_thread(th, pc=th.pc, expel=False)
            if th1:
                move_thread_higher(th1, than=cur_lo)

        def eof():
            yield ''

        for index, char in enumerate(chain(iter(text), eof())):
            thread_for_new_char(index)

            # as long as the ready ll is not empty
            while cur_hi.prio_later is not cur_lo:
                if DEBUG:
                    _print_state()

                # pick the ready thread with highest prioity and run it
                th = cur_hi.prio_later
                ins = program[th.pc]
                if isinstance(ins, InsStart):
                    put_thread(th, pc=th.pc + 1, expel=True)
                elif isinstance(ins, InsSuccess):
                    if nxt_hi.prio_later is nxt_lo:
                        # find a match, arrange it to be PTNodes, and re-init threads
                        node = th.to_ptnode(text)
                        assert node is not None
                        yield node
                        thread_map = {}
                        cur_hi.prio_later = cur_lo
                        cur_lo.prio_former = cur_hi
                        nxt_hi.prio_later = nxt_lo
                        nxt_lo.prio_former = nxt_hi
                        if th.starter != index:
                            thread_for_new_char(index)
                    else:
                        move_thread_higher(th, than=nxt_lo)
                elif isinstance(ins, InsCompare):
                    if ins.char == char:
                        # step on and put it to the lowest in next linked list
                        th1 = put_thread(th, pc=th.pc + 1, expel=True)
                        if th1:
                            move_thread_higher(th, than=nxt_lo)
                    else:
                        del_thread(th)
                elif isinstance(ins, InsForkHigher):
                    th1 = Thread(pc=ins.to, sp=index, starter=th.starter, marks=th.marks, counter=th.counter)
                    th1 = put_thread(th1, pc=th1.pc, expel=True)
                    if th1:
                        move_thread_higher(th1, than=th)
                    put_thread(th, pc=th.pc + 1, expel=True)
                elif isinstance(ins, InsForkLower):
                    th1 = Thread(pc=ins.to, sp=index, starter=th.starter, marks=th.marks, counter=th.counter)
                    th1 = put_thread(th1, pc=th1.pc, expel=True)
                    if th1:
                        move_thread_lower(th1, than=th)
                    put_thread(th, pc=th.pc + 1, expel=True)
                elif isinstance(ins, InsJump):
                    put_thread(th, pc=ins.to, expel=True)
                elif isinstance(ins, InsGroupStart):
                    th.marks.append(Mark(index=index, name=ins.group_id, is_open=True, depth=0))  # TODO depth
                    put_thread(th, pc=th.pc + 1, expel=True)
                elif isinstance(ins, InsGroupEnd):
                    th.marks.append(Mark(index=index, name=ins.group_id, is_open=False, depth=0))  # TODO depth
                    put_thread(th, pc=th.pc + 1, expel=True)
                elif isinstance(ins, InsPredicate):
                    if ins.pred(char, index, text):
                        th1 = put_thread(th, pc=th.pc + 1, expel=True)
                        if th1:
                            move_thread_higher(th, than=nxt_lo)
                    else:
                        del_thread(th)
                elif isinstance(ins, InsAny):
                    th1 = put_thread(th, pc=th.pc + 1, expel=True)
                    if th1:
                        move_thread_higher(th, than=nxt_lo)
                # TODO: a bug spotted: nowhere to clean the count, so if a thread enter a loop several times, the bug appears
                elif isinstance(ins, InsCount):
                    th.counter[ins] += 1
                    put_thread(th, pc=th.pc + 1, expel=True)
                elif isinstance(ins, InsCountGTE):
                    # print(th.counter, ins, ins.counter)
                    if th.counter.get(ins.counter, 0) >= ins.bounder:
                        put_thread(th, pc=th.pc + 1, expel=True)
                    else:
                        del_thread(th)
                elif isinstance(ins, InsCountLTE):
                    # print(th.counter, ins, ins.counter)
                    if th.counter.get(ins.counter, 0) <= ins.bounder:
                        put_thread(th, pc=th.pc + 1, expel=True)
                    else:
                        del_thread(th)
                elif isinstance(ins, InsAssert):
                    if ins.pred(char, index, text):
                        th1 = put_thread(th, pc=th.pc + 1, expel=True)
                        if th1:
                            move_thread_higher(th, than=cur_lo)
                    else:
                        del_thread(th)
                else:
                    raise TypeError('Invalid Instruction Type')

            if DEBUG:
                _print_state()

            cur_hi, cur_lo, nxt_hi, nxt_lo = nxt_hi, nxt_lo, cur_hi, cur_lo


def compile_pattern(pattern: Pattern) -> Finder:
    return Finder(_pattern_to_program(pattern).dump())


@singledispatch
def _pattern_to_program(pattern: Pattern) -> Program:
    raise TypeError('Pattern {} can\'t compiled to vm instructions'.format(pattern.__class__))


@_pattern_to_program.register(PText)
def _ptext_to_program(pattern: PText) -> Program:
    assert len(pattern.text) > 0
    return Program([InsCompare(c) for c in pattern.text])


@_pattern_to_program.register(PAnyChar)
def _panychar_to_program(pattern: PAny) -> Program:
    return Program([InsAny()])


@_pattern_to_program.register(PTag)
def _ptag_to_program(pattern: PTag) -> Program:
    return Program([
        InsGroupStart(group_id=pattern.tag),
        _pattern_to_program(pattern.pattern),
        InsGroupEnd(group_id=pattern.tag),
    ])


@_pattern_to_program.register(PInChars)
def _pinchars_to_program(pattern: PInChars) -> Program:
    return Program([InsPredicate((lambda c, i, t: c in pattern.chars))])


@_pattern_to_program.register(PNotInChars)
def _pnotinchars_to_program(pattern: PNotInChars) -> Program:
    return Program([InsPredicate((lambda c, i, t: c not in pattern.chars))])


@_pattern_to_program.register(PAny)
def _pany_to_program(pattern: PAny) -> Program:
    assert len(pattern.patterns) > 0
    if len(pattern.patterns) == 1:
        return _pattern_to_program(pattern.patterns[0])
    else:
        prog0 = _pattern_to_program(pattern.patterns[0])
        prog1 = _pattern_to_program(PAny(pattern.patterns[1:]))

        return Program([
            InsForkLower(program=prog1),
            prog0,
            InsJump(program=prog1, to_ending=True),
            prog1,
        ])


@_pattern_to_program.register(PRepeat)
def _prepeat_to_program(pattern: PRepeat) -> Program:
    fork: List[SubProgram]
    jump: List[SubProgram]

    prog0 = _pattern_to_program(pattern.pattern)

    # ?
    if (None, 1) == (pattern._from, pattern._to):
        if pattern.greedy:
            prog = Program([
                InsForkLower(program=prog0, to_ending=True),
                prog0
            ])
        else:
            prog = Program([
                InsForkHigher(program=prog0, to_ending=True),
                prog0
            ])

    # +
    elif (1, None) == (pattern._from, pattern._to):
        if pattern.greedy:
            prog = Program([
                prog0,
                InsForkHigher(program=prog0)
            ])
        else:
            prog = Program([
                prog0,
                InsForkLower(program=prog0)
            ])
    
    # *
    elif (0, None) == (pattern._from, pattern._to):
        ins_cls: Type[Instruction]
        if pattern.greedy:
            ins_cls = InsForkLower
        else:
            ins_cls = InsForkHigher
        prog = Program([
            # fork over
            prog0,
            # jump back
        ])
        fork = [ins_cls(program=prog, to_ending=True)]
        prog.sub = fork + prog.sub
        jump = [InsJump(program=prog)]
        prog.sub = prog.sub + jump

    # {x,}
    elif pattern._to is None:
        if pattern.greedy:
            ins_cls = InsForkLower
        else:
            ins_cls = InsForkHigher
        counter = InsCount()
        prog = Program([
            # fork over
            prog0,
            counter,
            # jump back
        ])
        prog.prepend(ins_cls(program=prog, to_ending=True))
        prog.append(InsJump(program=prog))
        count_checker = InsCountGTE(counter, pattern._from)
        prog = Program([
            prog,
            count_checker,
        ])

    # {x, y}
    else:
        assert pattern._to is not None
        if pattern.greedy:
            ins_cls = InsForkLower
        else:
            ins_cls = InsForkHigher
        counter = InsCount()
        prog = Program([
            # fork over
            # check upper bound
            prog0,
            counter,
            # jump back
            # check lower bound
        ])

        prog.prepend(InsCountLTE(counter, pattern._to - 1))
        prog.prepend(ins_cls(program=prog, to_ending=True))
        prog.append(InsJump(program=prog))
        
        count_checker1 = InsCountGTE(counter, pattern._from)
        prog = Program([
            prog,
            count_checker1,
        ])

    return prog


@_pattern_to_program.register(PAdjacent)
def _padjacent_to_program(pattern: PAdjacent) -> Program:
    return Program([_pattern_to_program(p) for p in pattern.patterns])


@_pattern_to_program.register(PExample)
def _pexample_to_program(pattern: PExample) -> Program:
    return _pattern_to_program(pattern.pattern)


@_pattern_to_program.register(PStarting)
def _pstarting_to_program(pattern: PStarting) -> Program:
    return Program([InsAssert(lambda c, i, t: i == 0)])


@_pattern_to_program.register(PEnding)
def _pending_to_program(pattern: PEnding) -> Program:
    return Program([InsAssert(lambda c, i, t: i == len(t))])
