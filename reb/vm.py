"""Implement Regular Expression with vm"""


from typing import Iterator, List, Optional, Union, Dict, Callable, Sequence
from functools import singledispatch

from .parse_tree import PTNode
from .pattern import (
    Pattern, PText, PAnyChar, PTag, PInChars, PNotInChars, PAny,
    PClause, PRepeat, PAdjacent)


class Instruction(object):
    """Instruction of the VM"""

    def ready(self):
        """Mark instruction as ready and should not be modified any more"""


class Program(object):
    """A Tree style instruction series"""

    def __init__(self, sub: List['SubProgram']):
        self.sub: List['SubProgram'] = sub

        self.inst_lst: List[Instruction] = []
        for s in self.sub:
            if isinstance(s, Instruction):
                self.inst_lst.append(s)
            elif isinstance(s, Program):
                self.inst_lst.extend(s.inst_lst)

        self.inst_count: int = len(self.inst_lst)
        
        self._offset: Optional[int] = None

    @property
    def offset(self) -> int:
        return self._offset if self._offset is not None else -1

    @offset.setter
    def offset(self, val: int):
        assert self._offset is None, 'Once program has an offset, it can\'t be modified'
        assert val >= 0, 'An program offset should be gle zero'
        self._offset = val

    def dump(self, is_root: bool = True, offset: int = 0) -> List[Instruction]:
        """Dump to a list of instructions for execution"""
        inst_lst: List[Instruction] = []
        if is_root:
            self.offset = 0
            inst_lst.append(InsStart())
            inst_lst.append(InsGroupStart(group_id=None))

        for s in self.sub:
            if isinstance(s, Instruction):
                inst_lst.append(s)
            elif isinstance(s, Program):
                s.offset = offset + len(inst_lst)
                ilst = s.dump(is_root=False, offset=s.offset)
                inst_lst.extend(ilst)

        if is_root:
            inst_lst.append(InsGroupEnd(group_id=None))
            inst_lst.append(InsSuccess())

            for inst in inst_lst:
                inst.ready()
        return inst_lst


SubProgram = Union[Instruction, Program]


class InsPointer(Instruction):
    """An kind of instruction that has link to a program"""

    def __init__(self, program: Program, to_ending: bool = False):
        self.program = program
        self.to_ending: bool = to_ending
        self.to: int = -1

    def ready(self):
        try:
            assert self.program.offset >= 0
        except AssertionError:
            import pdb; pdb.set_trace()
        if self.to_ending:
            self.to = self.program.offset + self.program.inst_count
        else:
            self.to = self.program.offset
        assert self.to >= 0


class InsStart(Instruction):
    """Start a re match"""


class InsSuccess(Instruction):
    """A match succeed in current thread"""


class InsCompare(Instruction):
    """Step to next instruction only if current character equals <char>, otherwise current thread fails"""
    def __init__(self, char: str):
        assert len(char) <= 1
        self.char: str = char


class InsForkHigher(InsPointer):
    """Step on, create a new thread with higher priority, and put it to instruction at <to>"""


class InsForkLower(InsPointer):
    """Step on, create a new thread with lower priority, and put it to instruction at <to>"""


class InsJump(InsPointer):
    """Goto instruction at <to>"""


class InsGroupStart(Instruction):
    """Mark a group with <group_id> as start"""
    def __init__(self, group_id):
        self.group_id = group_id


class InsGroupEnd(Instruction):
    """Mark a group with <group_id> as ending"""
    def __init__(self, group_id):
        self.group_id = group_id


class InsPredicate(Instruction):
    """Step on only if <pred> get True, else fail current thread"""
    def __init__(self, pred):
        self.pred: Callable[[str, int], bool] = pred


class InsAny(Instruction):
    """Step on unconditionally"""


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


class Thread(object):
    def __init__(self, pc: int, sp: int, starter: int, marks=None):
        self.pc: int = pc  # Program Counter
        self.sp: int = sp  # String pointer
        self.starter: int = starter  # where does the match started in the current string
        self.marks: List[Mark] = list(marks) if marks else []  # group start end end marks

        # Priority double ended link list
        self.prio_former: Optional['Thread'] = None
        self.prio_later: Optional['Thread'] = None

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


class Finder(object):
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
                    return None
                else:
                    del_thread(thread0)
            
            thread_map[to] = thread
            thread.pc = to
            return thread

        # TODO: ending
        for index, char in enumerate(text):
            # for every new char, create a new thread
            # it should be the lowest priority in current linked list
            th = Thread(pc=0, sp=index, starter=index)
            th1 = put_thread(th, pc=th.pc, expel=False)
            if th1:
                move_thread_higher(th1, than=cur_lo)

            # as long as the ready ll is not empty
            while cur_hi.prio_later is not cur_lo:
                # pick the ready thread with highest prioity and run it
                th = cur_hi.prio_later
                ins = program[th.pc]
                if isinstance(ins, InsStart):
                    put_thread(th, pc=th.pc + 1, expel=True)
                elif isinstance(ins, InsSuccess):
                    # find a match, arrange it to be PTNodes, and re-init threads
                    node = th.to_ptnode(text)
                    assert node is not None
                    yield node
                    thread_map = {}
                    cur_hi.prio_later = cur_lo
                    nxt_hi.prio_later = nxt_lo
                elif isinstance(ins, InsCompare):
                    if ins.char == char:
                        # step on and put it to the lowest in next linked list
                        put_thread(th, pc=th.pc + 1, expel=True)
                        move_thread_higher(th, than=nxt_lo)
                    else:
                        del_thread(th)
                elif isinstance(ins, InsForkHigher):
                    th1 = Thread(pc=ins.to, sp=index, starter=th.starter, marks=th.marks)
                    th1 = put_thread(th1, pc=th1.pc, expel=True)
                    if th1:
                        move_thread_higher(th1, than=th)
                    put_thread(th, pc=th.pc + 1, expel=True)
                elif isinstance(ins, InsForkLower):
                    th1 = Thread(pc=ins.to, sp=index, starter=th.starter, marks=th.marks)
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
                    if ins.pred(char, index):
                        put_thread(th, pc=th.pc + 1, expel=True)
                        move_thread_higher(th, than=nxt_lo)
                    else:
                        del_thread(th)
                elif isinstance(ins, InsAny):
                    put_thread(th, pc=th.pc + 1, expel=True)
                    move_thread_higher(th, than=nxt_lo)

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
    return Program([InsPredicate((lambda c, i: c in pattern.chars))])


@_pattern_to_program.register(PNotInChars)
def _pnotinchars_to_program(pattern: PNotInChars) -> Program:
    return Program([InsPredicate((lambda c, i: c not in pattern.chars))])


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
        import pdb; pdb.set_trace()
        prog = Program([
            # fork over
            prog0,
            # jump back
        ])
        prog.sub = [InsForkLower(program=prog, to_ending=True)] + prog.sub
        prog.sub = prog.sub + [InsJump(program=prog)]

    else:
        raise NotImplementedError('repeat exact times are not supported yet')

    return prog


@_pattern_to_program.register(PAdjacent)
def _padjacent_to_program(pattern: PAdjacent) -> Program:
    return Program([_pattern_to_program(p) for p in pattern.patterns])
