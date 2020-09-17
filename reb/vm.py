"""Implement Regular Expression with vm"""


from typing import Iterator, List, Optional

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
    def __init__(self, pc: int, sp: int, starter: int, groups=None):
        self.pc: int = pc  # Program Counter
        self.sp: int = sp  # String pointer
        self.starter: int = starter  # where does the match started in the current string
        self.groups = list(groups) or []  # group start end end marks

        # Priority double ended link list
        self.prio_former: 'Thread' = None
        self.prio_later: 'Thread' = None

    def to_ptnode(self) -> Optional[PTNode]:
        # TODO
        pass


class Mark(object):
    def __init__(self, index: int, name, is_open: bool, depth: int):
        self.index: int = index
        self.name = name
        self.is_open: bool = bool(is_open)
        self.is_close: bool = not self.is_open
        self.depth: int = depth


class Finder(object):
    def finditer(self, program: List[Instruction], text: str) -> Iterator[PTNode]:
        # TODO execute instructions on text and generate PTNodes

        # mapping threads pc to threads
        thread_map = {}

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
            if thread_map[thread.pc] is thread:
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
                    return
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
            th = put_thread(th, pc=th.pc, expel=False)
            if th:
                move_thread_higher(th, than=cur_lo)

            # as long as the ready ll is not empty
            while cur_hi.prio_later is not cur_lo:
                # pick the ready thread with highest prioity and run it
                th = cur_hi.prio_later
                ins = program[th.pc]
                if isinstance(ins, InsStart):
                    put_thread(th, pc=th.pc + 1, expel=True)
                elif isinstance(ins, InsSuccess):
                    # find a match, arrange it to be PTNodes, and re-init threads
                    yield th.to_ptnode()
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
                    th1 = Thread(pc=ins.to, sp=index, starter=th.starter, groups=th.groups)
                    th1 = put_thread(th1, pc=th1.pc, expel=True)
                    if th1:
                        move_thread_higher(th1, than=th)
                    put_thread(th, pc=th.pc + 1)
                elif isinstance(ins, InsForkLower):
                    th1 = Thread(pc=ins.to, sp=index, starter=th.starter, groups=th.groups)
                    th1 = put_thread(th1, pc=th1.pc, expel=True)
                    if th1:
                        move_thread_lower(th1, than=th)
                    put_thread(th, pc=th.pc + 1)
                elif isinstance(ins, InsJump):
                    put_thread(ins, pc=ins.to, expel=True)
                elif isinstance(ins, InsGroupStart):
                    th.groups.append(Mark(index=ins.index, name=ins.group_id, is_open=True, depth=0))  # TODO depth
                    put_thread(ins, pc=th.pc + 1, expel=True)
                elif isinstance(ins, InsGroupEnd):
                    th.groups.append(Mark(index=ins.index, name=ins.group_id, is_open=False, depth=0))  # TODO depth
                    put_thread(ins, pc=th.pc + 1, expel=True)
                elif isinstance(ins, InsPredicate):
                    if ins.pred(char, index=index):
                        put_thread(th, pc=th.pc + 1, expel=True)
                        move_thread_higher(th, than=nxt_lo)
                    else:
                        del_thread(th)

            cur_hi, cur_lo, nxt_hi, nxt_lo = nxt_hi, nxt_lo, cur_hi, cur_lo


def compile_pattern(pattern: Pattern) -> Finder:
    # TODO Turn a pattern into a series of instructions
    pass
