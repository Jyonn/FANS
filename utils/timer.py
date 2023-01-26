from typing import Dict

from utils.smart_printer import printer


class AtomTimer:
    def __init__(self, key):
        self.key = key
        self.time = 0
        self.count = 0

    def append(self, time):
        self.count += 1
        self.time += time
        return self


class Timer:
    def __init__(self):
        self.atom_timers = dict()  # type: Dict[str, AtomTimer]
        self.print = printer.Timer

    def append(self, key, time):
        if key not in self.atom_timers:
            self.atom_timers[key] = AtomTimer(key)
        self.atom_timers[key].append(time)

    def export(self):
        for key, atom_timer in self.atom_timers.items():
            self.print[key](atom_timer.time, atom_timer.time * 1000 / atom_timer.count)
