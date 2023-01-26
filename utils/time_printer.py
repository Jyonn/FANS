
import time
from logging import warning

from utils.logger import Logger


class SmartPrinter:
    def __init__(self, logger: Logger = None):
        self.start_time = time.time()
        self.logger = logger

    @staticmethod
    def div_num(n, base=60):
        return n // base, n % base

    def format_second(self, second):
        second = int(second)
        minutes, second = self.div_num(second)
        hours, minutes = self.div_num(minutes)
        return '[%02d:%02d:%02d]' % (hours, minutes, second)

    def __call__(self, *args):
        delta_time = time.time() - self.start_time
        line = ' '.join(map(str, [self.format_second(delta_time), *args]))
        print(line)
        if self.logger:
            self.logger(line)

    def with_warn(self, string):
        delta_time = time.time() - self.start_time
        warning('%s %s' % (self.format_second(delta_time), string))


printer = SmartPrinter()
