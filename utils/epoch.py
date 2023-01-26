class Epoch:
    def __init__(self, epochs=None, interval=None, until=None, start=0):
        self.epochs = epochs
        self.interval = interval
        self.until = until
        self.start = start

        assert self.interval is None or self.interval > 0
        assert self.epochs or (self.interval and self.until)

        self.reset()

    def reset(self):
        self.index = 0 if self.epochs else self.start

    def next(self):
        if self.epochs:
            if self.index >= len(self.epochs):
                return -1
            epoch = self.epochs[self.index]
            self.index += 1
            return epoch

        self.index += self.interval

        if not self.until or self.index - 1 <= self.until:
            return self.index - 1
        return -1
