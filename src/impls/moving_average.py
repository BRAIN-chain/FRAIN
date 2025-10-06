from collections import deque


class MovingAverage:
    def __init__(self, size, mode="wisa"):
        """
        :param size: Integer, window size.
        :param mode: "wisa" (moving average) or "brain" (latest / window-sum)
        """
        self.size = size
        self.mode = mode
        self.queue = deque()
        self.sum = 0

    def next(self, val):
        if len(self.queue) == self.size:
            self.sum -= self.queue.popleft()
        self.queue.append(val)
        self.sum += val

        if self.mode == "brain":
            return 0 if self.sum == 0 else val / self.sum
        return self.sum / len(self.queue)

    def current(self):
        if not self.queue:
            return 0
        if self.mode == "brain":
            return 0 if self.sum == 0 else self.queue[-1] / self.sum
        return self.sum / len(self.queue)
