from collections import deque


class MovingAverage:
    def __init__(self, size):
        """
        :param size: Integer, size of the window for the moving average.
        """
        self.size = size
        self.queue = deque()
        self.sum = 0

    def next(self, val):
        """
        Calculate the moving average with a new value.
        :param val: New value to add to the window for the moving average calculation.
        :return: Float, the current moving average.
        """
        if len(self.queue) == self.size:
            self.sum -= self.queue.popleft()
        self.queue.append(val)
        self.sum += val
        return self.sum / len(self.queue)

    def current(self):
        """
        Returns the current moving average without adding a new value.
        :return: Float, the current moving average.
        """
        if not self.queue:  # Handle case where the queue is empty
            return 0
        return self.sum / len(self.queue)
