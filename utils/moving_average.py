class RunningAverage:
    def __init__(self):
        self.count = 0
        self.average = 0

    def add(self, value):
        self.average = (self.average * self.count + value) / (self.count + 1)
        self.count += 1

    def get(self):
        return self.average
