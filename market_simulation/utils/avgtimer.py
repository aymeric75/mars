import time

class AvgTimer:
    def __init__(self, name, every=200):
        self.name = name
        self.every = every
        self.n = 0
        self.total = 0.0

    def add(self, dt):
        self.n += 1
        self.total += dt
        if self.n % self.every == 0:
            print(f"{self.name}: avg over {self.n} calls = {self.total / self.n:.6f}s", flush=True)
