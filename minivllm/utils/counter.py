

class Counter:
    def __init__(self, start: int = 0) -> None:
        super().__init__()

        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0