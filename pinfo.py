
import time
import sys


TIER = 0


class Task:

    def __init__(self, label=None, tier=None):
        self.start = time.time()
        self.size = 0

        global TIER
        self.tier = tier if tier is not None else TIER
        TIER += 1

        if label is not None:
            print(self.__base() + label)

    def __base(self):
        return(
            (" " * 4 * (self.tier - 1) if self.tier > 1 else "") +
            ("  | " if self.tier > 0 else "")
        )

    def stop(self, label, *objects):
        self.duration = time.time() - self.start

        for obj in objects:
            self.size += sys.getsizeof(obj)

        base = self.__base()

        if len(objects) == 0:
            print(
                "{b}[{t:.2f}s] {label}"
                .format(b=base, t=self.duration, label=label))
        else:
            print(
                "{b}[{t:.2f}s | {s:.2f}MB] {label}"
                .format(
                    b=base, t=self.duration,
                    label=label, s=self.size / 10**6))

        global TIER
        TIER -= 1
