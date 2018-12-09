
import time
import sys


TIER = 0


def base(tier):
    return(
        (" " * 4 * (tier - 1) if tier > 1 else "") +
        ("  | " if tier > 0 else "")
    )


def log(msg, tier=None):
    if tier is None:
        global TIER
        tier = TIER
    print(base(tier) + str(msg))


class Task:

    def __init__(self, label=None, tier=None):
        self.start = time.time()
        self.size = 0

        global TIER
        self.tier = tier if tier is not None else TIER
        TIER += 1

        if label is not None:
            print(base(self.tier) + label)

    def __base(self):
        return(
            (" " * 4 * (self.tier - 1) if self.tier > 1 else "") +
            ("  | " if self.tier > 0 else "")
        )

    def stop(self, label, *objects):
        self.duration = time.time() - self.start

        for obj in objects:
            self.size += sys.getsizeof(obj)

        b = base(self.tier)

        if len(objects) == 0:
            print(
                "{b}[{t:.2f}s] {label}"
                .format(b=b, t=self.duration, label=label))
        else:
            print(
                "{b}[{t:.2f}s | {s:.2f}MB] {label}"
                .format(
                    b=b, t=self.duration,
                    label=label, s=self.size / 10**6))

        global TIER
        TIER -= 1
