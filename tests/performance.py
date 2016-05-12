"""Benchmark performance of CellId.walk() and CellId.walk_fast().

This compares the performance against a C++-style while loop iteration that was
used before.
"""

from s2sphere import CellId


def default_walk(level=5):
    end = CellId.end(level)
    cell_id = CellId.begin(level)
    while cell_id != end:
        cell_id = cell_id.next()


def safe_implementation(level=5):
    for c in CellId.walk(level):
        pass


def fast_implementation(level=5):
    for c in CellId.walk_fast(level):
        pass


if __name__ == '__main__':
    import timeit
    print('CellId.walk_fast() time: {}'.format(timeit.timeit(
        "fast_implementation()",
        number=1000,
        setup="from __main__ import fast_implementation",
    )))
    print('CellId.walk() time: {}'.format(timeit.timeit(
        "safe_implementation()",
        number=1000,
        setup="from __main__ import safe_implementation",
    )))
    print('C++-style while loop time: {}'.format(timeit.timeit(
        "default_walk()",
        number=1000,
        setup="from __main__ import default_walk",
    )))
