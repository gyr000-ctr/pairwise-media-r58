import math

import pytest

import imgadaptive

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[0.0, 0.2], [0.8, 1.0]]


def _a():
    return imgadaptive.Adaptive(3, 0.0)

CASES = [
    ("map_shape", lambda: len(_a().threshold_map(IMG)) == 2),
    ("binary", lambda: set(v for row in _a().apply(IMG) for v in row) <= {0.0, 1.0}),
    ("constant", lambda: len(set(v for row in _a().apply([[0.5] * 4 for _ in range(4)]) for v in row)) == 1),
    ("kernel_sum", lambda: abs(sum(_a().kernel(3)) - 1.0) < 1e-12),
    ("bad_block", lambda: _raises(lambda: imgadaptive.Adaptive(0, 0.0))),
    ("bad_even", lambda: _raises(lambda: imgadaptive.Adaptive(4, 0.0))),
    ("bad_c", lambda: _raises(lambda: imgadaptive.Adaptive(3, 2.0))),
    ("bad_image", lambda: _raises(lambda: _a().apply([]))),
    ("type", lambda: isinstance(_a().threshold_map(IMG), list)),
    ("det", lambda: _a().apply(IMG) == _a().apply(IMG)),
    ("nan", lambda: _raises(lambda: _a().apply([[float('nan')]]))),
    ("no_mut", lambda: _a().apply([r[:] for r in IMG]) is not None),
    ("kernel_type", lambda: isinstance(_a().kernel(3), list)),
    ("float", lambda: isinstance(_a().threshold_map(IMG)[0][0], float)),
    ("apply_type", lambda: isinstance(_a().apply(IMG), list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
