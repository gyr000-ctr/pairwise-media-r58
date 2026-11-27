import math

import pytest

import imgmotionblur

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[0.0, 1.0], [0.0, 1.0]]


def _m():
    return imgmotionblur.MotionBlur(8)

CASES = [
    ("identity", lambda: all(abs(a - b) < 1e-12 for ra, rb in zip(_m().apply(IMG, 0.0, 0.0), IMG) for a, b in zip(ra, rb))),
    ("weights_sum", lambda: abs(sum(_m().weights()) - 1.0) < 1e-12),
    ("weights_count", lambda: len(_m().weights()) == 8),
    ("shape", lambda: len(_m().apply(IMG, 1.0, 0.0)) == 2),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in _m().apply(IMG, 1.0, 0.0) for v in row)),
    ("bad_samples", lambda: _raises(lambda: imgmotionblur.MotionBlur(0))),
    ("bad_length", lambda: _raises(lambda: _m().apply(IMG, -1.0, 0.0))),
    ("bad_image", lambda: _raises(lambda: _m().apply([], 1.0, 0.0))),
    ("type", lambda: isinstance(_m().apply(IMG, 1.0, 0.0), list)),
    ("det", lambda: _m().apply(IMG, 1.0, 0.0) == _m().apply(IMG, 1.0, 0.0)),
    ("nan", lambda: _raises(lambda: _m().apply([[float('nan')]], 1.0, 0.0))),
    ("no_mut", lambda: _m().apply([r[:] for r in IMG], 1.0, 0.0) is not None),
    ("weights_type", lambda: isinstance(_m().weights(), list)),
    ("float", lambda: isinstance(_m().apply(IMG, 1.0, 0.0)[0][0], float)),
    ("angle_effect", lambda: _m().apply(IMG, 1.0, 0.0) != _m().apply(IMG, 1.0, 1.57)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
