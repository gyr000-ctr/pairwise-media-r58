import math

import pytest

import imgalphaedge

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

P = (0.8, 0.4, 0.2, 0.5)


def _a():
    return imgalphaedge.AlphaEdge()

CASES = [
    ("premultiply", lambda: abs(_a().premultiply(P)[0] - 0.4) < 1e-12),
    ("roundtrip", lambda: all(abs(x - y) <= 1.0 / 255 for x, y in zip(_a().unpremultiply(_a().premultiply(P)), P))),
    ("dilate", lambda: isinstance(_a().dilate([[(0.0, 0.0, 0.0, 1.0), (0.5, 0.5, 0.5, 0.0)]], 1), list)),
    ("dilate_fill", lambda: abs(_a().dilate([[(0.0, 0.0, 0.0, 1.0), (0.5, 0.5, 0.5, 0.0)]], 1)[0][1][0] - 0.0) < 1e-12),
    ("bad_radius", lambda: _raises(lambda: _a().dilate([[(0.0, 0.0, 0.0, 1.0)]], -1))),
    ("bad_len", lambda: _raises(lambda: _a().premultiply((0.5, 0.5, 0.5)))),
    ("bad_alpha", lambda: _raises(lambda: _a().premultiply((0.5, 0.5, 0.5, 1.5)))),
    ("zero_alpha", lambda: all(abs(v) < 1e-12 for v in _a().premultiply((0.5, 0.5, 0.5, 0.0)))),
    ("type", lambda: isinstance(_a().premultiply(P), tuple)),
    ("det", lambda: _a().premultiply(P) == _a().premultiply(P)),
    ("nan", lambda: _raises(lambda: _a().premultiply((float('nan'), 0.0, 0.0, 1.0)))),
    ("no_mut", lambda: _a().dilate([[(0.0, 0.0, 0.0, 1.0)]], 1) is not None),
    ("dilate_type", lambda: isinstance(_a().dilate([[(0.0, 0.0, 0.0, 1.0)]], 1), list)),
    ("unpremultiply_type", lambda: isinstance(_a().unpremultiply(P), tuple)),
    ("clamped", lambda: all(0.0 <= v <= 1.0 for v in _a().unpremultiply(P))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
