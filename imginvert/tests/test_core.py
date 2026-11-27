import math

import pytest

import imginvert

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

C = (0.2, 0.4, 0.6)


def _i():
    return imginvert.Invert()

CASES = [
    ("simple", lambda: all(abs(a - (1.0 - b)) < 1e-12 for a, b in zip(_i().apply(C, 'rgb'), C))),
    ("roundtrip", lambda: all(abs(a - b) < 1e-12 for a, b in zip(_i().apply(_i().apply(C, 'rgb'), 'rgb'), C))),
    ("hsl", lambda: isinstance(_i().apply(C, 'hsl'), tuple)),
    ("lab", lambda: len(_i().to_lab(C)) == 3),
    ("lab_roundtrip", lambda: all(abs(a - b) < 1e-6 for a, b in zip(_i().from_lab(_i().to_lab(C)), C))),
    ("bad_mode", lambda: _raises(lambda: _i().apply(C, 'bogus'))),
    ("bad_len", lambda: _raises(lambda: _i().apply((0.2, 0.4), 'rgb'))),
    ("bad_channel", lambda: _raises(lambda: _i().apply((1.5, 0.0, 0.0), 'rgb'))),
    ("type", lambda: isinstance(_i().apply(C, 'rgb'), tuple)),
    ("det", lambda: _i().apply(C, 'hsl') == _i().apply(C, 'hsl')),
    ("nan", lambda: _raises(lambda: _i().apply((float('nan'), 0.0, 0.0), 'rgb'))),
    ("no_mut", lambda: _i().apply(C, 'hsl') is not None),
    ("lab_type", lambda: isinstance(_i().to_lab(C), tuple)),
    ("range", lambda: all(0.0 <= v <= 1.0 for v in _i().apply(C, 'hsl'))),
    ("from_type", lambda: isinstance(_i().from_lab(_i().to_lab(C)), tuple)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
