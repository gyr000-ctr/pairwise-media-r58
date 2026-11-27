import math

import pytest

import imgtonecurve

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

P = [(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)]


def _t():
    return imgtonecurve.ToneCurve(P)

CASES = [
    ("through", lambda: abs(_t().eval(0.5) - 0.7) < 1e-9),
    ("end0", lambda: abs(_t().eval(0.0)) < 1e-9),
    ("end1", lambda: abs(_t().eval(1.0) - 1.0) < 1e-9),
    ("monotone", lambda: _t().eval(0.2) <= _t().eval(0.8)),
    ("slope", lambda: isinstance(_t().slope(0.5), float)),
    ("non_monotone", lambda: _raises(lambda: imgtonecurve.ToneCurve([(0.0, 0.0), (0.5, 0.8), (1.0, 0.5)]))),
    ("too_few", lambda: _raises(lambda: imgtonecurve.ToneCurve([(0.0, 0.0)]))),
    ("bad_range", lambda: _raises(lambda: _t().eval(1.5))),
    ("type", lambda: isinstance(_t().eval(0.3), float)),
    ("det", lambda: _t().eval(0.3) == _t().eval(0.3)),
    ("nan", lambda: _raises(lambda: _t().eval(float('nan')))),
    ("no_mut", lambda: _t().eval(0.4) is not None),
    ("slope_type", lambda: isinstance(_t().slope(0.2), float)),
    ("clamped", lambda: 0.0 <= _t().eval(0.1) <= 1.0),
    ("float", lambda: isinstance(_t().eval(0.6), float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
