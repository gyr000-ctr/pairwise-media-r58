"""imgtonecurve 核心 API：控制点单调三次样条（Fritsch–Carlson / PCHIP）。

仅使用 Python 标准库。单调（非递减）控制点保证产生单调曲线，
曲线精确通过所有控制点，且一阶导数连续（C1）。
"""

import bisect
import math

__all__ = ["ToneCurve", "ToneCurveError", "NonMonotonePointsError"]


class ToneCurveError(ValueError):
    """色调曲线参数/求值错误的基类（可区分于普通 ValueError）。"""


class NonMonotonePointsError(ToneCurveError):
    """控制点纵坐标非单调（非递减）时抛出，可与其他错误区分。"""


class ToneCurve:
    """单调三次 Hermite 样条色调曲线。

    points: [(x, y), ...]，x 严格递增，y 非递减。
    require_endpoints: 为 True 时要求首点为 (0, 0)、末点为 (1, 1)。
    """

    def __init__(self, points, require_endpoints=True):
        try:
            pts = [(float(x), float(y)) for x, y in points]
        except (TypeError, ValueError) as exc:
            raise ToneCurveError("控制点必须是 (x, y) 数值对") from exc

        n = len(pts)
        if n < 2:
            raise ToneCurveError("至少需要 2 个控制点")

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        for v in xs + ys:
            if math.isnan(v) or math.isinf(v):
                raise ToneCurveError("控制点坐标必须是有限数值")

        for i in range(n - 1):
            if not xs[i + 1] > xs[i]:
                raise ToneCurveError("控制点 x 必须严格递增")

        for i in range(n - 1):
            if ys[i + 1] < ys[i]:
                raise NonMonotonePointsError(
                    "控制点 y 必须单调非递减（索引 %d: %r > %r）"
                    % (i, ys[i], ys[i + 1])
                )

        if require_endpoints:
            if xs[0] != 0.0 or ys[0] != 0.0:
                raise ToneCurveError("首控制点必须为 (0, 0)")
            if xs[-1] != 1.0 or ys[-1] != 1.0:
                raise ToneCurveError("末控制点必须为 (1, 1)")

        self._xs = xs
        self._ys = ys
        self._n = n

        # 区间长度与差商
        h = [xs[i + 1] - xs[i] for i in range(n - 1)]
        d = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]

        # Fritsch–Carlson 斜率（加权调和平均，保证单调性）
        m = [0.0] * n
        m[0] = d[0]
        m[n - 1] = d[n - 2]
        for i in range(1, n - 1):
            if d[i - 1] <= 0.0 or d[i] <= 0.0:
                m[i] = 0.0
            else:
                w1 = 2.0 * h[i] + h[i - 1]
                w2 = h[i] + 2.0 * h[i - 1]
                m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])

        # 每段预计算关于 t=(x-x_i)/h_i 的三次多项式系数：
        # y(t) = a t^3 + b t^2 + c t + dd
        self._coefs = []
        for i in range(n - 1):
            hi = h[i]
            dy = ys[i + 1] - ys[i]
            a = hi * (m[i] + m[i + 1]) - 2.0 * dy
            b = 3.0 * dy - hi * (2.0 * m[i] + m[i + 1])
            c = hi * m[i]
            self._coefs.append((a, b, c, ys[i], hi))
        self._coefs = tuple(self._coefs)

    def _check_x(self, x):
        try:
            x = float(x)
        except (TypeError, ValueError) as exc:
            raise ToneCurveError("x 必须是数值") from exc
        if math.isnan(x):
            raise ToneCurveError("x 不能为 NaN")
        xs = self._xs
        if x < xs[0] or x > xs[-1]:
            raise ToneCurveError("x=%r 超出定义域 [%r, %r]" % (x, xs[0], xs[-1]))
        return x

    def eval(self, x):
        """求曲线在 x 处的值。"""
        x = self._check_x(x)
        xs = self._xs
        i = bisect.bisect_right(xs, x) - 1
        if i >= self._n - 1:
            i = self._n - 2
        a, b, c, dd, hi = self._coefs[i]
        t = (x - xs[i]) / hi
        return ((a * t + b) * t + c) * t + dd

    def slope(self, x):
        """求曲线在 x 处的一阶导数。"""
        x = self._check_x(x)
        xs = self._xs
        i = bisect.bisect_right(xs, x) - 1
        if i >= self._n - 1:
            i = self._n - 2
        a, b, c, dd, hi = self._coefs[i]
        t = (x - xs[i]) / hi
        return ((3.0 * a * t + 2.0 * b) * t + c) / hi
