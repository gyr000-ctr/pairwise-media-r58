"""方向运动模糊（累积缓冲），仅使用 Python 标准库。

设计概述
--------
沿运动方向取 ``samples`` 个等距、等权的采样点（中点布局，覆盖
``[-length/2, length/2)``），权重和恒为 1（浮点误差 <= 1e-12）。采样取值
采用双线性插值，越界坐标按 clamp（边缘像素外扩）处理。离散累积即连续
（解析）线积分的等距中点求积近似。

轴向（水平/垂直）运动时双线性在运动方向退化为 1D，内层循环自动走更省
运算的分支；任意角度则使用完整 2D 双线性。

其它约定：``length == 0`` 时退化为恒等变换；非法长度与非法角度抛出可区分
的异常类型（均为 ``ValueError`` 子类，通用 ``ValueError`` 仍可捕获）。
"""

import math

__all__ = [
    "MotionBlur",
    "MotionBlurError",
    "InvalidLengthError",
    "InvalidAngleError",
    "InvalidImageError",
    "InvalidSamplesError",
]


class MotionBlurError(ValueError):
    """运动模糊参数错误基类（同时是 ValueError）。"""


class InvalidSamplesError(MotionBlurError):
    """采样数非法。"""


class InvalidLengthError(MotionBlurError):
    """模糊长度非法。"""


class InvalidAngleError(MotionBlurError):
    """模糊角度非法。"""


class InvalidImageError(MotionBlurError):
    """输入图像非法。"""


def _is_real_number(value):
    # bool 是 int 的子类，这里显式排除，避免 True/False 被当作数值。
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


class MotionBlur:
    """沿指定方向的累积缓冲运动模糊。

    Parameters
    ----------
    samples:
        沿运动方向的等距、等权采样数量，必须为正整数。权重列表长度即
        ``samples``，权重和恒为 1。
    """

    def __init__(self, samples):
        if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
            raise InvalidSamplesError(
                "samples 必须为正整数，实际为 %r" % (samples,)
            )
        self._samples = samples
        self._weights = self._build_weights(samples)

    @staticmethod
    def _build_weights(samples):
        """等权权重列表，并修正末项使权重和精确为 1。"""
        weight = 1.0 / samples
        weights = [weight] * samples
        weights[-1] = 1.0 - weight * (samples - 1)
        return weights

    def weights(self):
        """返回采样权重列表（等权、和为 1）。"""
        return list(self._weights)

    # ------------------------------------------------------------------
    # 参数校验
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_image(image):
        if not isinstance(image, (list, tuple)) or len(image) == 0:
            raise InvalidImageError("图像必须为非空的二维序列")
        width = None
        for row in image:
            if not isinstance(row, (list, tuple)) or len(row) == 0:
                raise InvalidImageError("图像的每一行必须为非空序列")
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise InvalidImageError("图像各行长度必须一致（矩形图像）")
            for value in row:
                if not _is_real_number(value):
                    raise InvalidImageError("图像像素必须为数值，发现 %r" % (value,))
                if not math.isfinite(value):
                    raise InvalidImageError(
                        "图像像素必须为有限数值，发现 %r" % (value,)
                    )
        return len(image), width

    @staticmethod
    def _validate_length(length):
        if not _is_real_number(length):
            raise InvalidLengthError("length 必须为数值，实际为 %r" % (length,))
        if not math.isfinite(length):
            raise InvalidLengthError("length 必须为有限数值，实际为 %r" % (length,))
        if length < 0.0:
            raise InvalidLengthError("length 不能为负，实际为 %r" % (length,))

    @staticmethod
    def _validate_angle(angle):
        if not _is_real_number(angle):
            raise InvalidAngleError("angle 必须为数值，实际为 %r" % (angle,))
        if not math.isfinite(angle):
            raise InvalidAngleError("angle 必须为有限数值，实际为 %r" % (angle,))

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def apply(self, image, length, angle):
        """对二维灰度图施加方向运动模糊。

        Parameters
        ----------
        image:
            二维列表（或元组）构成的灰度图，元素为有限数值。
        length:
            运动模糊长度（像素），非负。``0`` 表示恒等。
        angle:
            运动方向角（弧度），实数；0 为水平向右，pi/2 为垂直向下。

        Returns
        -------
        list[list[float]]
            与输入同尺寸的模糊结果，元素为 float。
        """
        height, width = self._validate_image(image)
        self._validate_length(length)
        self._validate_angle(angle)

        # 长度 0：恒等，返回 float 深拷贝（与输入不共享内存）。
        if length == 0.0:
            return [[float(v) for v in row] for row in image]

        return self._blur(image, height, width, length, angle)

    # ------------------------------------------------------------------
    # 等距等权采样 + 双线性（每行用列表推导加速）
    # 轴向（水平/垂直）运动时双线性退化为 1D，内层循环自动走快速分支。
    # ------------------------------------------------------------------
    def _blur(self, image, height, width, length, angle):
        taps = self._samples
        tap_weight = 1.0 / taps

        dx = math.cos(angle)
        dy = math.sin(angle)

        flat = [float(v) for row in image for v in row]
        accum = [0.0] * (height * width)

        x_indices = range(width)
        row_indices = range(height)
        w_limit = width - 1
        h_limit = height - 1

        for k in range(taps):
            # 等距中点布局：覆盖 [-length/2, length/2)。
            t = (-0.5 + (k + 0.5) / taps) * length
            ox = t * dx
            oy = t * dy

            ix = math.floor(ox)
            fx = ox - ix
            gx = 1.0 - fx
            iy = math.floor(oy)
            fy = oy - iy
            gy = 1.0 - fy

            col0 = [_clamp(x + ix, 0, w_limit) for x in x_indices]
            col1 = [_clamp(x + ix + 1, 0, w_limit) for x in x_indices]
            row0_idx = [_clamp(y + iy, 0, h_limit) for y in row_indices]
            row1_idx = [_clamp(y + iy + 1, 0, h_limit) for y in row_indices]

            if fy == 0.0:
                # 行方向无插值：退化为水平双线性。
                if fx == 0.0:
                    for y in row_indices:
                        base = y * width
                        source = flat[row0_idx[y] * width:
                                      row0_idx[y] * width + width]
                        accum[base:base + width] = [
                            acc + source[c] * tap_weight
                            for acc, c in zip(accum[base:base + width], col0)
                        ]
                else:
                    for y in row_indices:
                        base = y * width
                        source = flat[row0_idx[y] * width:
                                      row0_idx[y] * width + width]
                        accum[base:base + width] = [
                            acc + (source[c0] * gx + source[c1] * fx) * tap_weight
                            for acc, c0, c1 in zip(
                                accum[base:base + width], col0, col1)
                        ]
            elif fx == 0.0:
                # 列方向无插值：退化为垂直双线性。
                for y in row_indices:
                    base = y * width
                    source0 = flat[row0_idx[y] * width:
                                   row0_idx[y] * width + width]
                    source1 = flat[row1_idx[y] * width:
                                   row1_idx[y] * width + width]
                    accum[base:base + width] = [
                        acc + (source0[c] * gy + source1[c] * fy) * tap_weight
                        for acc, c in zip(accum[base:base + width], col0)
                    ]
            else:
                # 完整 2D 双线性：预算四角合并权重，内层只做乘加。
                w00 = gx * gy * tap_weight
                w01 = fx * gy * tap_weight
                w10 = gx * fy * tap_weight
                w11 = fx * fy * tap_weight
                for y in row_indices:
                    base = y * width
                    source0 = flat[row0_idx[y] * width:
                                   row0_idx[y] * width + width]
                    source1 = flat[row1_idx[y] * width:
                                   row1_idx[y] * width + width]
                    accum[base:base + width] = [
                        acc
                        + source0[c0] * w00
                        + source0[c1] * w01
                        + source1[c0] * w10
                        + source1[c1] * w11
                        for acc, c0, c1 in zip(
                            accum[base:base + width], col0, col1)
                    ]

        return [accum[y * width:(y + 1) * width] for y in row_indices]
