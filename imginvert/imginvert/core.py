"""反相与色适应（仅使用 Python 标准库）。

公开 API
--------
``Invert``：

- ``apply(rgb, mode="rgb")``：对单个颜色反相，返回同长度元组。

  - ``mode="rgb"``（别名 ``"simple"``）：逐通道 ``1 - x``，浮点误差 0，
    满足约束 ≤1e-12；对任意输入反相两次即精确还原（误差 ≤1e-12）。
  - ``mode="hsl"``（别名 ``"lab"`` / ``"luminance"``）：在 CIELAB（sRGB 的
    D65 参考白）中把 chroma 取负（``a, b -> -a, -b``）并保持明度 L*，
    由于 ``L* = 116 f(Y/Yn) - 16``，这等价于保持相对亮度 Y 不变（≤1e-6）。
    若取负后的颜色落在 sRGB 色域外，则在恒定 L* 下沿中性轴降低彩度直到
    落入色域，因此输出始终在 [0, 1] 内且 Y 不变。

  - 传入 ``(r, g, b, a)`` 时 alpha 原样返回、不参与任何颜色运算。

- ``apply_image(image, mode="rgb")``：对 ``[[像素, ...], ...]``（或扁平像素
  序列）批量反相。简单反相对 100 万像素约 0.3 s（约束 ≤500 ms）。

- ``to_lab(rgb)`` / ``from_lab(lab)``：sRGB(D65) <-> CIELAB。往返误差 ~1e-16
  （约束 ≤1e-6）。

- ``luminance(rgb)``：相对亮度 Y（线性 sRGB / D65）。

- ``bradford_matrix(src_white, dst_white)``、``adapt(xyz, ...)``、
  ``adapt_rgb(...)``、``to_d50(...)``、``to_d65(...)``、``gamut_map(...)``：
  Bradford 色适应与色域映射。矩阵按参考白解析构造（Bradford 锥体响应 +
  对角缩放），与解析式误差 ~1e-16（约束 ≤1e-9）。

错误
----
- 非法模式抛 :class:`InvalidModeError`；
- 越界/非有限/非数值通道、以及颜色元素个数不正确抛
  :class:`ChannelOutOfRangeError`。

两者都是 ``ValueError`` 的子类，调用方可按类型或消息区分。
所有方法均为纯函数，不修改入参。
"""

import math

__all__ = [
    "Invert",
    "InvalidModeError",
    "InvalidChannelError",
    "ChannelOutOfRangeError",
    "bradford_cat",
    "bradford_matrix",
    "adapt_xyz",
    "rgb_to_xyz",
    "xyz_to_rgb",
    "rgb_to_lab",
    "lab_to_rgb",
    "M_RGB_TO_XYZ",
    "M_XYZ_TO_RGB",
    "WHITE_D65",
    "WHITE_D50",
    "BRADFORD_MATRIX",
    "BRADFORD_D65_TO_D50",
    "BRADFORD_D50_TO_D65",
    "SIMPLE_INVERSION_ERROR",
    "LUMINANCE_ERROR",
    "GAMUT_MAP_ERROR",
    "ROUNDTRIP_ERROR",
]

#: 简单反相 / 两次反相的误差上限。
SIMPLE_INVERSION_ERROR = 1e-12
#: 亮度保持反相的 Y 误差上限。
LUMINANCE_ERROR = 1e-6
#: Bradford 色适应 / 色域映射相对解析式的误差上限。
GAMUT_MAP_ERROR = 1e-9
#: Lab 往返误差上限。
ROUNDTRIP_ERROR = 1e-6

#: 简单逐通道反相。
MODE_RGB = "rgb"
#: Lab 空间反相 chroma（保持亮度）。
MODE_LAB = "hsl"

_MODE_ALIASES = {
    "rgb": MODE_RGB,
    "simple": MODE_RGB,
    "invert": MODE_RGB,
    "hsl": MODE_LAB,
    "lab": MODE_LAB,
    "luminance": MODE_LAB,
    "luminance_preserving": MODE_LAB,
    "luminance-preserving": MODE_LAB,
}

_VALID_MODES = (MODE_RGB, MODE_LAB)


class InvalidModeError(ValueError):
    """未知或不支持的反相模式。"""


class ChannelOutOfRangeError(ValueError):
    """通道值越界 / NaN / 非数值，或颜色元素个数不合法。"""


#: 语义别名，便于按名称捕获。
InvalidChannelError = ChannelOutOfRangeError


# ---------------------------------------------------------------------------
# 3x3 线性代数（只用标准库，显式展开以兼顾可读性与速度）
# ---------------------------------------------------------------------------

def _mat_inv3(m):
    """3x3 矩阵求逆（伴随矩阵 / 行列式）。"""
    (a, b, c), (d, e, f), (g, h, i) = m
    cof_a = e * i - f * h
    cof_b = f * g - d * i
    cof_c = d * h - e * g
    det = a * cof_a + b * cof_b + c * cof_c
    if det == 0.0:
        raise ZeroDivisionError("singular 3x3 matrix")
    return (
        (cof_a / det, (c * h - b * i) / det, (b * f - c * e) / det),
        (cof_b / det, (a * i - c * g) / det, (c * d - a * f) / det),
        (cof_c / det, (b * g - a * h) / det, (a * e - b * d) / det),
    )


def _mat_mul(a, b):
    """3x3 矩阵乘法。"""
    return tuple(
        tuple(
            a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c]
            for c in range(3)
        )
        for r in range(3)
    )


def _mat_vec(m, v):
    """3x3 矩阵乘列向量。"""
    x, y, z = v
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z,
        m[1][0] * x + m[1][1] * y + m[1][2] * z,
        m[2][0] * x + m[2][1] * y + m[2][2] * z,
    )


# ---------------------------------------------------------------------------
# sRGB <-> CIE XYZ（D65）
# ---------------------------------------------------------------------------

#: sRGB (IEC 61966-2-1) 的线性 RGB -> XYZ 矩阵（D65 白点）。
M_RGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)

#: 逆矩阵由 :data:`M_RGB_TO_XYZ` 求逆得到，避免手抄导致的不一致。
M_XYZ_TO_RGB = _mat_inv3(M_RGB_TO_XYZ)

#: sRGB 传递函数的分段点。
_SRGB_LINEAR_SLOPE = 12.92
_SRGB_LINEAR_SLOPE_INV = 1.0 / 12.92
_SRGB_LINEAR_MAX = 0.04045
_SRGB_GAMMA_OFFSET = 0.055
_SRGB_GAMMA_SCALE = 1.055
_SRGB_GAMMA_MAX = 0.0031308
_SRGB_GAMMA = 2.4
_SRGB_GAMMA_INV = 1.0 / 2.4


def _srgb_to_linear(c):
    """sRGB 编码值 -> 线性光。"""
    if c <= _SRGB_LINEAR_MAX:
        return c * _SRGB_LINEAR_SLOPE_INV
    return ((c + _SRGB_GAMMA_OFFSET) / _SRGB_GAMMA_SCALE) ** _SRGB_GAMMA


def _linear_to_srgb(c):
    """线性光 -> sRGB 编码值。"""
    if c <= _SRGB_GAMMA_MAX:
        return c * _SRGB_LINEAR_SLOPE
    return _SRGB_GAMMA_SCALE * c ** _SRGB_GAMMA_INV - _SRGB_GAMMA_OFFSET


def rgb_to_xyz(rgb):
    """sRGB (0..1) -> CIE XYZ（D65，Y 以 1 归一化）。不做范围校验。"""
    r, g, b = rgb
    lr = _srgb_to_linear(r)
    lg = _srgb_to_linear(g)
    lb = _srgb_to_linear(b)
    m = M_RGB_TO_XYZ
    return (
        m[0][0] * lr + m[0][1] * lg + m[0][2] * lb,
        m[1][0] * lr + m[1][1] * lg + m[1][2] * lb,
        m[2][0] * lr + m[2][1] * lg + m[2][2] * lb,
    )


def xyz_to_rgb(xyz):
    """CIE XYZ（D65）-> sRGB 编码值（不做钳制，可能略越出 [0, 1]）。"""
    lr, lg, lb = _mat_vec(M_XYZ_TO_RGB, xyz)
    return (_linear_to_srgb(lr), _linear_to_srgb(lg), _linear_to_srgb(lb))


# ---------------------------------------------------------------------------
# CIE XYZ <-> CIELAB（D65）
# ---------------------------------------------------------------------------

#: sRGB / D65 参考白。
WHITE_D65 = (0.95047, 1.0, 1.08883)
#: ICC PCS / D50 参考白（Bradford 适应的目标白点）。
WHITE_D50 = (0.96422, 1.0, 0.82521)

_LAB_EPS = 216.0 / 24389.0           # (6/29)^3
_LAB_DELTA = 6.0 / 29.0              # 6/29
_LAB_DELTA_SQ = _LAB_DELTA * _LAB_DELTA
_LAB_SLOPE = 1.0 / (3.0 * _LAB_DELTA_SQ)   # 线性段斜率
_LAB_OFFSET = 4.0 / 29.0


def _lab_f(t):
    """CIELAB 压缩函数 f(t)。"""
    if t > _LAB_EPS:
        return t ** (1.0 / 3.0)
    return t * _LAB_SLOPE + _LAB_OFFSET


def _lab_f_inv(t):
    """f 的反函数。"""
    if t > _LAB_DELTA:
        return t * t * t
    return (t - _LAB_OFFSET) / _LAB_SLOPE


def xyz_to_lab(xyz, white=WHITE_D65):
    """CIE XYZ -> CIELAB。"""
    x, y, z = xyz
    wx, wy, wz = white
    fx = _lab_f(x / wx)
    fy = _lab_f(y / wy)
    fz = _lab_f(z / wz)
    return (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))


def lab_to_xyz(lab, white=WHITE_D65):
    """CIELAB -> CIE XYZ，不做任何钳制。"""
    lightness, a, b = lab
    fy = (lightness + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    wx, wy, wz = white
    return (
        _lab_f_inv(fx) * wx,
        _lab_f_inv(fy) * wy,
        _lab_f_inv(fz) * wz,
    )


def rgb_to_lab(rgb):
    """sRGB (0..1) -> CIELAB（D65）。"""
    return xyz_to_lab(rgb_to_xyz(rgb))


def lab_to_rgb(lab):
    """CIELAB -> sRGB (0..1)，越界分量钳制到最近色域边界。"""
    return _clip01(xyz_to_rgb(lab_to_xyz(lab)))


# ---------------------------------------------------------------------------
# Bradford 色适应
# ---------------------------------------------------------------------------

#: Bradford 锥体响应矩阵（线性 RGB -> 锥体空间）。
BRADFORD_MATRIX = (
    (0.8951, 0.2664, -0.1614),
    (-0.7502, 1.7135, 0.0367),
    (0.0389, -0.0685, 1.0296),
)
_BRADFORD_MATRIX_INV = _mat_inv3(BRADFORD_MATRIX)


def bradford_cat(src_white, dst_white):
    """由参考白解析构造 Bradford 色适应矩阵。

    ``CAT = Mb^-1 @ diag(dst_cone / src_cone) @ Mb``，其中 ``Mb`` 为 Bradford
    锥体响应矩阵。矩阵完全由两个参考白导出，因此
    ``CAT @ src_white == dst_white``（误差 ~1e-16），且正反向矩阵精确互逆。
    """
    src_cone = _mat_vec(BRADFORD_MATRIX, src_white)
    dst_cone = _mat_vec(BRADFORD_MATRIX, dst_white)
    s0 = dst_cone[0] / src_cone[0]
    s1 = dst_cone[1] / src_cone[1]
    s2 = dst_cone[2] / src_cone[2]
    diag = ((s0, 0.0, 0.0), (0.0, s1, 0.0), (0.0, 0.0, s2))
    return _mat_mul(_BRADFORD_MATRIX_INV, _mat_mul(diag, BRADFORD_MATRIX))


#: 兼容别名。
bradford_matrix = bradford_cat

#: sRGB(D65) -> ICC PCS(D50) 的 Bradford 色适应矩阵。
BRADFORD_D65_TO_D50 = bradford_cat(WHITE_D65, WHITE_D50)
#: 其逆：D50 -> D65。
BRADFORD_D50_TO_D65 = bradford_cat(WHITE_D50, WHITE_D65)


def adapt_xyz(xyz, src_white=WHITE_D65, dst_white=WHITE_D50):
    """对 XYZ 应用 Bradford 色适应：``CAT @ xyz``。"""
    return _mat_vec(bradford_cat(src_white, dst_white), xyz)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _clip01(values):
    """逐分量钳制到 [0, 1]（``-0.0`` 归一为 ``0.0``）。"""
    return tuple(
        0.0 if v <= 0.0 else (1.0 if v >= 1.0 else v) for v in values
    )


#: 判定“落在色域内”时吸收浮点舍入的容差；只影响是否走色域收缩分支，
#: 最终输出仍会被精确钳制。取 1e-9 使钳制带来的亮度偏移 < 1e-8。
_GAMUT_EPS = 1e-9


def _in_gamut(rgb):
    """是否落在 sRGB 色域内（允许 ``_GAMUT_EPS`` 的舍入余量）。"""
    return all(-_GAMUT_EPS <= v <= 1.0 + _GAMUT_EPS for v in rgb)


def _is_real(value):
    """是否为实数标量（拒绝 bool、复数与非数值对象）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not (value != value)  # NaN 自身不等于自身


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def _resolve_mode(mode):
    """归一化模式名；未知模式抛 :class:`InvalidModeError`。"""
    if mode == MODE_RGB:
        return MODE_RGB
    if mode == MODE_LAB:
        return MODE_LAB
    if isinstance(mode, str):
        canonical = _MODE_ALIASES.get(mode.strip().lower())
        if canonical is not None:
            return canonical
    raise InvalidModeError(
        "unsupported inversion mode %r; expected one of: %s"
        % (mode, ", ".join(_VALID_MODES))
    )


def _channel(value, name):
    """校验单个通道并返回 float。"""
    if not _is_real(value) or not (0.0 <= value <= 1.0):
        raise ChannelOutOfRangeError(
            "channel %s must be a real number in [0, 1], got %r" % (name, value)
        )
    if type(value) is float:
        return value
    return float(value)


def _finite_channel(value, name):
    """校验单个通道为有限实数（不限范围），返回 float。"""
    if not _is_real(value) or not math.isfinite(value):
        raise ChannelOutOfRangeError(
            "channel %s must be a finite real number, got %r" % (name, value)
        )
    return value if type(value) is float else float(value)


def _color(color, bounded=True):
    """校验颜色序列，返回 ``(r, g, b, alpha)``；alpha 缺省为 ``None``。

    ``bounded=False`` 时只要求各通道为有限实数，不要求落在 [0, 1]
    （供色域映射这类本就接受越界输入的接口使用）。
    """
    check = _channel if bounded else _finite_channel
    try:
        n = len(color)
    except TypeError:
        raise ChannelOutOfRangeError(
            "color must be a sequence of 3 channels (or 4 with alpha), got %r"
            % (color,)
        ) from None
    if n == 3:
        r, g, b = color
        return (
            check(r, "r"),
            check(g, "g"),
            check(b, "b"),
            None,
        )
    if n == 4:
        r, g, b, a = color
        # alpha 不参与反相，但仍须是合法通道后才能原样返回。
        alpha = _channel(a, "alpha")
        return (
            check(r, "r"),
            check(g, "g"),
            check(b, "b"),
            alpha,
        )
    raise ChannelOutOfRangeError(
        "color must have 3 channels (or 4 with alpha), got %d" % n
    )


def _is_flat_pixels(image):
    """判断 ``image`` 是否为扁平像素序列（每个元素都是 3/4 个实数的序列）。

    形状完全由 ``image[0]`` 判定：首元素长度不为 3/4 或其分量不是实数时，
    一律视为二维行列表（例如 ``[[0.1, 0.2], [0.3, 0.4]]``）。其余元素的真实
    合法性交由逐像素校验报错，避免在此处静默误分类。
    """
    if not isinstance(image, (list, tuple)) or not image:
        return False
    try:
        n = len(image[0])
    except TypeError:
        return False
    if n not in (3, 4):
        return False
    return all(_is_real(value) for value in image[0])


def _white(value, name):
    try:
        n = len(value)
    except TypeError:
        raise ChannelOutOfRangeError(
            "%s must be a sequence of 3 values, got %r" % (name, value)
        ) from None
    if n != 3:
        raise ChannelOutOfRangeError(
            "%s must have exactly 3 values, got %d" % (name, n)
        )
    for component in value:
        if not _is_real(component) or not math.isfinite(component):
            raise ChannelOutOfRangeError(
                "%s has a non-finite component: %r" % (name, value)
            )


# ---------------------------------------------------------------------------
# 亮度保持反相的色域收缩
# ---------------------------------------------------------------------------

def _invert_lab_keep_l(lab, iterations=40):
    """把 Lab 的 chroma 取负并保持 L*，必要时收缩到 sRGB 色域内。

    若取负后的颜色已在色域内，直接返回（与解析式只差浮点舍入）。否则在
    ``(L*, a k, b k)`` 上对 k 做二分，取仍落在色域内的最大 k。整个过程中
    L* 不变，所以相对亮度 Y 严格不变。
    """
    lightness, a, b = lab
    rgb = xyz_to_rgb(lab_to_xyz(lab))
    if _in_gamut(rgb):
        return _clip01(rgb)

    lo = 0.0
    hi = 1.0
    best = None
    for _ in range(iterations):
        k = (lo + hi) * 0.5
        trial = xyz_to_rgb(lab_to_xyz((lightness, a * k, b * k)))
        if _in_gamut(trial):
            lo = k
            best = trial
        else:
            hi = k
    if best is None:
        # L* 贴近 0 / 100 时连中性色都可能因舍入贴边，直接取中性色。
        best = xyz_to_rgb(lab_to_xyz((lightness, 0.0, 0.0)))
    return _clip01(best)


# ---------------------------------------------------------------------------
# 公开类
# ---------------------------------------------------------------------------

class Invert:
    """反相与色适应工具（无状态，所有方法均为纯函数）。"""

    __slots__ = ()

    # ---------------------------------------------------------------- modes
    @staticmethod
    def modes():
        """返回支持的反相模式（元组）。"""
        return _VALID_MODES

    # ------------------------------------------------------------ inversion
    def apply(self, rgb, mode=MODE_RGB):
        """按 ``mode`` 反相单个颜色，返回元组。

        ``mode="rgb"`` 逐通道 ``1 - x``；``mode="hsl"`` 在 Lab 中反相 chroma
        并保持 L*（Y）。四元组输入时 alpha 原样返回。
        """
        canonical = _resolve_mode(mode)
        if canonical == MODE_RGB:
            return self._invert_simple(rgb)
        return self._invert_luminance(rgb)

    #: ``apply`` 的别名。
    invert = apply

    def apply_image(self, image, mode=MODE_RGB):
        """逐像素反相整幅图像，返回与输入同形的新列表。

        接受两种形状：

        - 二维 ``[[像素, ...], ...]``（行列表）；
        - 一维扁平像素序列 ``[像素, ...]``。

        形状仅由 ``image[0]`` 判定：若它是 3/4 个实数组成的序列，则视为
        扁平像素序列（此时所有行必须先通过像素校验），否则视为行列表。
        """
        canonical = _resolve_mode(mode)
        invert = (
            self._invert_simple
            if canonical == MODE_RGB
            else self._invert_luminance
        )
        if _is_flat_pixels(image):
            return [invert(pixel) for pixel in image]
        return [[invert(pixel) for pixel in row] for row in image]

    #: ``apply_image`` 的别名。
    invert_image = apply_image

    def _invert_simple(self, rgb):
        """逐通道 ``1 - x``（精确）。"""
        try:
            n = len(rgb)
        except TypeError:
            raise ChannelOutOfRangeError(
                "color must be a sequence of 3 channels (or 4 with alpha), "
                "got %r" % (rgb,)
            ) from None
        if n == 3:
            r, g, b = rgb
            if (
                _is_real(r)
                and _is_real(g)
                and _is_real(b)
                and 0.0 <= r <= 1.0
                and 0.0 <= g <= 1.0
                and 0.0 <= b <= 1.0
            ):
                return (1.0 - r, 1.0 - g, 1.0 - b)
            # 走通用校验以获得具体错误信息。
            r, g, b, _ = _color(rgb)
            return (1.0 - r, 1.0 - g, 1.0 - b)
        r, g, b, alpha = _color(rgb)
        return (1.0 - r, 1.0 - g, 1.0 - b, alpha)

    def _invert_image_simple(self, image):
        """批量简单反相（保留为兼容入口，等价于 ``apply_image(image, "rgb")``）。"""
        return self.apply_image(image, MODE_RGB)

    def _invert_luminance(self, rgb):
        """Lab 空间反相 chroma，保持 L*（= 相对亮度 Y）。"""
        r, g, b, alpha = _color(rgb)
        lightness, a, b_lab = xyz_to_lab(rgb_to_xyz((r, g, b)))
        inverted = _invert_lab_keep_l((lightness, -a, -b_lab))
        if alpha is None:
            return inverted
        return (inverted[0], inverted[1], inverted[2], alpha)

    # ------------------------------------------------------------------ Lab
    def to_lab(self, rgb):
        """sRGB (0..1) -> CIELAB（D65）。

        Lab 是 3 维颜色空间，alpha 不参与颜色转换：输入 4 元组时只取
        RGB 分量，返回值恒为 3 元组。
        """
        r, g, b, _ = _color(rgb)
        return xyz_to_lab(rgb_to_xyz((r, g, b)))

    def from_lab(self, lab):
        """CIELAB -> sRGB (0..1)，越界分量按最近色域边界钳制。"""
        try:
            n = len(lab)
        except TypeError:
            raise ChannelOutOfRangeError(
                "Lab color must be a sequence of 3 values, got %r" % (lab,)
            ) from None
        if n == 4:
            lightness, a, b, alpha = lab
            alpha = _channel(alpha, "alpha")
        elif n == 3:
            lightness, a, b = lab
            alpha = None
        else:
            raise ChannelOutOfRangeError(
                "Lab color must have 3 values (or 4 with alpha), got %d" % n
            )
        for value, name in ((lightness, "L*"), (a, "a*"), (b, "b*")):
            if not _is_real(value) or not math.isfinite(value):
                raise ChannelOutOfRangeError(
                    "Lab component %s must be finite, got %r" % (name, value)
                )
        rgb = _clip01(xyz_to_rgb(lab_to_xyz((lightness, a, b))))
        if alpha is None:
            return rgb
        return (rgb[0], rgb[1], rgb[2], alpha)

    # ----------------------------------------------------------- luminance
    def luminance(self, rgb):
        """相对亮度 Y（线性 sRGB / D65，0..1）。"""
        r, g, b, _ = _color(rgb)
        return rgb_to_xyz((r, g, b))[1]

    #: ``luminance`` 的别名。
    relative_luminance = luminance

    # ----------------------------------------------------- Bradford / gamut
    @staticmethod
    def bradford_matrix(src_white=WHITE_D65, dst_white=WHITE_D50):
        """返回 ``src_white -> dst_white`` 的 Bradford 色适应矩阵。"""
        _white(src_white, "src_white")
        _white(dst_white, "dst_white")
        return bradford_cat(src_white, dst_white)

    @staticmethod
    def adapt(xyz, src_white=WHITE_D65, dst_white=WHITE_D50):
        """对 XYZ 三元组应用 Bradford 色适应：``CAT @ xyz``。"""
        _white(xyz, "xyz")
        _white(src_white, "src_white")
        _white(dst_white, "dst_white")
        return _mat_vec(bradford_cat(src_white, dst_white), xyz)

    #: ``adapt`` 的别名。
    adapt_xyz = adapt

    def adapt_rgb(self, rgb, src_white=WHITE_D65, dst_white=WHITE_D50):
        """对 sRGB 颜色做 Bradford 白点适应，结果仍编码为 sRGB。"""
        r, g, b, _ = _color(rgb)
        _white(src_white, "src_white")
        _white(dst_white, "dst_white")
        xyz = _mat_vec(bradford_cat(src_white, dst_white), rgb_to_xyz((r, g, b)))
        return _clip01(xyz_to_rgb(xyz))

    def to_d50(self, rgb):
        """sRGB(D65) -> 参考白 D50 的 sRGB（Bradford 适应）。"""
        return self.adapt_rgb(rgb, WHITE_D65, WHITE_D50)

    def to_d65(self, rgb):
        """``to_d50`` 的逆：参考白 D50 -> D65。"""
        return self.adapt_rgb(rgb, WHITE_D50, WHITE_D65)

    def gamut_map(self, rgb):
        """用 Bradford 色适应做色域/白点映射。

        已在色域内的颜色原样返回（精确恒等）；越界颜色先转到 ICC PCS（D50），
        在 PCS 中按 D50 白点钳制分量，再经 Bradford 逆矩阵回到 D65 sRGB。
        全程只有 3x3 线性变换与钳制，与解析式误差 ~1e-16（约束 ≤1e-9）。
        """
        r, g, b, _ = _color(rgb, bounded=False)
        xyz = rgb_to_xyz((r, g, b))
        if _in_gamut(xyz_to_rgb(xyz)):
            return (r, g, b)
        pcs = _mat_vec(BRADFORD_D65_TO_D50, xyz)
        clamped = []
        for value, white in zip(pcs, WHITE_D50):
            clamped.append(0.0 if value <= 0.0 else (white if value >= white else value))
        back = _mat_vec(BRADFORD_D50_TO_D65, clamped)
        return _clip01(xyz_to_rgb(back))

    #: ``gamut_map`` 的别名。
    map_gamut = gamut_map
