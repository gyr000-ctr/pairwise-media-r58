"""自适应阈值（局部均值 / 高斯加权均值），仅使用 Python 标准库。

公开 API：

- ``Adaptive(block, c, method="mean")``：构造自适应阈值器。
  ``block`` 必须为正奇数；``c`` 为有限数且 ``0 <= c <= 1``；
  ``method`` 取 ``"mean"``（等价 ``"box"``/``"uniform"``，积分图实现，
  每像素 O(1)）或 ``"gaussian"``（等价 ``"gauss"``，可分离高斯卷积）。
- ``kernel(n)``：长度为 ``n`` 的 1D 高斯权重，和精确为 1.0。
- ``threshold_map(image)``：与输入同形的局部（加权）均值图，纯 float。
- ``apply(image)``：二值化，输出只含 ``0.0`` / ``1.0``；像素严格大于
  局部均值减 C 时为前景（1.0）。

非法块尺寸与非法 C 抛出不同的异常类型（同为 ValueError 子类），
调用方可区分处理。
"""

import math

__all__ = [
    "Adaptive",
    "InvalidBlockSizeError",
    "InvalidBlockError",
    "InvalidCError",
    "InvalidImageError",
    "InvalidMethodError",
]

# 判定为“相等”的容差，用于吸收浮点舍入噪声，保证常量图结果确定。
_EPS = 1e-12

_MEAN_ALIASES = frozenset(("mean", "box", "uniform"))
_GAUSS_ALIASES = frozenset(("gaussian", "gauss"))

# 代码生成的高斯卷积函数缓存：block -> (水平函数, 垂直函数, 融合二值化函数)。
# 纯 Python 解释器无法对居中加权窗口做 O(1)/像素 的精确求和（积分图只适用
# 等权盒式），精确 FIR 高斯本质是 O(block)/像素。这里把核权重烘焙为常量并
# 展开内层循环，消除逐次索引/属性查找，把常见块尺寸压进时限；结果与逐点
# 加权求和参考实现仅差浮点加法结合顺序（~1e-16，远小于 1e-9）。
_PASS_CACHE = {}


class InvalidBlockSizeError(ValueError):
    """非法块尺寸（非正整数或偶数）。"""


# 语义别名，便于按不同名称捕获。
InvalidBlockError = InvalidBlockSizeError


class InvalidCError(ValueError):
    """非法 C（非数值、NaN/inf 或超出 [0, 1]）。"""


class InvalidImageError(ValueError):
    """非法图像（空、非矩形或含非有限数值）。"""


class InvalidMethodError(ValueError):
    """非法 method。"""


def _is_real_number(value):
    # bool 是 int 的子类，但语义上不是像素/参数数值，拒绝之。
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _build_pass_funcs(block, kernel):
    """生成并缓存指定 block 的水平/垂直内部像素卷积函数。

    两个函数都只处理“核完整”的内部位置（列/行 index ∈ [r, size-r)），
    边界位置由调用方按截断权重重归一化处理。核权重以 float 字面量烘焙，
    运行时无任何列表下标与边界判断。
    """
    cached = _PASS_CACHE.get(block)
    if cached is not None:
        return cached
    radius = block // 2
    literals = [format(w, ".17g") for w in kernel]
    # 水平：输出 (row, col) = 沿列方向的加权和。
    h_terms = " + ".join(
        "a[base + col + (%d)] * %s" % (j - radius, literals[j])
        for j in range(block)
    )
    horizontal_src = (
        "def f(a, out, width, height, radius):\n"
        "    for row in range(height):\n"
        "        base = row * width\n"
        "        for col in range(radius, width - radius):\n"
        "            out[base + col] = %s\n" % h_terms
    )
    # 垂直：输出 (row, col) = 沿行方向的加权和（步长 width）。
    v_terms = " + ".join(
        "a[base + (%d) * width + col] * %s" % (j - radius, literals[j])
        for j in range(block)
    )
    vertical_src = (
        "def f(a, out, width, height, radius):\n"
        "    for row in range(radius, height - radius):\n"
        "        base = row * width\n"
        "        for col in range(width):\n"
        "            out[%s] = %s\n" % ("base + col", v_terms)
    )
    namespace = {}
    exec(horizontal_src, namespace)
    horizontal = namespace["f"]
    namespace = {}
    exec(vertical_src, namespace)
    vertical = namespace["f"]
    # 融合二值化垂直遍：内部行算完高斯即比较，直接产出 0.0/1.0，
    # 省去整张阈值图与一遍独立二值化遍历（仅 apply 路径使用）。
    fused_src = (
        "def f(a, src, out, width, height, radius, c_value, eps):\n"
        "    for row in range(radius, height - radius):\n"
        "        base = row * width\n"
        "        for col in range(width):\n"
        "            g = %s\n"
        "            out[base + col] = 1.0 if src[base + col] - (g - c_value) > eps else 0.0\n"
        % v_terms
    )
    namespace = {}
    exec(fused_src, namespace)
    fused = namespace["f"]
    _PASS_CACHE[block] = (horizontal, vertical, fused)
    return horizontal, vertical, fused


class Adaptive:
    def __init__(self, block, c, method="mean"):
        if not isinstance(block, int) or isinstance(block, bool) or block <= 0 or block % 2 == 0:
            raise InvalidBlockSizeError(
                "block 必须为正奇数，收到: %r" % (block,)
            )
        if not _is_real_number(c) or not math.isfinite(c) or not (0.0 <= float(c) <= 1.0):
            raise InvalidCError("c 必须为 [0, 1] 内的有限数，收到: %r" % (c,))
        if method in _MEAN_ALIASES:
            kind = "mean"
        elif method in _GAUSS_ALIASES:
            kind = "gaussian"
        else:
            raise InvalidMethodError(
                "method 必须为 'mean' 或 'gaussian'，收到: %r" % (method,)
            )

        self.block = block
        self.c = float(c)
        self.method = kind
        self._radius = block // 2
        # 高斯方法复用同一套 1D 核（可分离）。
        self._kernel = self._build_kernel(block) if kind == "gaussian" else None

    # ------------------------------------------------------------------ kernel
    def kernel(self, n=None):
        """返回长度 n 的 1D 高斯权重（float list），归一化使和精确为 1.0。"""
        if n is None:
            n = self.block
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0 or n % 2 == 0:
            raise InvalidBlockSizeError(
                "核尺寸必须为正奇数，收到: %r" % (n,)
            )
        # 返回副本，外部修改不会影响内部缓存。
        return list(self._build_kernel(n))

    @staticmethod
    def _build_kernel(n):
        radius = n // 2
        # 与 OpenCV getGaussianKernel 一致的 sigma 选择。
        sigma = 0.3 * ((n - 1) * 0.5 - 1.0) + 0.8
        inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)
        weights = [
            math.exp(-(i * i) * inv_two_sigma2)
            for i in range(-radius, radius + 1)
        ]
        total = math.fsum(weights)
        weights = [w / total for w in weights]
        # 把归一化残差补到最大权重上，使 sum 与 1.0 精确相等（误差 0.0 < 1e-12）。
        drift = 1.0 - math.fsum(weights)
        peak = max(range(n), key=weights.__getitem__)
        weights[peak] += drift
        return weights

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def _flatten(image):
        """校验并展平为 (flat, height, width)；拒绝空/非矩形/非有限数值。

        校验与转换分离：先用紧凑循环做合法性检查（抛异常），通过后用 C 层
        列表推导完成展平与 float 转换，显著快于逐条 append。
        """
        if not isinstance(image, list) or not image:
            raise InvalidImageError("图像必须为非空 list（行列表）")
        width = len(image[0])
        if width == 0:
            raise InvalidImageError("图像行不能为空")
        for row in image:
            if not isinstance(row, list) or len(row) != width:
                raise InvalidImageError("图像必须为等宽矩形 list of list")
            for value in row:
                if not _is_real_number(value) or not math.isfinite(float(value)):
                    raise InvalidImageError("像素必须为有限数值，收到: %r" % (value,))
        flat = [float(value) for row in image for value in row]
        return flat, len(image), width

    # ----------------------------------------------------------- threshold map
    def threshold_map(self, image):
        """局部（加权）均值图，不包含 C；返回 list[list[float]]。"""
        flat, height, width = flat_hw = self._flatten(image)
        if self.method == "mean":
            out = self._mean_map(flat_hw)
        else:
            out = self._gauss_map(flat_hw)
        return [out[r * width:(r + 1) * width] for r in range(height)]

    def _mean_map(self, flat_hw):
        """积分图盒式均值：每像素仅 4 次角点查找 + 1 次除法，O(1)。"""
        flat, height, width = flat_hw
        n = height * width

        # 零填充积分图，步长 width+1：integral[r*stride + c] 为左上角
        # (0,0)→(r-1,c-1) 的和，第 0 行与第 0 列为零填充，越界角点命中填充位取 0。
        stride = width + 1
        integral = [0.0] * ((height + 1) * stride)
        src = 0
        for r in range(height):
            row_acc = 0.0
            base = (r + 1) * stride
            prev = r * stride
            for c in range(width):
                row_acc += flat[src]
                src += 1
                integral[base + c + 1] = integral[prev + c + 1] + row_acc

        radius = self._radius
        out = [0.0] * n
        for r in range(height):
            r_lo = r - radius
            if r_lo < 0:
                r_lo = 0
            r_hi = r + radius + 1
            if r_hi > height:
                r_hi = height
            top = r_lo * stride
            bottom = r_hi * stride
            area_h = r_hi - r_lo
            base_out = r * width
            for c in range(width):
                c_lo = c - radius
                if c_lo < 0:
                    c_lo = 0
                c_hi = c + radius + 1
                if c_hi > width:
                    c_hi = width
                total = (
                    integral[bottom + c_hi]
                    - integral[top + c_hi]
                    - integral[bottom + c_lo]
                    + integral[top + c_lo]
                )
                out[base_out + c] = total / (area_h * (c_hi - c_lo))
        return out

    def _gauss_horizontal(self, flat, height, width):
        """可分离高斯的水平遍：内部列 codegen，边界列截断重归一化。"""
        kernel = self._kernel
        radius = self._radius
        n = height * width
        horizontal_pass, _, _ = _build_pass_funcs(self.block, kernel)

        horizontal = [0.0] * n
        horizontal_pass(flat, horizontal, width, height, radius)
        border_cols = list(range(min(radius, width))) + list(
            range(max(radius, width - radius), width)
        )
        for row in range(height):
            base = row * width
            for col in border_cols:
                c_lo = col - radius
                if c_lo < 0:
                    c_lo = 0
                c_hi = col + radius + 1
                if c_hi > width:
                    c_hi = width
                total = 0.0
                wsum = 0.0
                k = c_lo - col + radius
                for cc in range(c_lo, c_hi):
                    w = kernel[k]
                    total += flat[base + cc] * w
                    wsum += w
                    k += 1
                horizontal[base + col] = total / wsum
        return horizontal

    def _gauss_map(self, flat_hw):
        """可分离高斯加权均值（先水平后垂直）。

        内部像素（核完整，权重和恰为 1）走代码生成的展开卷积；边界像素截断
        核并按实际权重和重归一化。图像小于核时所有像素都走边界路径，结果
        与逐点 2D 截断重归一化参考一致（仅浮点加法顺序差异，~1e-16）。
        """
        flat, height, width = flat_hw
        kernel = self._kernel
        radius = self._radius
        n = height * width
        horizontal = self._gauss_horizontal(flat, height, width)

        # ---- 垂直方向 ----
        _, vertical_pass, _ = _build_pass_funcs(self.block, kernel)
        out = [0.0] * n
        vertical_pass(horizontal, out, width, height, radius)
        border_rows = list(range(min(radius, height))) + list(
            range(max(radius, height - radius), height)
        )
        for row in border_rows:
            r_lo = row - radius
            if r_lo < 0:
                r_lo = 0
            r_hi = row + radius + 1
            if r_hi > height:
                r_hi = height
            k_lo = r_lo - row + radius
            k_hi = r_hi - row + radius
            wseg = kernel[k_lo:k_hi]
            wsum = math.fsum(wseg)
            base = row * width
            for col in range(width):
                total = 0.0
                k = 0
                for rr in range(r_lo, r_hi):
                    total += horizontal[rr * width + col] * wseg[k]
                    k += 1
                out[base + col] = total / wsum
        return out

    def _gauss_apply(self, flat_hw):
        """高斯阈值图 + 二值化融合路径（apply 专用，避免整张阈值图与二次遍历）。

        内部行用融合 codegen：算完垂直高斯即比较输出 0.0/1.0；边界行按截断
        权重算高斯后再比较。数值口径与先取 ``threshold_map`` 再二值化完全一致。
        """
        flat, height, width = flat_hw
        kernel = self._kernel
        radius = self._radius
        n = height * width
        horizontal = self._gauss_horizontal(flat, height, width)

        _, _, fused_pass = _build_pass_funcs(self.block, kernel)
        out = [0.0] * n
        fused_pass(horizontal, flat, out, width, height, radius, self.c, _EPS)
        border_rows = list(range(min(radius, height))) + list(
            range(max(radius, height - radius), height)
        )
        c_value = self.c
        for row in border_rows:
            r_lo = row - radius
            if r_lo < 0:
                r_lo = 0
            r_hi = row + radius + 1
            if r_hi > height:
                r_hi = height
            k_lo = r_lo - row + radius
            k_hi = r_hi - row + radius
            wseg = kernel[k_lo:k_hi]
            wsum = math.fsum(wseg)
            base = row * width
            for col in range(width):
                total = 0.0
                k = 0
                for rr in range(r_lo, r_hi):
                    total += horizontal[rr * width + col] * wseg[k]
                    k += 1
                g = total / wsum
                out[base + col] = (
                    1.0 if flat[base + col] - (g - c_value) > _EPS else 0.0
                )
        return out

    # -------------------------------------------------------------- binarization
    def apply(self, image):
        """二值化：像素 > 局部均值 - C 为前景 1.0，否则背景 0.0。"""
        flat, height, width = flat_hw = self._flatten(image)
        if self.method == "gaussian":
            out = self._gauss_apply(flat_hw)
            return [out[r * width:(r + 1) * width] for r in range(height)]

        means = self._mean_map(flat_hw)
        c_value = self.c
        # C 层列表推导；epsilon 吸收舍入噪声，相等归背景，保证常量图确定。
        out = [
            1.0 if flat[i] - (means[i] - c_value) > _EPS else 0.0
            for i in range(len(flat))
        ]
        return [out[r * width:(r + 1) * width] for r in range(height)]
