"""imgalphaedge 核心 API（仅标准库）。

约定：
- 像素为长度 4 的序列 (r, g, b, a)，各通道取值 [0.0, 1.0]，NaN/Inf 非法。
- 颜色采用直线（非预乘）RGB 表示；``premultiply`` / ``unpremultiply``
  在直线 RGB 与预乘 alpha 表示之间转换。
- ``dilate`` 以 alpha == 1.0 的不透明像素为种子，按 8 邻域 Chebyshev
  距离把最近不透明像素的颜色外扩至多 ``radius`` 像素；alpha 通道保持
  不变，alpha < 1 的像素（含 alpha == 0）绝不作为颜色来源。
"""

_BIG = 1 << 30


def _check_pixel(p):
    """校验像素为 4 个 [0, 1] 内的有限数；非法时抛 ValueError。"""
    try:
        n = len(p)
    except TypeError:
        raise ValueError("像素必须是长度为 4 的序列 (r, g, b, a)")
    if n != 4:
        raise ValueError("像素必须是长度为 4 的序列 (r, g, b, a)")
    for v in p:
        if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
            # NaN/Inf 及越界值都会使比较链为 False
            raise ValueError("像素通道取值必须是 [0, 1] 内的有限数")


class AlphaEdge:
    def premultiply(self, rgba):
        """直线 RGB -> 预乘 alpha：返回 (r*a, g*a, b*a, a)。"""
        _check_pixel(rgba)
        r, g, b, a = rgba
        return (r * a, g * a, b * a, a)

    def unpremultiply(self, rgba):
        """预乘 alpha -> 直线 RGB；alpha 为 0 时颜色取 0，结果截断到 [0, 1]。"""
        _check_pixel(rgba)
        r, g, b, a = rgba
        if a == 0.0:
            return (0.0, 0.0, 0.0, 0.0)
        inv = 1.0 / a
        r = r * inv
        g = g * inv
        b = b * inv
        return (
            0.0 if r < 0.0 else (1.0 if r > 1.0 else r),
            0.0 if g < 0.0 else (1.0 if g > 1.0 else g),
            0.0 if b < 0.0 else (1.0 if b > 1.0 else b),
            a,
        )

    def dilate(self, rgba, radius):
        """把最近不透明像素的颜色外扩到透明/半透明边缘像素。

        返回与输入同尺寸的新嵌套列表（行列表、像素元组），不修改输入。
        """
        if isinstance(radius, bool) or not isinstance(radius, int):
            raise ValueError("radius 必须是非负整数")
        if radius < 0:
            raise ValueError("radius 必须是非负整数")
        if not isinstance(rgba, list) or len(rgba) == 0:
            raise ValueError("rgba 必须是非空的二维像素列表")
        first = rgba[0]
        if not isinstance(first, list) or len(first) == 0:
            raise ValueError("rgba 必须是非空的二维像素列表")
        h = len(rgba)
        w = len(first)
        n = h * w
        is_num = isinstance
        num_types = (int, float)

        # 单趟完成结构/取值校验、展平与种子登记（不调用逐通道辅助函数）。
        # dist/src 使用与图像同尺寸的一维网格（无填充边），边界在行循环
        # 中处理，避免填充网格带来的额外内存与索引运算
        flat = [None] * n
        dist = [_BIG] * n
        src = [-1] * n
        idx = 0
        for y in range(h):
            row = rgba[y]
            if not isinstance(row, list) or len(row) != w:
                raise ValueError("rgba 各行必须等长且为列表")
            for x in range(w):
                pix = row[x]
                try:
                    r, g, b, a = pix
                except (TypeError, ValueError):
                    # 非序列或长度不是 4
                    raise ValueError(
                        "像素必须是长度为 4 的序列 (r, g, b, a)")
                if (not is_num(r, num_types) or not (0.0 <= r <= 1.0)
                        or not is_num(g, num_types) or not (0.0 <= g <= 1.0)
                        or not is_num(b, num_types) or not (0.0 <= b <= 1.0)
                        or not is_num(a, num_types) or not (0.0 <= a <= 1.0)):
                    # NaN/Inf 及越界值都会使比较链为 False
                    raise ValueError("像素通道取值必须是 [0, 1] 内的有限数")
                flat[idx] = pix
                if a == 1.0:
                    dist[idx] = 0
                    src[idx] = idx
                idx += 1

        if radius == 0:
            return [list(row) for row in rgba]

        D = dist
        S = src
        BIG = _BIG
        rmax = radius

        # 前向扫描，邻域：左、左上、上、右上（种子距离为 0，保持不变）
        for y in range(h):
            row0 = y * w
            if y == 0:
                # 首行只有左侧邻居
                for p in range(row0 + 1, row0 + w):
                    if D[p] == 0:
                        continue
                    if D[p - 1] < BIG:
                        D[p] = D[p - 1] + 1
                        S[p] = S[p - 1]
                continue
            up = row0 - w
            # 行首：没有左侧及左上邻居
            p = row0
            if D[p] != 0:
                best_d = D[up]
                best_s = S[up]
                d = D[up + 1]
                if d < best_d:
                    best_d = d
                    best_s = S[up + 1]
                if best_d < BIG:
                    D[p] = best_d + 1
                    S[p] = best_s
            # 行内：四个邻居齐全
            for p in range(row0 + 1, row0 + w - 1):
                if D[p] == 0:
                    continue
                best_d = D[p - 1]
                best_s = S[p - 1]
                d = D[up + p - row0]
                if d < best_d:
                    best_d = d
                    best_s = S[up + p - row0]
                d = D[up + p - row0 - 1]
                if d < best_d:
                    best_d = d
                    best_s = S[up + p - row0 - 1]
                d = D[up + p - row0 + 1]
                if d < best_d:
                    best_d = d
                    best_s = S[up + p - row0 + 1]
                if best_d < BIG:
                    D[p] = best_d + 1
                    S[p] = best_s
            if w > 1:
                # 行尾：没有右侧及右上邻居
                p = row0 + w - 1
                if D[p] != 0:
                    best_d = D[p - 1]
                    best_s = S[p - 1]
                    d = D[up + w - 1]
                    if d < best_d:
                        best_d = d
                        best_s = S[up + w - 1]
                    d = D[up + w - 2]
                    if d < best_d:
                        best_d = d
                        best_s = S[up + w - 2]
                    if best_d < BIG:
                        D[p] = best_d + 1
                        S[p] = best_s

        # 反向扫描，邻域：右、左下、下、右下；扫描完一行立即重建该行输出，
        # 省掉独立的第三趟遍历。距离 <= radius 的像素取最近不透明像素
        # 的颜色，alpha 一律保持原值
        out = [None] * h
        for y in range(h - 1, -1, -1):
            row0 = y * w
            if y == h - 1:
                # 末行只有右侧邻居
                for p in range(row0 + w - 2, row0 - 1, -1):
                    if D[p] == 0:
                        continue
                    if D[p + 1] < BIG:
                        nd = D[p + 1] + 1
                        if nd < D[p]:
                            D[p] = nd
                            S[p] = S[p + 1]
            else:
                dn = row0 + w
                # 行尾：没有右侧及右下邻居
                p = row0 + w - 1
                if D[p] != 0:
                    best_d = D[dn + w - 1]
                    best_s = S[dn + w - 1]
                    d = D[dn + w - 2]
                    if d < best_d:
                        best_d = d
                        best_s = S[dn + w - 2]
                    if best_d < BIG:
                        nd = best_d + 1
                        if nd < D[p]:
                            D[p] = nd
                            S[p] = best_s
                # 行内：四个邻居齐全
                for p in range(row0 + w - 2, row0, -1):
                    if D[p] == 0:
                        continue
                    best_d = D[p + 1]
                    best_s = S[p + 1]
                    d = D[dn + (p - row0)]
                    if d < best_d:
                        best_d = d
                        best_s = S[dn + (p - row0)]
                    d = D[dn + (p - row0) - 1]
                    if d < best_d:
                        best_d = d
                        best_s = S[dn + (p - row0) - 1]
                    d = D[dn + (p - row0) + 1]
                    if d < best_d:
                        best_d = d
                        best_s = S[dn + (p - row0) + 1]
                    if best_d < BIG:
                        nd = best_d + 1
                        if nd < D[p]:
                            D[p] = nd
                            S[p] = best_s
                if w > 1:
                    # 行首：没有左侧及左下邻居
                    p = row0
                    if D[p] != 0:
                        best_d = D[p + 1]
                        best_s = S[p + 1]
                        d = D[dn]
                        if d < best_d:
                            best_d = d
                            best_s = S[dn]
                        d = D[dn + 1]
                        if d < best_d:
                            best_d = d
                            best_s = S[dn + 1]
                        if best_d < BIG:
                            nd = best_d + 1
                            if nd < D[p]:
                                D[p] = nd
                                S[p] = best_s

            row_out = [None] * w
            p = row0
            for x in range(w):
                d = D[p]
                orig = flat[p]
                if 0 < d <= rmax:
                    seed = flat[S[p]]
                    row_out[x] = (seed[0], seed[1], seed[2], orig[3])
                else:
                    row_out[x] = (orig[0], orig[1], orig[2], orig[3])
                p += 1
            out[y] = row_out
        return out
