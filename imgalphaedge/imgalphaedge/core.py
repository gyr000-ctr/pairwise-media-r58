"""imgalphaedge 核心 API（初始未实现，仅标准库）。"""


class AlphaEdge:
    def premultiply(self, rgba):
        """预乘"""
        raise NotImplementedError

    def unpremultiply(self, rgba):
        """反预乘"""
        raise NotImplementedError

    def dilate(self, rgba, radius):
        """外扩"""
        raise NotImplementedError
