"""imgadaptive 核心 API（初始未实现，仅标准库）。"""


class Adaptive:
    def __init__(self, block, c):
        """构造"""
        raise NotImplementedError

    def threshold_map(self, image):
        """阈值图"""
        raise NotImplementedError

    def apply(self, image):
        """二值化"""
        raise NotImplementedError
