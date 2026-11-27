"""imgtonecurve 核心 API（初始未实现，仅标准库）。"""


class ToneCurve:
    def __init__(self, points):
        """构造"""
        raise NotImplementedError

    def eval(self, x):
        """求值"""
        raise NotImplementedError

    def slope(self, x):
        """斜率"""
        raise NotImplementedError
