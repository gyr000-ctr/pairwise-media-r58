"""imginvert 核心 API（初始未实现，仅标准库）。"""


class Invert:
    def apply(self, rgb, mode):
        """反相"""
        raise NotImplementedError

    def to_lab(self, rgb):
        """Lab"""
        raise NotImplementedError

    def from_lab(self, lab):
        """回 RGB"""
        raise NotImplementedError
