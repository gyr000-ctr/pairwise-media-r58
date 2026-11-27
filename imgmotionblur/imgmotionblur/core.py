"""imgmotionblur 核心 API（初始未实现，仅标准库）。"""


class MotionBlur:
    def __init__(self, samples):
        """构造"""
        raise NotImplementedError

    def weights(self):
        """采样权重"""
        raise NotImplementedError

    def apply(self, image, length, angle):
        """模糊"""
        raise NotImplementedError
