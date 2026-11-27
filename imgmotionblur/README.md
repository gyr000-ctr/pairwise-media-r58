# imgmotionblur

## 题面（可直接粘贴）

方向运动模糊（累积缓冲）。约束：沿方向按等距采样累积且权重和为 1（≤1e-12）；长度 0 时恒等（≤1e-12）；与解析卷积（连续积分）差 ≤1e-2；边界按 clamp；100 万像素不超过 800 ms；非法长度与角度可区分报错。 工作目录为仓库内的 imgmotionblur/。 验收：cd imgmotionblur && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd imgmotionblur && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
