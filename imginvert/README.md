# imginvert

## 题面（可直接粘贴）

反相与色适应。约束：简单反相 1-x 逐通道精确（≤1e-12）；亮度保持反相必须在 Lab 空间反相 chroma 且 Y 不变（≤1e-6）；色域映射必须用 Bradford 色适应矩阵，与解析式 ≤1e-9；两次反相往返恒等（≤1e-12）；alpha 不参与；100 万像素不超过 500 ms；非法模式与越界通道可区分报错。 工作目录为仓库内的 imginvert/。 验收：cd imginvert && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd imginvert && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
