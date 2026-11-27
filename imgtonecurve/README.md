# imgtonecurve

## 题面（可直接粘贴）

曲线调整（控制点单调样条）。约束：单调控制点必须产生单调曲线（≤1e-12）；曲线通过所有控制点（≤1e-9）；一阶导连续（数值校验 ≤1e-6）；端点固定 0→0、1→1 可配；10 万次求值不超过 150 ms；非单调控制点可区分报错。 工作目录为仓库内的 imgtonecurve/。 验收：cd imgtonecurve && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd imgtonecurve && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
