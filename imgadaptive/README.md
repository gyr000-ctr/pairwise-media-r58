# imgadaptive

## 题面（可直接粘贴）

自适应阈值（局部均值/高斯）。约束：阈值图为局部（高斯）加权均值，核和 1（≤1e-12）；与积分图实现的对照 ≤1e-9（O(1) 每像素）；二值化输出只含两值；常量图全为前景或全为背景（确定）；100 万像素不超过 700 ms；非法块尺寸与非法 C 可区分报错。 工作目录为仓库内的 imgadaptive/。 验收：cd imgadaptive && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd imgadaptive && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
