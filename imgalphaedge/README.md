# imgalphaedge

## 题面（可直接粘贴）

透明边缘处理（去黑边/去白边）。约束：按预乘 alpha 合成且反预乘往返 ≤1/255；边缘像素颜色由最近不透明像素外扩填充（扩展距离与设定值一致）；alpha=0 处颜色不参与平均（可测）；合成到任意背景无暗边（边缘亮度差 ≤2%）；100 万像素不超过 700 ms；非法边距可区分报错。 工作目录为仓库内的 imgalphaedge/。 验收：cd imgalphaedge && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd imgalphaedge && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
