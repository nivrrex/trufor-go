"""最小 timm 兼容层（仅测试/导出用）。

上游 TruFor 源码从 timm 导入 DropPath / to_2tuple / trunc_normal_。这三者只影响
权重初始化，而推理路径的权重全部来自 checkpoint，因此这里用最小实现替代，
避免为跑参考实现而安装完整 timm（及其 torchvision 依赖）。
"""
import torch
import torch.nn as nn


def to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    with torch.no_grad():
        tensor.normal_(mean, std)
        tensor.clamp_(min=mean + a * std, max=mean + b * std)
    return tensor


class DropPath(nn.Module):
    """eval 模式下恒等；这里不实现训练行为。"""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return x

    def extra_repr(self):
        return f"drop_prob={self.drop_prob}"
