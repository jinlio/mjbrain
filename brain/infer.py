"""单向前向评分核：(ob, legal≥2) -> 温度校准后的 per-marker logits。

arena B3（eval.laya_bot）与线上 advisor（advisor.recommend）共用这一个
实现：同设备 + 同精度参数 ⇒ 数值逐位一致，"两套前向各自漂移"从结构上
不可能再发生。精度策略只在本模块定义：
  "fp32"  永不半精度；
  "fp16"  仅 CUDA 走 autocast fp16（mps/cpu 无半精度路径，行为同 fp32）。
历史口径：arena B3 一直是 "fp16"（cuda 上即 fp16，发布成绩 74.1% 由此
测得）；advisor 一直是 "fp32"。两者同机分叉的唯一入口就是本参数。
"""

from __future__ import annotations

import contextlib

PRECISIONS = ("fp32", "fp16")


class Truncated(ValueError):
    """选项被序列截断（markers 与 crit 数不等）。独立类让上层区分兜底口径。"""


def check_precision(p: str) -> str:
    if p not in PRECISIONS:
        raise ValueError(f"precision 须为 {PRECISIONS} 之一，得 {p!r}")
    return p


def use_half(dev_type: str, precision: str) -> bool:
    check_precision(precision)
    return precision == "fp16" and dev_type == "cuda"


def forward_scaled_logits(L: dict, dev, ob, legal: list[str],
                          precision: str) -> "np.ndarray":
    """返回温度校准后的 numpy 向量（长度=len(legal)）。调用方保证 len≥2。"""
    import torch
    from laya.common import build_sequence, temp_bucket

    from brain.serialize import laya_question, state_text

    check_precision(precision)
    q = laya_question(legal)
    ids, markers = build_sequence(
        L["tok"], state_text(ob), q, L["max_len"], L["head_max_len"])
    if len(markers) != len(q["crit"]):
        raise Truncated("选项被序列截断（加大 head_max_len）")
    n, k = len(ids), len(markers)
    batch = (
        torch.tensor([ids], dtype=torch.long),
        torch.ones(1, n, dtype=torch.long),
        torch.tensor([markers], dtype=torch.long),
        torch.ones(1, k, dtype=torch.bool),
        torch.tensor([0], dtype=torch.long),
    )
    # 半精度上下文只在真要半精度时才建：fp32 路径与"无 autocast 上下文"
    # 的旧 advisor 写法逐位等价（autocast(enabled=False) 不建则不扰）
    ctx = (torch.autocast(dev.type, dtype=torch.float16)
           if use_half(dev.type, precision) else contextlib.nullcontext())
    with torch.no_grad(), ctx:
        logits, _ = L["model"](*(x.to(dev) for x in batch))
    z = logits.float().cpu().numpy()[0, :k]
    t = L["temps_by_opts"].get(temp_bucket(0, k), L["temps"][0])
    return z / max(1e-3, float(t))
