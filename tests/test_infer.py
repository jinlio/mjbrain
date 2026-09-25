"""brain.infer 唯一前向核：精度策略纯函数 + 假模型端到端接线。

不碰真权重：build_sequence/state_text 打桩，模型返回固定 logits——
钉住的是"batch 构造、温度缩放、截断语义"这层接线，advisor 与 arena
共用后数值只可能差在真权重/真设备上。
"""

from __future__ import annotations

import laya.common
import numpy as np
import pytest
import torch

import brain.infer as infer
import brain.serialize as bs

LEGAL = ['{"type":"dahai","pai":"5p"}', '{"type":"none"}']


def _fake_seq(monkeypatch, markers_n=None):
    monkeypatch.setattr(bs, "state_text", lambda ob: "S")

    def fake(tok, st, q, max_len, head_max_len):
        k = len(q["crit"]) if markers_n is None else markers_n
        return list(range(k + 1)), list(range(k))

    monkeypatch.setattr(laya.common, "build_sequence", fake)


class _Fixed:
    def __init__(self, rows):
        self.t = torch.tensor([rows], dtype=torch.float32)

    def __call__(self, *xs):
        return self.t, None


def _L(rows):
    return {"tok": None, "model": _Fixed(rows), "max_len": 8,
            "head_max_len": 8, "temps": [2.0], "temps_by_opts": {}}


def test_precision_policy_table():
    assert infer.use_half("cuda", "fp16") is True
    assert infer.use_half("cuda", "fp32") is False
    assert infer.use_half("mps", "fp16") is False  # 非 cuda 无半精度路径
    assert infer.use_half("cpu", "fp16") is False
    with pytest.raises(ValueError, match="precision"):
        infer.check_precision("fp64")


def test_core_batch_and_temperature(monkeypatch):
    _fake_seq(monkeypatch)
    z = infer.forward_scaled_logits(_L([2.0, 4.0]), torch.device("cpu"),
                                    "ob", LEGAL, "fp32")
    assert isinstance(z, np.ndarray) and z.shape == (2,)
    assert np.allclose(z, [1.0, 2.0])  # logits / temps[0]=2（temp_bucket 缺表回退）
    # cpu 上 "fp16" 与 "fp32" 必须逐位同（策略只对 cuda 生效）
    z2 = infer.forward_scaled_logits(_L([2.0, 4.0]), torch.device("cpu"),
                                     "ob", LEGAL, "fp16")
    assert np.array_equal(z, z2)


def test_core_truncation_is_valueerror(monkeypatch):
    _fake_seq(monkeypatch, markers_n=1)  # crit 2 个，markers 只 1 个
    with pytest.raises(infer.Truncated, match="head_max_len"):
        infer.forward_scaled_logits(_L([2.0]), torch.device("cpu"),
                                    "ob", LEGAL, "fp32")
    assert issubclass(infer.Truncated, ValueError)  # advisor 侧旧捕获口径不变
