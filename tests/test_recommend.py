"""advisor.recommend 守卫：空 legal、top 钳制、共享缓存设备改道冲突、
精度解析与端到端数值接线。

全部走假模型（monkeypatch laya_bot._load + brain.infer 桩），不碰真权重——
本文件可在无 GPU/无 checkpoint 的机器上跑。
"""

from __future__ import annotations

import pytest
import torch

import advisor.recommend as R
import eval.laya_bot as lb


class _FakeModel:
    def __init__(self) -> None:
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


@pytest.fixture
def entry(monkeypatch):
    e = {"model": _FakeModel(), "device": torch.device("cpu")}
    monkeypatch.setattr(lb, "_load", lambda ckpt: e)
    return e


def test_topk_empty_legal_raises(entry):
    a = R.Adviser("fake")
    with pytest.raises(ValueError, match="legal 为空"):
        a.topk(None, [])


def test_topk_single_action_and_top_clamp(entry):
    a = R.Adviser("fake")
    assert a.topk(None, ['{"type":"none"}']) == [("none", 1.0)]
    # top=0/负数过去让 out[:top] 静默截成空列表
    assert len(a.topk(None, ['{"type":"dahai","pai":"5p"}'], top=0)) == 1


def test_shared_cache_device_conflict_refused(entry):
    # 第一例显式设备：就地迁移并盖章（arena 单进程内 B3 各席共享此条目）
    a = R.Adviser("fake", device="meta")
    assert a._dev == torch.device("meta")
    assert entry["model"].moved_to == torch.device("meta")
    # 第二例再指定别的设备 = 会把第一例的模型悄悄搬走 → 必须拒绝
    with pytest.raises(RuntimeError, match="拒绝再迁"):
        R.Adviser("fake", device="cpu")
    # 不显式给设备 → 沿用缓存现状，不动模型
    b = R.Adviser("fake")
    assert b._dev == torch.device("meta")


def test_adviser_precision_resolution(entry, monkeypatch):
    # 默认 = 线上历史口径 fp32；env 可切；显式参数压过 env；非法值构造期炸
    assert R.Adviser("fake")._prec == "fp32"
    monkeypatch.setenv("MJBRAIN_ADVISOR_PRECISION", "fp16")
    assert R.Adviser("fake")._prec == "fp16"
    assert R.Adviser("fake", precision="fp32")._prec == "fp32"
    with pytest.raises(ValueError, match="precision"):
        R.Adviser("fake", precision="fp64")


def test_topk_end_to_end_probs(entry, monkeypatch):
    import math

    import laya.common

    import brain.serialize as bs

    class FixedModel:
        def __call__(self, *xs):
            return torch.tensor([[1.0, 3.0]]), None

    entry["model"] = FixedModel()
    entry.update(tok=None, max_len=8, head_max_len=8,
                 temps=[2.0], temps_by_opts={})
    legal = ['{"type":"dahai","pai":"5p"}', '{"type":"pon","pai":"5p"}']
    monkeypatch.setattr(bs, "state_text", lambda ob: "S")
    monkeypatch.setattr(
        laya.common, "build_sequence",
        lambda tok, st, q, a, b: (list(range(len(q["crit"]) + 1)),
                                  list(range(len(q["crit"])))))
    a = R.Adviser("fake")
    out = a.topk("ob", legal, top=2)
    # z=[0.5,1.5]（logits [1,3] / 温度 2）的 softmax——钉死与 core 的接线
    e = math.exp(-1.0)
    assert [d for d, _ in out] == ["pon:5p", "dahai:5p"]
    assert out[0][1] == pytest.approx(1.0 / (1.0 + e))
    assert out[1][1] == pytest.approx(e / (1.0 + e))
