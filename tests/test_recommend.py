"""advisor.recommend 守卫：空 legal、top 钳制、共享缓存设备改道冲突。

全部走假模型（monkeypatch laya_bot._load），不碰真权重——本文件可在
无 GPU/无 checkpoint 的机器上跑。
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
