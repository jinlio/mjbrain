"""档1 推荐服务内核（advisor.server.react）测试：不起 HTTP，直调函数。

真实前向走 CPU（gate ckpt，模型 ~1s/手），避免与训练抢卡。
"""

import json
from pathlib import Path

import pytest

from advisor.server import react

CKPT = Path("D:/projects/mjbrain/checkpoints/20260921T184331Z-rlcd-gate")
PARQUET = Path("D:/projects/mjbrain/data/raw/tenhou_houou_mjai/data/tenhou-00000.parquet")


@pytest.fixture(scope="module")
def first89():
    if not PARQUET.exists():
        pytest.skip("棋谱数据未同步")
    if not CKPT.exists():
        pytest.skip("gate ckpt 不在")
    import pandas as pd
    df = pd.read_parquet(PARQUET)
    events = [json.loads(ln)
              for ln in df.iloc[0]["events"].splitlines() if ln.strip()][:89]
    return events


def test_react_kifu_seat1(first89):
    out = react(first89, seat=1, ckpt=str(CKPT), device="cpu")
    assert out["applied"] == 89
    assert out["open_seats"] == [1]
    d = out["decisions"][0]
    assert d["legal_n"] == 15
    assert d["recommend"].startswith("dahai:")
    ps = sum(x["p"] for x in d["top"])
    assert 0.9 < ps <= 1.001  # 温度 softmax：截 top5 前和=1


def test_react_bad_stream_reports_error():
    # 缺必填字段：riichienv 拒收（未知类型会静默忽略，不能用 bogus）
    bad = [{"type": "start_kyoku"}]
    out = react(bad, seat=None, ckpt="unused-because-error-first", device="cpu")
    assert "error" in out and out["applied"] == 0
