"""档1 推荐服务内核（advisor.server.react）测试：不起 HTTP，直调函数。

真实前向走 CPU（gate ckpt，模型 ~1s/手），避免与训练抢卡。
"""

import json
import os
from pathlib import Path

import pytest

from advisor.server import _has_reach, _reach_declare_window, react

_ROOT = Path(__file__).parents[1]   # 仓库根：棋谱/权重按仓内相对位置找（环境变量可覆盖）
CKPT = Path(os.environ.get("MJBRAIN_CKPT") or _ROOT / "checkpoints/20260921T184331Z-rlcd-gate")
PARQUET = Path(os.environ.get("MJBRAIN_PARQUET") or _ROOT / "data/raw/tenhou_houou_mjai/data/tenhou-00000.parquet")

# 合成立直局：庄家 111m222m333m45m99m + 摸 1z(E)。切 E 是唯一保听打牌（听 3m/6m），
# 故主窗 15 项 = dahai×14 + 一个裸 reach，宣言窗只剩 dahai:E（riichienv 两段式：
# 先声明，再开"只剩保听打牌"的窗）。tile 数以 13+tsumo 计——庄家在 start_kyoku
# 里给 13 张、第 14 张走 tsumo 事件（与 capture/mjai/state.py 同形状）。
REACH_HAND13 = ["1m", "1m", "1m", "2m", "2m", "2m", "3m", "3m", "3m",
                "4m", "5m", "9m", "9m"]
REACH_STREAM = [
    {"type": "start_game", "id": "0", "names": ["a", "b", "c", "d"], "num_players": 4},
    {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0,
     "oya": 0, "dora_marker": "1z", "scores": [25000] * 4, "num_players": 4,
     "tehais": [REACH_HAND13, ["?"] * 13, ["?"] * 13, ["?"] * 13]},
    {"type": "tsumo", "actor": 0, "pai": "1z"},
]


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
    # 该窗立直可选 → 附带宣言牌段：只剩保听打牌的分布（实测 切西 58.3%/切7饼 41.7%）
    r = d["reach"]
    assert r["recommend"].startswith("dahai:")
    assert r["legal_n"] >= 1
    assert 0.9 < sum(x["p"] for x in r["top"]) <= 1.001


def test_react_bad_stream_reports_error():
    # 缺必填字段：riichienv 拒收（未知类型会静默忽略，不能用 bogus）
    bad = [{"type": "start_kyoku"}]
    out = react(bad, seat=None, ckpt="unused-because-error-first", device="cpu")
    assert "error" in out and out["applied"] == 0


# ---------- 立直两段式（宣言牌建议） ----------

def test_has_reach_on_legal_strings():
    assert _has_reach(['{"actor":0,"pai":"1m","type":"dahai"}',
                       '{"actor":0,"type":"reach"}'])
    assert not _has_reach(['{"actor":0,"pai":"1m","type":"dahai"}', "not-json"])
    assert not _has_reach([])


def test_reach_declare_window_absent_without_window():
    # 13 张未摸牌局面：该座无决策窗 → 宣言段按 None 省去（不抛，也不碰 adviser）
    assert _reach_declare_window(REACH_STREAM[:2], 0, object(), top=5) is None


def test_react_reach_declaration_branch():
    if not CKPT.exists():
        pytest.skip("gate ckpt 不在")
    out = react(REACH_STREAM, seat=0, ckpt=str(CKPT), device="cpu")
    d = out["decisions"][0]
    assert d["legal_n"] == 15 and d["recommend"] == "reach"
    r = d["reach"]
    assert r["legal_n"] == 1 and r["recommend"] == "dahai:E"  # 唯一保听 = 摸到的 1z
    assert r["top"][0]["p"] == 1.0  # 单候选短路
