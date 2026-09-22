import json

from engine.sim import MahjongSim
from eval import BOTS


def test_tonpuusen_completes_and_logs():
    bots = [BOTS["B1"](seed=i) for i in range(2)] + [BOTS["B0"](seed=i) for i in range(2)]
    res = MahjongSim(seed=11, hanchan=False).play(bots)
    assert len(res.scores) == 4
    assert all(s >= 0 for s in res.scores)
    assert len(res.mjai_log) > 100
    first = json.loads(res.mjai_log[0])
    assert first["type"] == "start_game"
    assert any(json.loads(e)["type"] == "end_game" for e in res.mjai_log[-5:])


def test_legal_actions_are_mjai_roundtrip():
    sim = MahjongSim(seed=5, hanchan=False)
    sim.reset()
    legal = sim.legal(0)
    assert legal
    for mj in legal:
        assert isinstance(mj, str)
        ev = json.loads(mj)  # 必须是合法 JSON 的 mjai 事件
        assert "type" in ev


def test_events_view_lazy():
    from engine.sim import EventsView

    raw = ['{"type":"start_game"}', '{"type":"dahai","pai":"1p","actor":0}']
    v = EventsView(lambda: raw)
    assert len(v) == 2
    assert v[-1]["pai"] == "1p"
    assert v[0]["type"] == "start_game"
    raw.append('{"type":"end_kyoku"}')  # 增长后增量解析
    assert len(v) == 3 and v[2]["type"] == "end_kyoku"


def test_rotation_symmetry():
    """同实力 bot 轮换席位：各 spec 的 top1 合计应均匀（bug 回归测试）。"""
    from eval.arena import play_seed

    rows = play_seed(["B0#0", "B0#1", "B0#2", "B0#3"], seed=123, hanchan=False)
    assert all(r["games"] == 4 for r in rows)
    assert sum(r["top1"] for r in rows) == 4  # 每局恰好一个 top1


def test_heuristic_beats_random():
    """框架合理性：5 个种子下 B1 合计 top1 应明显高于 B0（单一种子噪声大）。"""
    from eval.arena import play_seed

    b1 = b0 = 0
    for s in range(5):
        rows = play_seed(["B1#1", "B1#2", "B0#3", "B0#4"], seed=4242 + s, hanchan=False)
        b1 += rows[0]["top1"] + rows[1]["top1"]
        b0 += rows[2]["top1"] + rows[3]["top1"]
    assert b1 > b0
