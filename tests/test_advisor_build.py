"""档0 局面合成器（advisor.build_state）的金标测试。

核心是**往返验证**：真棋谱（凤王 mjai）的每个决策点 → BoardSpec 反推
→ build_events 重建事件流 → 新 RiichiEnv 重放 → 与原 env 比对
legal_actions + state_text 全等。等于用 env 自带的完整规则校验器当
合成器的正确性闸门。

豁免（诚实记录，非缺陷）：
- 含立直后的窗口：本档合成器不重放立直棒/宣言牌时序，整局跳过 reach 后决策。
- "刚鸣牌就要出牌"的窗口（tail=chi/pon/kan）：档0 手输场景等价表达为
  摸牌式输入，不做重放（unsup 计数断言>0 但从不 fail）。
- 岭上杠宝牌指示牌：虚构牌山翻出的后续指示牌与真谱不同，state_text 的
  ドラ 段只比对首张指示牌。
"""

import json
import re
from pathlib import Path

import pytest

cvt = pytest.importorskip("riichienv.convert")
from riichienv import RiichiEnv

from advisor.build_state import BoardSpec, Furo, build_events
from brain.serialize import state_text
from engine.replay import replay_decisions

PARQUET = Path("D:/projects/mjbrain/data/raw/tenhou_houou_mjai/data/tenhou-00000.parquet")


def norm_state(s: str) -> str:
    return re.sub(r"ドラ\[([^\s\]]+)[^\]]*\]", r"ドラ[\1]", s)


class Unsupported(Exception):
    pass


def spec_from_game(events, d) -> BoardSpec:
    prefix = events[: d.idx]
    ob = d.ob
    sk_i = max(i for i, e in enumerate(prefix) if e["type"] == "start_kyoku")
    sk = prefix[sk_i]
    ky = prefix[sk_i:]  # 本局内前缀：副露/时点严禁跨局
    hand_m = [cvt.tid_to_mjai(t) for t in ob.hand]
    tail = ky[-1]
    draw = last = None
    if tail["type"] == "tsumo" and tail["actor"] == d.pid:
        draw = tail["pai"]
        hand13 = list(hand_m)
        hand13.remove(draw)
    elif tail["type"] == "dahai":
        last = (tail["pai"], tail["actor"])
        hand13 = hand_m
    else:
        raise Unsupported(f"tail={tail['type']}")
    rivers = [[cvt.tid_to_mjai(t) for t in rv] for rv in ob.discards]
    furos = []
    for i, e in enumerate(ky):
        t = e.get("type")
        if t in ("chi", "pon", "daiminkan", "kakan", "ankan"):
            pos = sum(
                1 for x in ky[:i]
                if x["type"] == "dahai" and x["actor"] == e["actor"])
            cons = list(e.get("consumed", []))
            pai = e.get("pai") or (cons[0] if cons else "")
            furos.append(Furo(e["actor"], t, pai, e.get("target", -1), cons, pos))
    return BoardSpec(
        seat=d.pid, hand=hand13, rivers=rivers, furos=furos,
        draw=draw, last=last, oya=ob.oya, round_wind=ob.round_wind,
        kyoku=ob.kyoku_index + 1, honba=ob.honba, sticks=ob.riichi_sticks,
        scores=list(sk.get("scores") or [25000] * 4), dora=sk["dora_marker"])


@pytest.fixture(scope="module")
def roundtrip_stats():
    if not PARQUET.exists():
        pytest.skip(f"棋谱数据未同步: {PARQUET}")
    import pandas as pd
    df = pd.read_parquet(PARQUET)
    ok = unsup = 0
    failures = []
    for gi in range(4):
        events = [json.loads(ln) for ln in df.iloc[gi]["events"].splitlines() if ln.strip()]
        for d in replay_decisions(events):
            legal = list(d.legal)
            prefix = events[: d.idx]
            if any(e.get("type") in ("reach", "reach_accepted") for e in prefix):
                continue  # 立直后窗口豁免
            if any(('"' + w + '"') in a for a in legal
                   for w in ("hora", "tsumo", "ryuukyoku", "skip")):
                continue  # 和牌/过牌窗口不在模型
            try:
                spec = spec_from_game(events, d)
            except Unsupported:
                unsup += 1
                continue
            try:
                evs = build_events(spec)
                env2 = RiichiEnv(game_mode="4p-red-half")
                for e in evs:
                    env2.apply_event(e)
                ob2 = env2.get_observation(d.pid)
            except Exception as ex:  # noqa: BLE001 —— 往返闸门收集一切异常入列
                failures.append(f"g{gi} idx{d.idx} pid{d.pid}: {type(ex).__name__} {ex}")
                continue
            l1 = sorted(legal)
            l2 = sorted(str(a.to_mjai()) for a in ob2.legal_actions())
            s1, s2 = norm_state(state_text(d.ob)), norm_state(state_text(ob2))
            if l1 == l2 and s1 == s2:
                ok += 1
            else:
                diff = "legal" if l1 != l2 else "state"
                failures.append(f"g{gi} idx{d.idx} pid{d.pid}: {diff} mismatch")
    return ok, unsup, failures


def test_roundtrip_all_green(roundtrip_stats):
    ok, unsup, failures = roundtrip_stats
    assert not failures, f"{len(failures)} 失败，样例：\n" + "\n".join(failures[:6])
    assert ok > 300, f"有效往返仅 {ok} 例，覆盖面回归"
    assert unsup > 0  # 豁免路径确实被走到（防测试静默失效）


def test_synthetic_pon_board():
    """手输局面：我家碰过 5m、巡首摸 3s；对手牌虚构不违约。"""
    # 时间线自洽：seat0 于 3 打 9p 后 chii（弃 8s）；我家（seat2）于 1 打
    # 5m 后 pon（弃 W），巡 3 轮到我家摸 3s 出牌。
    spec = BoardSpec(
        seat=2,
        hand=["1m", "2m", "4m", "7p", "8p", "3s", "5s", "9s", "E", "S"],
        rivers=[["5s", "8s"], ["3p", "5m"], ["4s", "W"], ["1s", "9p"]],
        furos=[Furo(0, "chi", "9p", frm=3, consumed=["7p", "8p"], pos=1),
               Furo(2, "pon", "5m", frm=1, consumed=["5m", "5m"], pos=1)],
        draw="3s", last=None, oya=3, round_wind=0, kyoku=2, dora="1s")
    evs = build_events(spec)
    env = RiichiEnv(game_mode="4p-red-half")
    for e in evs:
        env.apply_event(e)
    ob = env.get_observation(2)
    legal = sorted(str(a.to_mjai()) for a in ob.legal_actions())
    assert any('"dahai"' in a for a in legal)
    assert len(legal) > 8  # 门前10+摸张=11 张可弃


def test_conservation_rejects_bad_river():
    """河对不上起手：守恒方程应报错而非静默造牌。"""
    spec = BoardSpec(
        seat=0,
        hand=[str(n) + "m" for n in range(1, 10)],
        rivers=[["1p", "2p", "3p"], ["4p"], ["5p"], ["6p"]],
        furos=[Furo(0, "pon", "4p", frm=3, consumed=["4p", "4p"], pos=99)],
        draw=None, last=("6p", 3), oya=0)
    with pytest.raises(ValueError):
        build_events(spec)
