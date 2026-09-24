"""advisor.hints：向听/待牌/宝牌（HUD 进阶提示）的离线单测 + 真 env 集成。

FakeOb 只造 hints 读取的那几个 Observation 属性（hand/player_id/melds/
dora_indicators/legal_actions），但 Meld/Action 用真 riichienv 对象，
保证与 server 侧 open_windows 拿到的类型一致。
"""

from __future__ import annotations

import pytest
from riichienv import ActionType, GameRule, Meld, MeldType, RiichiEnv
from riichienv import Action as RAction
from riichienv import convert as cvt

from advisor.hints import _dora_of, _type_name, hint_from_obs


def tid(name: str) -> int:
    return cvt.mjai_to_tid(name)


class FakeOb:
    def __init__(self, hand: list[str], pid: int = 0,
                 melds: list | None = None, dora: list[str] | None = None,
                 discards: list[str] | None = None):
        self.hand = [tid(t) for t in hand]
        self.player_id = pid
        me = [melds or []] + [[] for _ in range(3)]
        if melds:
            me = [me[i] if i != pid else melds for i in range(4)]
        self.melds = me
        self.dora_indicators = [tid(d) for d in (dora or [])]
        # 有弃牌选项=弃牌窗；给 None 即"无 dahai 的 13 张窗"
        acts = [] if discards is None else [
            RAction(ActionType.DISCARD, tile=tid(t)) for t in discards]
        self._acts = acts

    def legal_actions(self):
        return self._acts


# ---------- 纯函数：宝牌映射 ----------

@pytest.mark.parametrize("ind,dora", [
    (0, 1),     # 1m 指示 → 2m
    (8, 0),     # 9m 指示 → 1m（组内回环）
    (26, 18),   # 9s → 1s
    (27, 28),   # 东 → 南
    (30, 27),   # 北 → 东（风回环）
    (31, 32),   # 白 → 发
    (33, 31),   # 中 → 白（箭回环）
])
def test_dora_of(ind, dora):
    assert _dora_of(ind) == dora


def test_type_name_avoids_red_five():
    assert _type_name(4) == "5m"   # 16 是赤5m 的物理 id，取同种非赤
    assert _type_name(13) == "5p"
    assert _type_name(22) == "5s"
    assert _type_name(27) == "E"


# ---------- 门前清弃牌窗 ----------

TENPAI13 = ["1m", "1m", "1m", "2m", "2m", "2m", "3m", "3m", "3m",
            "4m", "5m", "9m", "9m"]


def test_menzen_tenpai_window():
    # 111222333m45m99m + 摸东：切东唯一听牌打法；听面实测 3m/6m/9m
    # （+9m 拆 111m 222m 345m 33m 999m，同样和牌——get_waits 与手算一致）
    ob = FakeOb(TENPAI13 + ["E"], dora=["E"], discards=TENPAI13 + ["E"])
    h = hint_from_obs(ob)
    assert h is not None
    assert h["shanten"] == 0 and not h["melded"]
    assert h["waits"] == ["3m", "6m", "9m"]
    assert h["dora"] == ["S"]      # 指示东 → 宝牌南
    assert h["dora_in_hand"] == 0  # 手里没南


def test_menzen_shanten_counts_down():
    # 听牌型拆掉一张好牌 → 向听 -1 变成 1；无听牌线则 waits 空
    ob = FakeOb(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "1s", "1s",
                 "9p", "9p", "E", "S", "C"],
                dora=["1p"], discards=["1m", "2m", "3m", "4m", "5m", "6m",
                                        "7m", "1s", "9p", "E", "S", "C"])
    h = hint_from_obs(ob)
    assert h["shanten"] >= 1
    assert h["waits"] == []
    assert h["dora"] == ["2p"]


def test_menzen_13tile_no_discard_window():
    # 荣和窗（无弃牌选项的 13 张听牌）：向听 0，按整手算 waits
    ob = FakeOb(TENPAI13, dora=["9m"], discards=None)
    h = hint_from_obs(ob)
    assert h["shanten"] == 0
    assert h["waits"] == ["3m", "6m", "9m"]
    assert h["dora"] == ["1m"]      # 指示 9m → 宝牌 1m
    assert h["dora_in_hand"] == 3   # 手里三张 1m


def test_dora_in_hand_count():
    ob = FakeOb(TENPAI13, dora=["8m"], discards=None)  # 指示 8m → 宝牌 9m
    h = hint_from_obs(ob)
    assert h["dora"] == ["9m"] and h["dora_in_hand"] == 2


# ---------- 副露：向听降级、待牌精确（HandEvaluator 支持 melds） ----------

def test_melded_tenpai_no_shanten():
    # 副露 111s（碰）+ 门前 123m456m78m99p（+摸 9s）：听 3m/6m/9m
    # （+3m 拆 123/345/678、+6m 拆 123/456/678、+9m 拆 123/456/789，99p 雀头）。
    # calculate_shanten 不收 melds → 向听如实降级 None，待牌走 HandEvaluator。
    base = tid("1s") // 4 * 4  # 三张同种不同物理 id，避免 Rust 侧查重意外
    pon = [Meld(MeldType.Pon, [base, base + 1, base + 2], True, 1)]
    ob = FakeOb(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m",
                 "9p", "9p", "9s"], pid=0, melds=pon, dora=["1p"],
                discards=["9s"])
    h = hint_from_obs(ob)
    assert h["melded"] is True
    assert h["shanten"] is None
    assert h["waits"] == ["3m", "6m", "9m"]


# ---------- 真 env 集成（与 advisor.server.open_windows 同款取窗） ----------

def test_hints_on_real_env_window():
    env = RiichiEnv(game_mode="4p-red-half", rule=GameRule.default_mjsoul())
    env.apply_event({"type": "start_game", "id": "0",
                     "names": ["a", "b", "c", "d"], "num_players": 4})
    env.apply_event({"type": "start_kyoku", "bakaze": "E", "kyoku": 1,
                     "honba": 0, "kyotaku": 0, "oya": 0, "dora_marker": "1z",
                     "scores": [25000] * 4, "num_players": 4,
                     "tehais": [TENPAI13, ["?"] * 13, ["?"] * 13, ["?"] * 13]})
    env.apply_event({"type": "tsumo", "actor": 0, "pai": "1z"})
    ob = env.get_observation(0)
    assert ob is not None
    h = hint_from_obs(ob)
    assert h["shanten"] == 0 and h["waits"] == ["3m", "6m", "9m"]
    assert h["dora"] == ["S"]      # dora_marker 1z=东 → 宝牌南
    assert h["melded"] is False


# ---------- 防御：任何怪输入不得抛异常，只回 None ----------

def test_hint_never_raises():
    class Bad:
        pass
    assert hint_from_obs(Bad()) is None
    ob = FakeOb(TENPAI13, discards=["1m"])
    ob.hand = None
    assert hint_from_obs(ob) is None
