"""eval — 竞技场（M1，一切强度结论的裁判）。

方法学（docs/PLAN.md M1-4）：同一发牌种子，被测 bot 轮换四席位，
每种子打满四局取平均；pt = +90/+45/0/-135；报告含按种子聚类的 95% CI。

基线：B0 随机合法动作 | B1 启发式(和牌优先+现物防守,兼线上兜底)。
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable


@runtime_checkable
class Bot(Protocol):
    """一切参赛者(含 LAYA)的最小接口。events 为当前局 mjai 事件前缀(惰性视图)。

    legal_actions 是本决策点可用的 mjai 动作 JSON 字符串列表；
    react 必须原样返回其中一个。
    """

    name: str

    def react(self, events, seat: int, legal_actions: list[str]) -> str: ...


class RandomBot:
    name = "B0-random"

    def __init__(self, seed: int = 0) -> None:
        import random

        self._rng = random.Random(seed)

    def react(self, events, seat: int, legal_actions: list[str]) -> str:
        return self._rng.choice(legal_actions)


class HeuristicBot:
    """B1：确定性启发式，作为线上超时兜底与强度下限。

    规则(全部只读 mjai 事件流，动作词表实测于 riichienv:
    和牌=hora, 过=none, 立直=reach)：
    1. hora/reach 出现即选(和牌与立直永远优先)；
    2. 有人已立直且轮到弃牌：避开该家的现物(立直后其牌河出过的牌)；
    3. 鸣牌(pon/chi/daiminkan/kakan/ankan/kita)一律 none(不冒进)；
    4. 弃牌：从事件流重入手牌，选使向听数最小的牌(本 bot 永不鸣牌，
       门前清假设成立)；并列时种子随机。
    """

    name = "B1-heuristic"

    def __init__(self, seed: int = 0) -> None:
        import random

        self._rng = random.Random(seed)

    def react(self, events, seat: int, legal_actions: list[str]) -> str:
        def typ(a):
            return json.loads(a)["type"]

        for want in ("hora", "reach"):
            for a in legal_actions:
                if typ(a) == want:
                    return a
        if any(typ(a) in ("pon", "daiminkan", "chi", "kakan", "ankan", "kita") for a in legal_actions):
            for a in legal_actions:
                if typ(a) == "none":
                    return a
        discards = [a for a in legal_actions if typ(a) == "dahai"]
        if not discards:
            return legal_actions[self._rng.randrange(len(legal_actions))]
        gen = self._genbutsu(events)
        safe = [a for a in discards if json.loads(a)["pai"] not in gen]
        pool = safe or discards
        pool = self._by_shanten(events, seat, pool) or pool
        return pool[self._rng.randrange(len(pool))]

    @staticmethod
    def _tid(tile: str) -> int:
        """mjai 牌名 -> riichienv 物理 tile id (0..135)。
        calculate_shanten 的 Rust 侧自己做 id//4，传类型 id 会越界。
        字牌事件流用 E/S/W/N/P/F/C，parse_tile 要 1z..7z。"""
        from riichienv import parse_tile

        alt = {"E": "1z", "S": "2z", "W": "3z", "N": "4z", "P": "5z", "F": "6z", "C": "7z"}
        return parse_tile(alt.get(tile, tile))

    def _by_shanten(self, events, seat: int, discards: list[str]) -> list[str]:
        """过滤出保持最小向听数的弃牌候选；手牌/鸣牌不可得时原样返回。"""
        try:
            from riichienv import calculate_shanten
        except ImportError:
            return discards

        _t = self._tid
        hand: list[int] = []
        melded = False
        for ev in events:
            t = ev.get("type")
            if t == "start_kyoku":
                hand = [_t(p) for p in ev["tehais"][seat]]
                melded = False
            elif t == "tsumo" and ev.get("actor") == seat:
                hand.append(_t(ev["pai"]))
            elif t == "dahai" and ev.get("actor") == seat:
                tile = _t(ev["pai"])
                if tile in hand:
                    hand.remove(tile)
            elif t in ("pon", "chi", "daiminkan", "kakan", "ankan") and ev.get("actor") == seat:
                melded = True
        if melded or len(hand) < 8:
            return discards
        scored = []
        for a in discards:
            tile = _t(json.loads(a)["pai"])
            if tile not in hand:
                return discards  # 手牌重建失配：放弃优化
            rest = hand[:]
            rest.remove(tile)
            try:
                scored.append((calculate_shanten(rest), a))
            except Exception:  # noqa: BLE001 — Rust 侧任何异常都退回不优化，绝不让 B1 崩
                return discards
        best = min(s for s, _ in scored)
        out = [a for s, a in scored if s == best]
        return out or discards

    @staticmethod
    def _genbutsu(events) -> set:
        """所有已立直者(含 reach_accepted)的现物集合：立直后他们打过的牌。"""
        gen: set = set()
        riichis: set = set()
        for ev in events:
            t = ev.get("type")
            if t == "reach_accepted":
                riichis.add(ev.get("actor"))
            elif t == "dahai" and ev.get("actor") in riichis:
                gen.add(ev.get("pai"))
        return gen


BOTS = {
    "B0": RandomBot,
    "B1": HeuristicBot,
    # B3 = 微调 LAYA（M2 候选）：'B3@checkpoints/<run_id>'
    "B3": "eval.laya_bot.LayaBot",
}


def make_bot(spec: str, seat: int):
    """'B0' / 'B0#7'（指定 rng 种子后缀）/ 'B3@<ckpt目录>' -> bot 实例。"""
    if "@" in spec:
        name, _, ck = spec.partition("@")
        cls = BOTS[name]
        if isinstance(cls, str):
            import importlib

            mod, _, attr = cls.rpartition(".")
            cls = getattr(importlib.import_module(mod), attr)
        return cls(seed=seat, ckpt=ck)
    name, _, seed_s = spec.partition("#")
    cls = BOTS[name]
    if isinstance(cls, str):
        import importlib

        mod, _, attr = cls.rpartition(".")
        cls = getattr(importlib.import_module(mod), attr)
    return cls(seed=int(seed_s) if seed_s else seat)
