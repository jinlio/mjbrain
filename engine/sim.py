"""riichienv 封装：一局牌的 bot 决策循环。

设计：对外只暴露 mjai 事件流与 mjai 字符串动作（与 /v1/react 契约同一词表），
riichienv 的 Action/Observation 对象不外漏，方便将来换引擎。

注意：Bot.react 里 seat 是座位号(0=亲家起)。合法动作文本可能有极小概率重复
（同牌两张不同的副露构造），bot 只需从列表里选字符串，引擎负责反解。
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from riichienv import GameType, RiichiEnv

if TYPE_CHECKING:
    from eval import Bot


class EventsView(Sequence):
    """mjai 事件流的惰性只读视图：底层日志随牌局增长，按索引增量解析并缓存。

    raw_provider 每次调用返回当前完整原始日志（O(n) 但 n<=2k 实测 <3ms；
    json 解析只对未缓存的尾部做，全程合计 O(n)）。
    """

    def __init__(self, raw_provider: Callable[[], list[str]]) -> None:
        self._provider = raw_provider
        self._parsed: list[dict] = []

    def _refresh(self) -> None:
        raw = self._provider()
        if len(raw) < len(self._parsed):  # 不应发生（reset 后重建视图）
            self._parsed.clear()
        for s in raw[len(self._parsed):]:
            # riichienv 的 mjai_log 元素已是 dict；其它来源给 JSON 字符串。
            # 原生 dict 入缓存必须 deepcopy：若引擎后续复用同一对象原地改，
            # "惰性历史视图"会看到被篡改的过去（n≤2k，防御成本可忽略）
            self._parsed.append(json.loads(s) if isinstance(s, str) else copy.deepcopy(s))

    def __len__(self) -> int:
        self._refresh()
        return len(self._parsed)

    def __getitem__(self, idx):
        self._refresh()
        return self._parsed[idx]

    def __iter__(self):
        self._refresh()
        return iter(self._parsed)


@dataclass
class GameResult:
    scores: list[int]
    seed: int
    steps: int
    mjai_log: list[str] = field(repr=False, default_factory=list)


class MahjongSim:
    """一整局（东风战/半庄）。用法：sim.play(bots) 或驱动循环用 step。"""

    def __init__(self, seed: int, hanchan: bool = True) -> None:
        gt = GameType.YON_HANCHAN if hanchan else GameType.YON_TONPUSEN
        self._env = RiichiEnv(gt, seed=seed)
        self.seed = seed
        self._obs: dict = {}
        self._view: EventsView | None = None

    def reset(self) -> None:
        self._obs = self._env.reset()
        self._view = EventsView(lambda: self._env.mjai_log)

    @property
    def done(self) -> bool:
        return self._env.done()

    def legal(self, pid: int) -> list[str]:
        ob = self._obs.get(pid)
        if ob is None:
            return []
        return [str(a.to_mjai()) for a in ob.legal_actions()]

    def events(self) -> EventsView:
        if self._view is None:  # assert 会被 python -O 剥离，显式 raise
            raise RuntimeError("先 reset()")
        return self._view

    def step(self, actions: dict[int, str]) -> None:
        """actions: {pid: mjai字符串}，只包含本帧 obs 里出现的玩家。"""
        env_actions = {}
        for pid, mj in actions.items():
            ob = self._obs[pid]
            a = ob.select_action_from_mjai(mj)
            if a is None:
                raise ValueError(f"seat {pid}: illegal mjai action {mj!r}")
            env_actions[pid] = a
        self._obs = self._env.step(env_actions)

    def scores(self) -> list[int]:
        return list(self._env.scores())

    def play(self, bots: list[Bot]) -> GameResult:
        if len(bots) != 4:  # 同上：-O 下 assert 失效会变裸 IndexError
            raise ValueError("4p 需要 4 个 bot")
        self.reset()
        steps = 0
        while not self.done:
            acts: dict[int, str] = {}
            for pid in self._obs:
                legal = self.legal(pid)
                if not legal:
                    continue
                if getattr(bots[pid], "wants_ob", False):
                    # 协议扩展：LAYA 类 bot 需要决策时刻的真实 Observation
                    # （step 制牌山按 seed 播种，bot 侧外部重演复现不了）。
                    chosen = bots[pid].react(self.events(), pid, legal, ob=self._obs[pid])
                else:
                    chosen = bots[pid].react(self.events(), pid, legal)
                if chosen not in legal:
                    raise ValueError(f"seat {pid} bot {bots[pid].name} 返回非法动作 {chosen!r}")
                acts[pid] = chosen
            self.step(acts)
            steps += 1
            if steps > 100_000:
                raise RuntimeError("game loop stuck")
        return GameResult(
            scores=self.scores(),
            seed=self.seed,
            steps=steps,
            mjai_log=[e if isinstance(e, str) else json.dumps(e, ensure_ascii=False) for e in self._env.mjai_log],
        )
