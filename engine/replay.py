"""外部牌谱（mjai 服务器视角事件流）的确定性重演 -> 决策点 + 合法动作集。

训练数据生成用。核心是 riichienv 的 `apply_event` + `get_observation`
（Apache-2.0；README 明示此路径即面向"replay parsing, training data
generation"，牌山/手牌由事件流携带，无需播种）。

两个凤王日志的关键事实（实测于 data/raw/tenhou_houou_mjai）：
1. 日志**不记录 pass**：所有未鸣牌决策是静默的。凡是"决策窗口开着，
   而下一条日志不是该家的动作"，即补记一个隐式 none 标签；
2. 天凤的 dahai/reach/hora 事件在 mjai 中本来就是分开的行
   （reach -> dahai -> reach_accepted），与 riichienv 的两步流同构。
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

from riichienv import GameRule, RiichiEnv

# 日志中代表"玩家做出决策"的事件类型；其余是环境回声(start_kyoku/tsumo/
# dora/reach_accepted/ryukyoku/end_*/…)或整场边界。
ACTION_TYPES = frozenset(
    {"dahai", "pon", "chi", "daiminkan", "kakan", "ankan", "reach", "hora"}
)


@dataclass
class Decision:
    idx: int                 # 原事件流中动作行下标；决策前缀 = events[:idx]
    pid: int
    legal: list[str]         # riichienv 规范 mjai 字符串（与 arena legal 同源）
    label: str               # 规范字符串；隐式过牌时为 none 串
    implicit: bool = False   # True = 天凤静默过牌，非日志显式动作
    # 决策时刻的座位观察快照（get_observation 已实测为快照而非活动视图，
    # 可安全跨 apply_event 持有）。供内存态消费者使用（serialize 文本化、
    # 教师打分）；写盘消费者（build_decisions）忽略此字段。
    ob: object | None = None
    # 窗口是在哪一行事件被打开的（= 玩家视角 Bot 对哪一行 react 产出了
    # 决策响应）。idx 是动作行，两者对弃牌家本人差一整段（摸牌行 vs 打
    # 出行之）；对鸣牌家通常差 1。教师配对用它。
    opened_at: int = -1


def _key(t: str, pai):
    return (t, pai)


def _key_of(d: dict):
    """决策日志行/环境 legal 串的匹配键。ankan 行没有 pai 字段；且赤牌
    暗槓的牌面标识两家约定不同（天凤 consumed[0] 可能给非赤、env 的
    pai 给赤牌），故 ankan 一律以 consumed 集合为键。"""
    t = d.get("type")
    if t == "ankan":
        return (t, frozenset(d.get("consumed") or []))
    return (t, d.get("pai"))


def _none_str(legal: list[str]) -> str | None:
    for s in legal:
        if json.loads(s)["type"] == "none":
            return s
    return None


def replay_decisions(
    events: list[dict],
    *,
    game_mode: str = "4p-red-half",
    rule: GameRule | None = None,
    stats: Counter | None = None,
) -> Iterator[Decision]:
    """逐决策点产出 Decision；stats 累积异常计数（desync_*）。

    要求 events[0] 为 start_game（凤王日志即如此），且为单场完整流。
    """
    stats = stats if stats is not None else Counter()
    env = RiichiEnv(game_mode=game_mode, rule=rule or GameRule.default_tenhou())
    # pid -> (打开行下标, 窗口内容指纹)。窗口"连续可见"但内容变了（过牌
    # 后立即摸牌开打、pon 后强制打牌等）也要重记打开行，否则教师配对串窗。
    win_open: dict[int, tuple[int, frozenset]] = {}

    def _oa(pid: int) -> int:
        return win_open.get(pid, (-1, None))[0]

    def observe() -> dict[int, tuple[object, list[str]]]:
        """每一行都跑（含 dora/并响跳行），负责 open/close 簿记。

        窗口由第 a 行 apply 打开 ⇒ 在第 a+1 行的轮询里首次可见 ⇒
        opened_at=idx-1 恒等于真实打开行（前提：每行都轮询，dora/并响
        行也不能漏，否则首见行前移、opened_at 指错）。
        """
        out: dict[int, tuple[object, list[str]]] = {}
        for pid in range(4):
            try:
                ob = env.get_observation(pid)
            except Exception:  # noqa: BLE001,S112 — 非行动帧的取观测异常视为无窗口
                continue
            if ob is None:
                continue
            la = [str(a.to_mjai()) for a in ob.legal_actions()]
            if la:
                out[pid] = (ob, la)
        for pid, (_, la) in out.items():
            sig = frozenset(la)
            if win_open.get(pid, (None, None))[1] != sig:
                win_open[pid] = (idx - 1, sig)
        for pid in [p for p in win_open if p not in out]:
            del win_open[pid]
        return out

    hora_skip: set[int] = set()
    for idx, ev in enumerate(events):
        t = ev.get("type")
        if t in ("dora",) or idx in hora_skip:
            # dora=赤宝牌公开事件，出现在 tsumo 与 actor 的 dahai 之间，
            # 不关闭任何窗口；hora_skip=同巡并响行，已在组首行归档。
            # 簿记照常（opened_at 精度要求每行轮询），但不排空任何窗口。
            observe()
            try:
                env.apply_event(ev)
            except Exception:  # noqa: BLE001 — 引擎对坏事件一律弃本场
                stats["apply_fail"] += 1
                return
            continue
        wins = observe()
        if t in ACTION_TYPES:
            actor = ev.get("actor")
            win_actor = wins.get(actor)
            if win_actor is None:
                # 窗口没开但日志有动作：脱节，记录并继续（不中断批处理）
                stats["desync_no_window"] += 1
            else:
                obs_actor, legal_actor = win_actor
                want = _key_of(ev)
                chosen = next(
                    (s for s in legal_actor if _key_of(json.loads(s)) == want),
                    None,
                )
                if chosen is None:
                    stats["desync_illegal_label"] += 1
                else:
                    yield Decision(
                        idx,
                        actor,
                        legal_actor,
                        chosen,
                        ob=obs_actor,
                        opened_at=_oa(actor),
                    )
            # hora 行时同巡开着 hora 窗的并响家：按后续连续 hora 行组
            # 前瞻记 hora / 记隐式过牌，组内后续行不再产生决策。
            group_actors: set[int] = set()
            if t == "hora":
                j = idx + 1
                while j < len(events) and events[j].get("type") == "hora":
                    group_actors.add(events[j].get("actor"))
                    hora_skip.add(j)
                    j += 1
            for pid, (ob_p, legal) in wins.items():
                if pid == actor:
                    continue
                if pid in group_actors:
                    hc = next(
                        (s for s in legal if _key_of(json.loads(s)) == ("hora", None)),
                        None,
                    )
                    if hc is not None:
                        yield Decision(
                            idx, pid, legal, hc, ob=ob_p, opened_at=_oa(pid)
                        )
                    else:
                        stats["desync_illegal_label"] += 1
                    continue
                ns = _none_str(legal)
                if ns is not None:
                    yield Decision(
                        idx,
                        pid,
                        legal,
                        ns,
                        implicit=True,
                        ob=ob_p,
                        opened_at=_oa(pid),
                    )
                else:
                    stats["desync_no_none"] += 1
        else:
            # 回声/边界行：已开的(鸣牌)窗口全部隐式过牌；
            # 无 none 的窗口(如九种九牌前 actor 的 dahai 窗)跳过并计数。
            for pid, (ob_p, legal) in wins.items():
                ns = _none_str(legal)
                if ns is not None:
                    yield Decision(
                        idx,
                        pid,
                        legal,
                        ns,
                        implicit=True,
                        ob=ob_p,
                        opened_at=_oa(pid),
                    )
                else:
                    stats["unresolved_window"] += 1
        try:
            env.apply_event(ev)
        except Exception:  # noqa: BLE001 — 单场一条坏事件：弃本场
            stats["apply_fail"] += 1
            return
