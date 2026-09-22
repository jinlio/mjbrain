"""牌局快照 -> 文本序列化（M2 核心研究点，LAYA 的 state 输入）。

约束（可行性调研 + laya 实测）：
- en 检查点默认 512 上下文（head_max_len=192 给 options），typed/multilingual
  检查点 1024（head 256）。麻将弃牌最多 34+3 选项，选项预算刚过百 token，
  state 文本目标 ≤ ~250 tokens。
- **实测（scripts/token_audit.py，20 局 400 决策点）**：选项区 p50=54 /
  p99=66（预算 192 零超）；state p50=162 / p99=278 / max=291；整序列
  max=346 < 512 —— v1 格式在 en 检查点窗口内无截断，留有 ~170 余量。
- [C-022]/[E-044]：计数型快照丢弃牌顺序=丢防守输入 —— v1 保序全河展开。

原则：
- **确定性**：同一 Observation 永远产出同一字符串；选项顺序=env legal 顺序，
  与 B2 教师概率向量 1:1 对齐；
- **零泄漏**：一切只取自 riichienv Observation——引擎已完成座位遮蔽
  （对手 concealed hands 为空），本模块不补任何外部信息；
- 规则特征（向听/waits）v1 不注入，留给 M2 消融决定（PLAN M2-3）。
"""

from __future__ import annotations

import json

import riichienv.convert as cvt

_WIND = {0: "E", 1: "S", 2: "W", 3: "N"}
_MELD_NAME = {
    "Chi": "chi",
    "Pon": "pon",
    "Daiminkan": "dakan",
    "Kakan": "kakan",
    "Ankan": "ankan",
}


def _names(tiles) -> list[str]:
    return [cvt.tid_to_mjai(t) for t in tiles]


def _kyoku_label(ob) -> str:
    # round_wind 0E..3N + 局内序号(kyoku_index%4, 1起)：东1/S2...
    return f"{_WIND.get(int(ob.round_wind), '?')}{int(ob.kyoku_index) % 4 + 1}"


def _meld_str(melds) -> str:
    parts = []
    for m in melds:
        mt = str(m.meld_type).split(".")[-1]
        kind = _MELD_NAME.get(mt, mt.lower())
        tiles = "".join(_names(m.tiles or []))  # tiles 已含被鸣的那张
        parts.append(f"{kind}:{tiles}<-{int(m.from_who)}")
    return " ".join(parts)


def _wall_est(ob) -> int:
    """活牌山近似：起手 122-52 配牌=70，每次摸打 -1；鸣牌让一家跳过摸牌 +1。
    杠的振替牌取死山不计。近似误差 ≤2，模型只需特征一致。"""
    draws = sum(len(d) for d in ob.discards)
    calls = sum(len(m) for m in ob.melds)
    return max(0, 70 - draws + calls)


def state_text(ob) -> str:
    """Observation -> 单行文本快照。"""
    hand = " ".join(sorted(_names(ob.hand)))
    drawn = cvt.tid_to_mjai(int(ob.drawn_tile)) if ob.drawn_tile is not None else "-"
    dora = " ".join(_names(ob.dora_indicators))
    pts = " ".join(str(int(s)) for s in ob.scores)
    parts = [
        (
            f"{_kyoku_label(ob)}局{int(ob.honba)}本棒{int(ob.riichi_sticks)}立直棒"
            f" 亲{int(ob.oya)} 壁~{_wall_est(ob)} ドラ[{dora}] 点数 {pts}"
        ),
        f"我={int(ob.player_id)} 手[{hand}] 摸{drawn}",
    ]
    for pid in range(len(ob.discards)):
        river = " ".join(_names(ob.discards[pid]))
        if ob.riichi_declared[pid]:
            seg = f"河{pid}[R][{river}]"
        else:
            seg = f"河{pid}[{river}]"
        melds = _meld_str(ob.melds[pid]) if pid < len(ob.melds) else ""
        if melds:
            seg += f" 副露[{melds}]"
        parts.append(seg)
    return " | ".join(parts)


def options_from_legal(legal: list[str]) -> list[str]:
    """env 的 mjai 规范串 -> LAYA choice 选项 key 列表（与 legal 严格 1:1 同序）。"""
    keys: list[str] = []
    for s in legal:
        a = json.loads(s)
        t = a["type"]
        if t == "dahai":
            keys.append(str(a["pai"]))
        elif t == "none":
            keys.append("pass")
        elif t == "hora":
            keys.append("hora")
        elif t == "reach":
            keys.append("reach")
        elif t == "pon":
            keys.append(f"pon:{a['pai']}")
        elif t == "chi":
            cons = "".join(a.get("consumed") or [])
            keys.append(f"chi:{cons}<-{a['pai']}")
        elif t == "daiminkan":
            keys.append(f"dakan:{a['pai']}")
        elif t == "kakan":
            keys.append(f"kakan:{a['pai']}")
        elif t == "ankan":
            pai = a.get("pai") or (a.get("consumed") or ["?"])[0]
            keys.append(f"ankan:{pai}")
        else:  # 未知动作类型：原样保底，绝不破坏 1:1
            keys.append(f"{t}:{a.get('pai', '')}")
    return keys


def laya_question(legal: list[str]) -> dict:
    """build_sequence 的 q 参数（{t,ins,crit}）。value=None -> 选项文本=key。

    同名 key（理论上不重复）加 '#n' 后缀消歧，仍与 legal 1:1。
    """
    fixed: dict[str, None] = {}
    seen: dict[str, int] = {}
    for k in options_from_legal(legal):
        if k in fixed:
            seen[k] = seen.get(k, 1) + 1
            k = f"{k}#{seen[k]}"
        fixed[k] = None
    return {"t": "choice", "ins": "choose your mahjong action", "crit": fixed}
