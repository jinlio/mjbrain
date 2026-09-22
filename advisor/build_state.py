"""静态局面 → 合法可行的本局 mjai 事件流（档0 核心）。

输入"牌桌快照"（手牌/河/副露/场况），输出可直接喂给
`RiichiEnv.apply_event` 的事件流；重放到末尾后
`get_observation(seat).legal_actions()` 即决策窗口的合法集。

不追求还原真实时间线的唯一性，只保证**合法且与快照逐项一致**
（河序/副露归属/手牌构成/场况），obs 与训练同源（同为 get_observation
产物），state_text 零漂移。

合成规则（v1：4 人、无立直——立直/3 人见 PLAN 后续）：
- 每家弃牌巡：tsumo(该巡河头, moqie)+dahai(同张)；鸣牌家：鸣牌事件后
  直接 dahai 其河下一张（跳过摸牌）。
- 对手起始手从未占用牌中虚构 13 张（被遮蔽，取法不影响任何家决策语义）。
- 决策者起始手 = 门前手 + 自己河 + 自己鸣牌消耗（精确可推）。
- chi/pon/daiminkan 按 (from, pai) 贪心定位到河的弃牌之后；ankan/kakan
  无法从终局河反推时点，需 `pos`（杠主在该家入河该张之前的张数）。
- 决策点二选一收尾：`draw`（我方摸进，14 张弃牌窗）或
  `last=(pai, from)`（他家刚打出，我方鸣牌/和牌窗）。

用 `riichienv.RiichiEnv` 自带校验当正确性闸门：任何不自洽（河/鸣牌/牌数
对不上）在 apply_event 处抛错，advisor 据此给出诚实的输入错误提示。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from riichienv.convert import mjai_to_tid_list

_ALL_KINDS = [
    *[f"{n}{s}" for s in ("m", "p", "s") for n in "123456789"],
    *[f"{n}z" for n in "1234567"],
]
_WINDS = ["E", "S", "W", "N"]
_REDS = {"5m": "5mr", "5p": "5pr", "5s": "5sr"}
_MELD_ORDER = ("chi", "pon", "daiminkan")  # 同一家一巡内鸣牌互斥，取第一个匹配


def _tid(t: str) -> int:
    return mjai_to_tid_list([t])[0]


def _base(t: str) -> str:
    """去赤后的基础牌（鸣牌匹配用：普通 5 可鸣赤 5）。"""
    return t.rstrip("r")


def _kind(t: str) -> str:
    """牌种键：赤五与普通五是不同种（每种赤牌全局仅 1 枚），牌数校验用。"""
    return t if len(t) == 3 else t.rstrip("r")


def _norm(t: str) -> str:
    t = t.strip()
    if len(t) == 2 and t[0] == "0":
        return _REDS.get(f"5{t[1]}", t)
    return t


def _tiles(s: str) -> list[str]:
    """牌串解析：'2s3s' / '2s 3s' / '2s,3s' / 含赤 '5pr5m' / 字牌 'E S W' → 列表。"""
    toks = re.findall(r"\d[mpsz]r?|[ESWNPFC]", s)
    if "".join(toks) != re.sub(r"[,\s]", "", s):
        raise ValueError(f"无法解析牌串: {s!r}")
    return [_norm(t) for t in toks]


@dataclass
class Furo:
    actor: int
    kind: Literal["chi", "pon", "daiminkan", "ankan", "kakan"]
    pai: str = ""
    frm: int = -1
    consumed: list[str] = field(default_factory=list)
    pos: int = -1

    @staticmethod
    def parse(s: str) -> Furo:
        """`actor|kind|pai|from|consumed…|pos=n`，尾部空段可省。

        例 `2|pon|5m|0`、`1|chi|1s|0|2s3s`、`3|ankan|5p|-|-|pos=4`、
        `0|kakan|5s|0`（加杠 pai=新加枚；consumed 自动=原 pon 的三张）。
        赤牌接受 `5pr` 或 `0p`。
        """
        parts = s.split("|")
        if len(parts) < 3:
            raise ValueError(f"furo 段数不足: {s!r}")
        kind = parts[1]
        if kind not in ("chi", "pon", "daiminkan", "ankan", "kakan"):
            raise ValueError(f"未知 furo kind: {kind!r}")
        actor = int(parts[0])
        pai = _norm(parts[2]) if parts[2] not in ("", "-") else ""
        frm = int(parts[3]) if len(parts) > 3 and parts[3] not in ("", "-") else -1
        consumed: list[str] = []
        pos = -1
        for tk in parts[4:]:
            m = re.fullmatch(r"pos=(\d+)", tk)
            if m:
                pos = int(m.group(1))
            else:
                consumed.extend(_tiles(tk))
        if kind in ("chi", "pon", "daiminkan") and (not pai or frm < 0):
            raise ValueError(f"{kind} 需 pai 与 from：{s!r}")
        if kind in ("ankan", "kakan") and not pai:
            raise ValueError(f"{kind} 需 pai：{s!r}")
        return Furo(actor, kind, pai, frm, consumed, pos)

    def mjai(self) -> dict:
        b = _base(self.pai)
        if self.kind == "chi":
            return {"type": "chi", "actor": self.actor, "target": self.frm,
                    "pai": self.pai, "consumed": list(self.consumed) or self._chi_default()}
        if self.kind == "pon":
            return {"type": "pon", "actor": self.actor, "target": self.frm,
                    "pai": self.pai, "consumed": self.consumed or [b, b]}
        if self.kind == "daiminkan":
            return {"type": "daiminkan", "actor": self.actor, "target": self.frm,
                    "pai": self.pai, "consumed": self.consumed or [b, b, b]}
        if self.kind == "ankan":
            return {"type": "ankan", "actor": self.actor,
                    "consumed": self.consumed or [b, b, b, b]}
        return {"type": "kakan", "actor": self.actor, "pai": self.pai,
                "consumed": self.consumed or [b, b, b]}

    def _chi_default(self) -> list[str]:
        n = int(self.pai[0])
        s = self.pai[1]
        if s == "z":
            raise ValueError("chi 不能鸣字牌")
        if n >= 3:
            return [f"{n - 2}{s}", f"{n - 1}{s}"]
        if n >= 2:
            return [f"{n - 1}{s}", f"{n + 1}{s}"]
        return [f"{n + 1}{s}", f"{n + 2}{s}"]

    def _dedup_consumed(self, n: int) -> list[str]:
        return self.consumed[:n] or [_base(self.pai)] * n


@dataclass
class BoardSpec:
    seat: int
    hand: list[str]
    rivers: list[list[str]] = field(default_factory=list)
    furos: list[Furo] = field(default_factory=list)
    draw: str | None = None
    last: tuple[str, int] | None = None
    oya: int = 0
    round_wind: int = 0
    kyoku: int = 1
    honba: int = 0
    sticks: int = 0
    scores: list[int] = field(default_factory=lambda: [25000] * 4)
    dora: str = "1m"

    def validate(self) -> None:
        own_meld = any(f.actor == self.seat for f in self.furos)
        if len(self.hand) < 1 or len(self.hand) > 13:
            raise ValueError(f"门前手须 1~13 张，得 {len(self.hand)}")
        if not own_meld and len(self.hand) != 13:
            # 无自家副露时门前必为整 13 张；有副露交给 build_events 守恒方程
            raise ValueError(
                f"无副露时手牌须 13 张门前（不含摸张），得 {len(self.hand)}")
        if len(self.rivers) != 4:
            raise ValueError("须给 4 家河（可空）")
        if (self.draw is None) == (self.last is None):
            raise ValueError("draw 与 last 必须二选一")
        if self.last:
            pai, frm = self.last
            rv = self.rivers[frm]
            if not rv or _base(rv[-1]) != _base(pai):
                raise ValueError(
                    f"last {pai}<-{frm} 须为 seat{frm} 河尾（现="
                    f"{rv[-1] if rv else '空'}）")
        for i, f in enumerate(self.furos):
            if f.kind == "chi" and f.actor != (f.frm + 1) % 4:
                raise ValueError(f"furo[{i}] chi 只能鸣上家")
            if f.kind in ("pon", "daiminkan") and f.actor == f.frm:
                raise ValueError(f"furo[{i}] 不能鸣自家")
            if f.kind in ("ankan", "kakan") and f.pos < 0:
                raise ValueError(
                    f"furo[{i}] {f.kind} 需 pos=<杠前该家已入河张数>（无法从终局河反推时点）")
        self._count_check()

    def _count_check(self) -> None:
        c: dict[str, int] = {}
        tiles: list[str] = list(self.hand)
        tiles += [t for rv in self.rivers for t in rv]  # last 枚即河尾，已计
        tiles += [x for x in (self.dora, self.draw or "") if x]
        for t in tiles:
            c[_kind(t)] = c.get(_kind(t), 0) + 1
        for f in self.furos:
            if f.actor == self.seat:
                for t in _hand_consume(f):
                    c[_kind(t)] = c.get(_kind(t), 0) + 1
                # 门前快照不含 consumed 枚；暗杠第 4 枚=岭上摸即切张，在河已计
            else:
                # 对手副露：consumed 不可见但占牌种，按最保守口径计
                for t in (f.consumed or []):
                    c[_kind(t)] = c.get(_kind(t), 0) + 1
        cap = {"5m": 3, "5p": 3, "5s": 3}  # 普通五各仅 3 枚（第 4 枚为赤）
        for kind, n in c.items():
            if n > cap.get(kind, 4):
                raise ValueError(f"{kind} 出现 {n} 次>{cap.get(kind, 4)}：输入不一致")


def _occupied(spec: BoardSpec, my_start: list[str]) -> dict[str, int]:
    """按精确口径统计占用的牌种数：决策者起手 + 全河（last 枚=河尾已含）
    + dora/摸张 + 对手副露的手牌消耗。"""
    c: dict[str, int] = {}
    def add(t: str) -> None:
        c[_kind(t)] = c.get(_kind(t), 0) + 1
    for t in my_start:
        add(t)
    for rv in spec.rivers:
        for t in rv:
            add(t)
    for extra in (spec.dora, spec.draw or ""):
        if extra:
            add(extra)
    for f in spec.furos:
        if f.actor != spec.seat:
            for t in (f.consumed or []):
                add(t)
    return c


_FAKE_CAP = {"5m": 3, "5p": 3, "5s": 3}  # 普通五仅 3 枚；赤五 1 枚


def _fake_start_from(used: dict[str, int], take: int, offset: int) -> list[str]:
    """从共享 used 池虚构 take 张对手起手，边取边递减（跨调用不互撞）。"""
    avail: dict[str, int] = {
        k: _FAKE_CAP.get(k, 4) - used.get(k, 0) for k in _ALL_KINDS}
    keys = list(_ALL_KINDS)
    out: list[str] = []
    i = 0
    guard = 0
    while len(out) < take:
        guard += 1
        if guard > 400:
            raise ValueError("牌面不足虚构对手起手（输入异常）")
        k = keys[(i + offset) % len(keys)]
        if avail.get(k, 0) > 0:
            out.append(k)
            avail[k] -= 1
            used[k] = used.get(k, 0) + 1  # 共享池递减：后续对手避让
        i += 1
    return sorted(out, key=_tid)


def _hand_consume(f: Furo) -> list[str]:
    """该副露从**起始手**消耗的牌（不含鸣到的 pai、不含当巡摸进又进杠的张）。

    chi/pon=2、daiminkan=3、kakan=0（加杠枚=当巡摸牌；原 pon 的 2 枚已计）、
    ankan=4（杠在巡首摸牌前宣告，4 枚全从起手来；岭上摸=当巡"摸即切"张）。
    """
    if f.kind in ("chi", "pon"):
        return (f.consumed or [_base(f.pai)] * 2)[:2]
    if f.kind == "daiminkan":
        return (f.consumed or [_base(f.pai)] * 3)[:3]
    if f.kind == "ankan":
        return (f.consumed or [_base(f.pai)] * 4)[:4]
    return []  # kakan


def _turn_plan(spec: BoardSpec, events: list[dict]) -> dict[int, list[str]]:
    """沿时间线发射事件直到决策窗口，返回 ys。

    ys[pid] = pid 在"鸣牌后无摸直接弃"的巡里弃掉的河张——这些牌来自起始手，
    必须进起手方程；其余弃张按"摸即切"虚构（当巡摸进，不留手）。
    """
    river_i = [0] * 4
    melds = [f for f in spec.furos if f.kind in _MELD_ORDER]
    kans = sorted((f for f in spec.furos if f.kind in ("ankan", "kakan")),
                  key=lambda f: (f.actor, f.pos))
    ys: dict[int, list[str]] = {p: [] for p in range(4)}

    def eat(pid: int) -> str | None:
        if river_i[pid] < len(spec.rivers[pid]):
            t = spec.rivers[pid][river_i[pid]]
            river_i[pid] += 1
            return t
        return None

    def emit(ev: dict) -> None:
        events.append(ev)

    def last_is(pid: int, tile: str) -> bool:
        return bool(spec.last and pid == spec.last[1]
                    and _base(tile) == _base(spec.last[0])
                    and river_i[pid] == len(spec.rivers[pid]))

    STOP, ADV, PASS = "stop", "adv", "pass"
    cur = spec.oya % 4

    def all_drained() -> bool:
        return all(river_i[p] >= len(spec.rivers[p]) for p in range(4))

    def place_melds_after(pid: int, tile: str) -> str:
        """pid 刚弃 tile。一张牌至多一次鸣：鸣牌家立即弃 y（起手牌，计 ys），
        巡次跳鸣牌家下家；y 可再被鸣（递归）。cur 修改后由主循环 continue。"""
        nonlocal cur
        if last_is(pid, tile):
            return STOP
        for j, f in enumerate(melds):
            if f.frm == pid and _base(f.pai) == _base(tile):
                emit(f.mjai())
                melds.pop(j)
                if river_i[f.actor] >= len(spec.rivers[f.actor]):
                    raise ValueError(
                        f"{f.kind} actor{f.actor} 鸣 {tile} 后无河张可打——河/鸣牌不自洽")
                y = eat(f.actor)
                if f.kind == "daiminkan":
                    # 明杠后弃张=岭上摸即切（非起手牌，不进 ys）
                    emit({"type": "tsumo", "actor": f.actor, "pai": y})
                    emit({"type": "dahai", "actor": f.actor, "pai": y,
                          "tsumogiri": True})
                else:
                    ys[f.actor].append(y)
                    emit({"type": "dahai", "actor": f.actor, "pai": y,
                          "tsumogiri": False})  # 鸣后弃的是起手牌，非摸张
                cur = (f.actor + 1) % 4
                r = place_melds_after(f.actor, y)  # y 再被鸣则继续嵌套
                return STOP if r == STOP else ADV
        return PASS

    def place_kans_before(pid: int) -> str:
        """pid 巡首、入河张数==pos 时宣告杠；岭上摸=该家河下一张（摸即切）。
        河已尽时杠后无岭上张可发（决策点 draw 即岭上，由调用方补 tsumo）。"""
        nonlocal cur
        placed = False
        progressed = True
        while progressed:
            progressed = False
            for j, f in enumerate(kans):
                if f.actor == pid and river_i[pid] == f.pos:
                    emit(f.mjai())
                    kans.pop(j)
                    placed = True
                    if river_i[pid] >= len(spec.rivers[pid]):
                        break
                    nxt = eat(pid)
                    emit({"type": "tsumo", "actor": pid, "pai": nxt})
                    emit({"type": "dahai", "actor": pid, "pai": nxt,
                          "tsumogiri": True})
                    cur = (pid + 1) % 4  # 杠巡以岭上弃结束
                    r = place_melds_after(pid, nxt)
                    if r == STOP:
                        return STOP
                    progressed = True
                    break
        return ADV if placed else PASS

    guard = 0
    while True:
        guard += 1
        if guard > 1000:
            raise RuntimeError("时间线发散：输入不自洽")
        # draw 窗口：全河排尽、轮到 me 巡首（我上巡弃牌后各家均已摸打）
        if spec.draw and cur == spec.seat and all_drained():
            place_kans_before(cur)  # 巡首杠：岭上摸=draw
            emit({"type": "tsumo", "actor": cur, "pai": spec.draw})
            return ys
        r = place_kans_before(cur)
        if r == STOP:
            return ys
        if r == PASS:
            if river_i[cur] < len(spec.rivers[cur]):
                tile = eat(cur)
                emit({"type": "tsumo", "actor": cur, "pai": tile})
                emit({"type": "dahai", "actor": cur, "pai": tile,
                      "tsumogiri": True})
                r2 = place_melds_after(cur, tile)
                if r2 == STOP:
                    return ys
                if r2 == PASS:
                    cur = (cur + 1) % 4
            elif all_drained() and not melds and not kans and not spec.draw:
                return ys  # last 窗口河尽兜底（draw 窗口须推进到 seat 巡首）
            else:
                cur = (cur + 1) % 4
        # ADV：巡次已由杠/鸣牌推进




def build_events(spec: BoardSpec) -> list[dict]:
    """合成到决策点为止的本局事件流（以 start_game 打头）。

    起手守恒：|门前手| + Σ(己方副露的手牌消耗) + |ys(鸣后即打的河张)| == 13；
    不满足=输入不自洽（河/副露对不上），诚实报错。
    """
    spec.validate()
    me = spec.seat
    events: list[dict] = []
    ys = _turn_plan(spec, events=events)

    my_start = list(spec.hand) + [t for f in spec.furos if f.actor == me for t in _hand_consume(f)] + ys[me]
    if len(my_start) != 13:
        raise ValueError(
            f"局面不自洽：门前{len(spec.hand)}+鸣牌消耗+鸣后弃张={len(my_start)}≠13——"
            "检查河是否含全部被鸣张、副露归属与张数")
    my_start = sorted(my_start, key=_tid)

    tehais: list[list[str]] = []
    used = _occupied(spec, my_start)
    for p in range(4):
        if p == me:
            tehais.append(my_start)
        else:
            fh = _fake_start_from(used, 13, offset=p)
            tehais.append(fh)

    kyoku_event = {
        "type": "start_game", "names": [f"P{i}" for i in range(4)]}
    events = [kyoku_event, {
        "type": "start_kyoku", "bakaze": _WINDS[spec.round_wind % 4],
        "kyoku": spec.kyoku, "honba": spec.honba, "kyotaku": spec.sticks,
        "oya": spec.oya, "dora_marker": spec.dora,
        "scores": list(spec.scores), "tehais": tehais}] + events
    return events
