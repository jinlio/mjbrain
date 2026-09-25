"""档 0 独立建议窗口（M4 形态决策 2026-09-22：HUD 非必需，独立界面即可）。

用法 A（从已落盘的凤王语料抽任一前缀做"查一手"演练，零协议零风险）：
    python scripts/live_advisor.py --raw data/raw/tenhou_houou_mjai \\
        --file-index 0 --game-index 0 --until 120
用法 B（喂外部 mjai 事件流；雀魂官方导出经 riichienv MjSoulReplay 规整同格式）：
    python scripts/live_advisor.py --events kyoku.json --seat 2

在事件流截止处重演环境（engine/replay.py 同款 apply_event 驱动），对当前
所有开着决策窗的座位输出 B3 推荐 + 候选概率分布；若日志在截止后还有动作，
参考显示"实际下一手"，人机对照。M4 档 1 的捕获层只需把 MITM/CDP 事件灌
--events 的同一条路，推荐内核零改动复用。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def _open_windows(env):
    """当前对所有座位可见的决策窗：pid -> (ob, legal mjai 串列表)。"""
    out = {}
    for pid in range(env.num_players):
        try:
            ob = env.get_observation(pid)
        except Exception:  # noqa: BLE001,S112 — 非行动帧无观测
            continue
        if ob is None:
            continue
        la = [str(a.to_mjai()) for a in ob.legal_actions()]
        if la:
            out[pid] = (ob, la)
    return out


_WIND = {"E": 0, "S": 1, "W": 2, "N": 3}


def _tenpai_note(ob, pid: int, pai: str) -> str:
    """弃 pai 后是否听牌（仅门前无副露的出牌态有意义，有副露时诚实留空）。"""
    from riichienv.convert import mjai_to_tid_list, tid_to_mjai
    from riichienv.hand import Conditions, HandEvaluator

    if ob.melds is not None and len(list(ob.melds[pid])) > 0:
        return ""
    hand = list(ob.hand)
    if (len(hand) - 1) % 3 != 1:
        return ""
    want = set(mjai_to_tid_list([pai]))
    rest = None
    for i, t in enumerate(hand):
        if t in want:
            rest = hand[:i] + hand[i + 1:]
            break
    if rest is None:
        return ""
    try:
        he = HandEvaluator(rest)
        if not he.is_tenpai():
            return "  未听"
        # 本版本 riichienv 的 get_waits 不可靠（对照测试返回错牌），
        # 用 calc 枚举 34 种牌的有和型判定代劳。
        reps: dict[str, int] = {}
        for t in range(136):
            reps.setdefault(tid_to_mjai(t), t)
        waits = set()
        for x in reps.values():  # 每种牌（含赤五独立种）一枚代表
            try:
                r = he.calc(x, conditions=Conditions(tsumo=True))
                if getattr(r, "has_win_shape", False):
                    waits.add(tid_to_mjai(x))
            except Exception:  # noqa: BLE001,S112 —— 个别枚判定失败跳过
                continue
        return "  听:" + " ".join(sorted(waits)) if waits else "  听:"
    except Exception:  # noqa: BLE001 — 牌型评估失败不阻断推荐输出
        return ""


def _manual_env(args):
    """手输模式：BoardSpec → build_events → 重放 → 自家决策窗。"""
    from riichienv import RiichiEnv

    from advisor.build_state import BoardSpec, Furo, _tiles, build_events

    rivers = [_tiles(row) for row in args.rivers.split("|")]
    if len(rivers) != 4:
        raise SystemExit("--rivers 须恰有 3 个 '|' 分 4 段")
    last = None
    if args.last:
        pai, _, frm = args.last.partition("<-")
        last = (pai.strip(), int(frm))
    spec = BoardSpec(
        seat=args.seat, hand=_tiles(args.hand), rivers=rivers,
        furos=[Furo.parse(s) for s in args.furo],
        draw=args.draw, last=last, oya=args.oya, round_wind=_WIND[args.bakaze],
        kyoku=args.kyoku, honba=args.honba, sticks=args.sticks,
        scores=[int(x) for x in args.scores.split(",")], dora=args.dora)
    env = RiichiEnv(game_mode="4p-red-half")
    for e in build_events(spec):
        env.apply_event(e)
    return env, spec.seat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--raw", help="凤王 mjai 语料目录（含 data/*.parquet）")
    g.add_argument("--events", help="外部 mjai 事件数组 JSON 文件")
    g.add_argument("--hand", help="手输模式：门前手牌串（不含摸张），如 1m2m4m7p…")
    ap.add_argument("--file-index", type=int, default=0)
    ap.add_argument("--game-index", type=int, default=0)
    ap.add_argument("--until", type=int, default=-1, help="截止事件行（-1=整条流尾）")
    ap.add_argument("--seat", type=int, default=-1, help="只看某座位（-1=所有开窗家）")
    ap.add_argument(
        "--ckpt",
        default="checkpoints/m3-final-0.7410",
        help="B3 权重目录",
    )
    # ---- 手输模式参数（配 --hand）----
    ap.add_argument("--draw", default=None, help="本巡摸牌（与 --last 二选一）")
    ap.add_argument("--last", default=None,
                    help="鸣牌响应：刚被打出的牌 'pai<-frm'，如 '5m<-1'")
    ap.add_argument("--rivers", default="|||",
                    help="四家河 '|' 分隔：'E 3p|9s|W|1m'")
    ap.add_argument("--furo", action="append", default=[],
                    help="副露，可重复：'actor|kind|pai|frm|consumed|pos=n'")
    ap.add_argument("--oya", type=int, default=0)
    ap.add_argument("--bakaze", default="E", choices=list(_WIND))
    ap.add_argument("--kyoku", type=int, default=1)
    ap.add_argument("--honba", type=int, default=0)
    ap.add_argument("--sticks", type=int, default=0, help="立直棒数")
    ap.add_argument("--scores", default="25000,25000,25000,25000")
    ap.add_argument("--dora", default="1m", help="宝牌指示牌")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", default=None,
                    help="cpu/cuda/mps（手输模式默认 cpu，不打扰训练）")
    args = ap.parse_args()

    # ---- 局面来源：手输 / 棋谱重演 ----
    wins = None
    if args.hand:
        if args.seat < 0:
            raise SystemExit("手输模式必须给 --seat")
        env, _ = _manual_env(args)
        wins = _open_windows(env)
        if not wins:
            print("手输局面无法到达决策点（输入不自洽？见上方守恒/校验错误）。")
            return 0
        if args.seat not in wins:
            print(f"座位 {args.seat} 此刻无决策窗（开窗家：{sorted(wins)}）。")
            return 0
    else:
        if args.events:
            events = json.loads(pathlib.Path(args.events).read_text(encoding="utf-8"))
            if isinstance(events, dict):  # {"log":{"raw":[...]}} 官方导出外壳
                events = events.get("log", {}).get("raw") or events.get("events")
        else:
            import pyarrow.parquet as pq

            f = sorted(pathlib.Path(args.raw).glob("data/*.parquet"))[args.file_index]
            for b in pq.ParquetFile(f).iter_batches(batch_size=args.game_index + 1):
                rec = b.to_pylist()[args.game_index]
                break
            events = [json.loads(ln) for ln in rec["events"].splitlines()]

        cut = len(events) if args.until < 0 else min(args.until, len(events))

        from riichienv import GameRule, RiichiEnv

        env = RiichiEnv(game_mode="4p-red-half", rule=GameRule.default_mjsoul())
        applied = 0
        for ev in events[:cut]:
            try:
                env.apply_event(ev)
                applied += 1
            except Exception:  # noqa: BLE001 — 流尾环境要求补 pass 等：停在可解释处
                break
        wins = _open_windows(env)
        if not wins:
            print(f"截止行 {applied}：无决策窗开着（牌桌在等环境事件，不是等玩家）。")
            return 0

    # ---- B3 前向（与 /v1/react 同一内核 advisor.recommend.Adviser）----
    from advisor.recommend import Adviser

    adv = Adviser(args.ckpt, device=args.device)

    ref = None
    if args.hand is None:
        for ev in events[cut:]:
            if ev.get("type") in ("dahai", "pon", "chi", "daiminkan",
                                  "kakan", "ankan", "reach", "hora"):
                ref = ev
                break

    for pid, (ob, legal) in sorted(wins.items()):
        if args.seat >= 0 and pid != args.seat:
            continue
        try:
            ranked = adv.topk(ob, legal, top=args.topk)
        except ValueError as ex:
            print(f"座位{pid}: {ex}")
            continue
        print(f"座位{pid} 决策窗 {len(legal)} 项 → 推荐 {ranked[0][0]}")
        for desc, pr in ranked:
            note = ""
            if desc.startswith("dahai:"):
                note = _tenpai_note(ob, pid, desc.split(":", 1)[1])
            print(f"    {pr:6.1%}  {desc}{note}")
        if ref is not None and ref.get("actor", ref.get("who")) == pid:
            print(f"    （实际下一手: {ref['type']}"
                  f"{':' + str(ref.get('pai', '')) if ref.get('pai') else ''}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
