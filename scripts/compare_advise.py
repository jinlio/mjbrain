#!/usr/bin/env python
"""M4 档1 验收工具：推荐记录 × 实际打法逐窗对照（模型推荐 vs 真人出牌）。

输入 = 两个 JSONL（都由现成脚本产出，本工具只读、不联网、不调模型）：
  --frames  run_capture.py 落盘的帧 JSONL（解出全量 mjai 事件流 + 自家座位）
  --advise  live_from_capture.py 落盘的推荐 JSONL（默认 <frames 同名>.advise.jsonl）

对齐原理：推荐行 `event_n` = 发请求那一刻的事件数，故"该座在其后的第一个
动作事件"就是玩家对该决策窗的回应（打牌/鸣牌/立直；跳过=该座下一个事件是
摸牌 tsumo）；把模型 raw top-1（mjai 描述串）与该动作比对即得逐窗一致率。
立直窗另比"宣言牌"：reach 事件之后该座的第一个打牌 vs 服务端第二段建议。

一致率只说明"模型建议与本人打法吻合度"——验收用途是回归信号与失配取证，
不是质量判据（打法不同未必错）。

用法：
  python scripts/compare_advise.py --frames data/raw/ms_frames/run6.jsonl
  python scripts/compare_advise.py --frames .../run6.jsonl --json runs/live-run6-vs.json
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# 该座的动作事件（不含 tsumo 摸牌本身算"跳过"的判据，故一并列入）
SELF_ACTION_TYPES = {"dahai", "reach", "chi", "pon", "daiminkan", "ankan", "kakan",
                     "hora", "kita", "tsumo", "ryukyoku", "kyuushu_kyuuhai"}

# 赤牌两种记法归一到普通 5（比对只认点数：赤 5 与普通 5 视为同张）
_RED = {"0m": "5m", "0p": "5p", "0s": "5s",
        "5mr": "5m", "5pr": "5p", "5sr": "5s"}


def norm_tile(t: str) -> str:
    return _RED.get(t, t)


def load_events(frames: pathlib.Path, seat: int | None):
    """帧 JSONL → (mjai 事件流, 状态机)。解码失败/噪声行静默跳过（同 live 客户端）。"""
    from capture.liqi import runtime

    p, st = runtime.make_stack()
    if seat is not None:
        st.seat = seat
    events: list[dict] = []
    for line in frames.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            fr = p.parse(base64.b64decode(json.loads(line)["b64"]))
        except Exception:  # noqa: BLE001,S112 —— 噪声/截断行
            continue
        if fr is None:
            continue
        try:
            events.extend(st.dispatch(fr.kind, fr.method, fr.payload))
        except Exception:  # noqa: BLE001,S112 —— 丢帧不丢整局
            continue
    return events, st


def next_self_action(events: list[dict], start: int, seat: int):
    """该座在 start 之后的第一个动作事件（= 对该决策窗的回应）。"""
    for i in range(start, len(events)):
        e = events[i]
        if e.get("actor") == seat and e.get("type") in SELF_ACTION_TYPES:
            return i, e
    return None, None


def boundary_between(events: list[dict], a: int, b: int) -> bool:
    """(a,b) 之间夹着 start_game/start_kyoku → 跨局或断线重连的重放段。

    雀魂断线重连会下发 syncGame，把当前一局的 actions 从头重放，于是捕获流里同一
    局出现两遍（后一遍为准，重放引擎结果正确，但"窗→回应"的对齐会错位）。
    """
    return any(e.get("type") in ("start_game", "start_kyoku") for e in events[a:b])


def raw_desc(a: dict) -> str:
    """mjai 动作 -> 模型描述串（与 advisor.recommend.Adviser._desc 同规则）。"""
    d = a["type"]
    if a.get("pai"):
        d += f":{a['pai']}"
    return d


def norm_desc(d: str) -> str:
    """描述串里的牌归一（赤 5 → 5），用于 top-k 命中判定。"""
    t, _, p = d.partition(":")
    return f"{t}:{norm_tile(p)}" if p else d


def actual_desc(ev: dict) -> str:
    """实际动作 -> 与模型选项同一词表：跳过窗的实际动作是摸牌（tsumo→none）。"""
    t = ev.get("type")
    if t == "tsumo":
        return "none"
    if t == "ankan":
        cons = ev.get("consumed") or []
        return norm_desc(f"ankan:{cons[0]}") if cons else "ankan:?"
    return norm_desc(raw_desc(ev))


def category(rec: str) -> str:
    if rec.startswith("dahai:"):
        return "打牌"
    if rec == "none":
        return "跳过"
    if rec == "reach":
        return "立直"
    if rec == "hora":
        return "和牌"
    return "鸣牌"


def judge(rec: str, ev: dict, ev_idx: int, events: list[dict], seat: int):
    """(是否一致, 实际动作描述, 附加说明)。ev=None 表示无后续（对不上，不计分）。"""
    if ev is None:
        return None, "（该窗之后无自家动作）", ""
    t = ev.get("type")
    if rec.startswith("dahai:"):
        want = norm_tile(rec.split(":", 1)[1])
        if t == "dahai":
            note = "摸切" if ev.get("tsumogiri") else ""
            return want == norm_tile(ev.get("pai", "")), f"dahai {ev.get('pai')}", note
        if t == "tsumo":  # 该座没打牌先摸了牌：窗与回应没对齐（如换局座位错配）
            return False, "跳过（先摸牌）", "对齐可疑"
        return False, f"{t} {ev.get('pai', '')}", ""
    if rec == "none":
        return t == "tsumo", ("跳过（下一手摸牌）" if t == "tsumo" else f"{t} {ev.get('pai', '')}"), ""
    if rec == "reach":
        return t == "reach", "reach", ""
    if rec == "hora":
        return t == "hora", "hora", ""
    # 鸣牌：chi/pon/daiminkan/kakan/ankan 比动作类型 + 被鸣的牌
    kind = rec.split(":", 1)[0]
    want = norm_tile(rec.split(":", 1)[1]) if ":" in rec else ""
    got_pai = norm_tile(ev.get("pai", "") or "")
    consumed = [norm_tile(x) for x in (ev.get("consumed") or [])]
    if kind == "ankan":
        return (t == "ankan" and want in consumed), f"ankan {ev.get('consumed')}", ""
    return (t == kind and want == got_pai), f"{t} {ev.get('pai', '')}", ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", required=True, type=pathlib.Path)
    ap.add_argument("--advise", type=pathlib.Path, default=None)
    ap.add_argument("--seat", type=int, default=-1,
                    help="自家座位；缺省取帧流里 authGame 识别的")
    ap.add_argument("--max-examples", type=int, default=10, help="失配样例打印条数")
    ap.add_argument("--json", type=pathlib.Path, default=None, help="汇总落盘路径")
    args = ap.parse_args(argv)

    adv_path = args.advise or args.frames.with_name(args.frames.stem + ".advise.jsonl")
    if not args.frames.exists() or not adv_path.exists():
        print(f"缺文件：{args.frames if not args.frames.exists() else adv_path}",
              file=sys.stderr)
        return 1

    events, st = load_events(args.frames, args.seat if args.seat >= 0 else None)
    seat = args.seat if args.seat >= 0 else st.seat
    rows = [json.loads(l) for l in adv_path.read_text(encoding="utf-8").splitlines()
            if l.strip()]

    stats = {"frames_events": len(events), "seat": seat, "advise_rows": len(rows),
             "窗关闭": 0, "无建议": 0, "不可判": 0, "可判": 0, "一致": 0, "失配": 0,
             "在top3": 0, "在top5": 0, "独立窗": 0, "独立窗一致": 0, "边界": 0}
    seats_seen: dict[int, int] = {}
    seen_keys: set = set()
    by_cat: dict[str, list[int]] = {}
    mis_examples: list[dict] = []
    reach_n = reach_ok = decl_n = decl_ok = 0
    rtts: list[float] = []

    for r in rows:
        if isinstance(r.get("rtt_s"), (int, float)):
            rtts.append(float(r["rtt_s"]))
        decs = [d for d in (r.get("resp") or {}).get("decisions", [])]
        picked = next((d for d in decs if d.get("seat") == seat),
                      decs[0] if decs else None)
        if picked is None or picked.get("window") is False or "recommend" not in picked:
            stats["窗关闭" if (picked and picked.get("window") is False) else "无建议"] += 1
            continue
        rec = picked["recommend"]
        # 窗主座位 = resp 里该决策的 seat（换局会变！用追踪座位去配 game1 的窗会错配）
        tgt = int(picked.get("seat", seat))
        seats_seen[tgt] = seats_seen.get(tgt, 0) + 1
        ev_i, ev = next_self_action(events, int(r.get("event_n", 0)), tgt)
        n_window = int(r.get("event_n", 0))
        if (ev is not None and rec != "none"
                and boundary_between(events, n_window, ev_i)):
            stats["边界"] += 1     # 动作型窗跨局/跨重连：捕获流里没有对应回应
            continue
        ok, act, note = judge(rec, ev, ev_i, events, tgt)
        if ok is None:
            stats["不可判"] += 1
            continue
        key = (ev_i, rec)
        dup = key in seen_keys     # 同一窗重复发帖（去重后才是独立窗数）
        seen_keys.add(key)
        cat = category(rec)
        c = by_cat.setdefault(cat, [0, 0])
        c[0] += 1
        stats["可判"] += 1
        stats["一致" if ok else "失配"] += 1
        if not dup:
            stats["独立窗"] += 1
            stats["独立窗一致"] += ok
        # 宽容口径：实际打法是否落在模型 top-3 / top-5 内
        if ev is not None:
            tops = [norm_desc(x["a"]) for x in picked.get("top", [])]
            got = actual_desc(ev)
            stats["在top3"] += got in tops[:3]
            stats["在top5"] += got in tops
        if ok:
            c[1] += 1
        elif len(mis_examples) < args.max_examples:
            mis_examples.append({
                "event_n": r.get("event_n"), "窗口": cat,
                "模型": (r.get("zh_recommend") or rec), "raw": rec,
                "实际": act, "备注": note,
                "top": [x["a"] for x in picked.get("top", [])[:3]]})
        # 立直窗附加：宣言牌（第二段建议 vs reach 后的第一个打牌）
        if cat == "立直":
            reach_n += 1
            reach_ok += ok
            rw = (picked.get("reach") or {}).get("recommend")
            if ok and rw:
                j, da = next_self_action(events, (ev_i or 0) + 1, tgt)
                if da is not None and da.get("type") == "dahai":
                    decl_n += 1
                    if norm_tile(rw.split(":", 1)[1]) == norm_tile(da.get("pai", "")):
                        decl_ok += 1
                    else:
                        mis_examples.append({
                            "event_n": r.get("event_n"), "窗口": "宣言牌",
                            "模型": (r.get("zh_reach") or {}).get("recommend", rw),
                            "raw": rw, "实际": f"dahai {da.get('pai')}", "备注": "",
                            "top": [x["a"] for x in
                                    (picked.get("reach") or {}).get("top", [])[:3]]})

    rtts.sort()
    def pct(q: float) -> float:
        return round(rtts[min(len(rtts) - 1, int(q * len(rtts)))], 3) if rtts else 0.0

    out = {
        "frames": str(args.frames), "advise": str(adv_path),
        "events": stats["frames_events"], "seat": seat,
        "advise_rows": stats["advise_rows"],
        "窗口关闭": stats["窗关闭"], "无建议": stats["无建议"],
        "不可判": stats["不可判"], "可判": stats["可判"],
        "一致": stats["一致"], "失配": stats["失配"],
        "独立窗": stats["独立窗"], "独立窗一致": stats["独立窗一致"],
        "跨局/重连边界": stats["边界"],
        "窗主座位": seats_seen,
        "一致率": round(stats["一致"] / stats["可判"], 4) if stats["可判"] else None,
        "独立窗一致率": (round(stats["独立窗一致"] / stats["独立窗"], 4)
                    if stats["独立窗"] else None),
        "实际在top3率": round(stats["在top3"] / stats["可判"], 4) if stats["可判"] else None,
        "实际在top5率": round(stats["在top5"] / stats["可判"], 4) if stats["可判"] else None,
        "分类": {k: {"可判": v[0], "一致": v[1]} for k, v in sorted(by_cat.items())},
        "立直窗": {"可判": reach_n, "一致": reach_ok,
                   "宣言牌可比": decl_n, "宣言牌一致": decl_ok},
        "rtt_s": {"n": len(rtts), "p50": pct(0.50), "p90": pct(0.90),
                  "max": rtts[-1] if rtts else None},
        "失配样例": mis_examples,
    }
    print(f"# 推荐 vs 实际（{args.frames.name}，座位 {seat}）")
    print(f"事件 {out['events']} ｜ 推荐记录 {out['advise_rows']} 行"
          f"（窗关闭 {out['窗口关闭']} / 无建议 {out['无建议']}）")
    print(f"可判 {out['可判']} 窗 → 一致 {out['一致']}、失配 {out['失配']}"
          f"（一致率 {out['一致率']}）；不可判 {out['不可判']}")
    print(f"独立窗去重后 {out['独立窗']} 个 → 一致 {out['独立窗一致']}"
          f"（{out['独立窗一致率']}）；窗主座位分布 {out['窗主座位']}")
    if out["跨局/重连边界"]:
        print(f"剔除跨局/断线重连边界行 {out['跨局/重连边界']} 条（捕获流里无对应回应）")
    print(f"宽容口径：实际打法落在模型 top-3 内 {out['实际在top3率']}、"
          f"top-5 内 {out['实际在top5率']}")
    for k, v in out["分类"].items():
        print(f"  - {k}: {v['一致']}/{v['可判']}")
    rc = out["立直窗"]
    print(f"立直窗 {rc['一致']}/{rc['可判']}"
          + (f"；宣言牌 {rc['宣言牌一致']}/{rc['宣言牌可比']}" if rc["宣言牌可比"] else ""))
    rs = out["rtt_s"]
    print(f"往返 {rs['n']} 次：p50 {rs['p50']}s / p90 {rs['p90']}s / max {rs['max']}s")
    if mis_examples:
        print("失配样例：")
        for m in mis_examples:
            print(f"  @事件{m['event_n']} {m['窗口']}：模型 {m['模型']}（raw {m['raw']}）"
                  f" vs 实际 {m['实际']} {m['备注']}  候选={m['top']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"汇总 -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())