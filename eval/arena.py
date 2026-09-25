"""竞技场 CLI（M1）。

    conda activate mjbrain
    python -m eval.arena --bots B1,B0,B0,B0 --games 200 --seed-base 20260921

方法学：
- 四席位填四个 bot 名（同库允许重复，各自独立实例）；
- 每个种子先按给定顺序打一局，再做 3 次轮转共 4 局，席位效应对所有 bot 平均掉；
- pt 表：tenhou = +90/+45/0/-135 顺位马（25000 起分，
  （按最终分排名，并列按座位序稳定打破）；majsoul = 终局净点/1000 + 马 ±15/±5
  （雀魂段位场公式，不含底徽分项；净点项使避四大Top 的边际激励显式进表）；
- 聚合：1位率/前二率/平均顺位/每局平均净pt，CI 按种子聚类（正态近似 95%）。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from engine.sim import MahjongSim
from eval import make_bot

PT = {0: 90.0, 1: 45.0, 2: 0.0, 3: -135.0}
UMA_MAJSOUL = (15.0, 5.0, -5.0, -15.0)  # 雀魂马点（一位/二位/三位/四位）
ORIGIN = 25000.0  # 4 麻起分（两平台一致）


def _pt_fn(table: str):
    """rank, score -> 该局 pt。"""
    if table == "tenhou":
        return lambda rank, score: PT[rank]
    if table == "majsoul":
        return lambda rank, score: (score - ORIGIN) / 1000.0 + UMA_MAJSOUL[rank]
    raise SystemExit(f"未知 --pt-table: {table}")


_PT_TENHOU = _pt_fn("tenhou")
_PT_MAJSOUL = _pt_fn("majsoul")


def play_seed(specs: list[str], seed: int, hanchan: bool) -> list[dict]:
    """一个种子 = 4 次席位轮转。返回每 bot 的聚合 dict。

    两 pt 表（tenhou 顺位马 / majsoul 净点+马）对同一批终局分数做纯函数
    计分，单遍同场双计——省一次全量重跑，且两口径天然同种子严格配对。
    """
    rows = [
        {
            "pts_tenhou": 0.0,
            "pts_majsoul": 0.0,
            "games": 0,
            "rank_sum": 0,
            "top1": 0,
            "top2": 0,
        }
        for _ in specs
    ]
    for rot in range(4):
        order = [specs[(i - rot) % 4] for i in range(4)]
        bots = [make_bot(s, i) for i, s in enumerate(order)]
        sim = MahjongSim(seed=seed * 4 + rot, hanchan=hanchan)
        res = sim.play(bots)
        # 最终分排名（并列按座位号稳定打破）
        ranking = sorted(range(4), key=lambda p: (-res.scores[p], p))
        rank_of = {p: r for r, p in enumerate(ranking)}
        for i in range(4):
            r = rank_of[i]
            owner = (i - rot) % 4  # seat i 上的 bot 来自 specs 的原位置
            sc = float(res.scores[i])
            row = rows[owner]
            row["pts_tenhou"] += _PT_TENHOU(r, sc)
            row["pts_majsoul"] += _PT_MAJSOUL(r, sc)
            row["games"] += 1
            row["rank_sum"] += r + 1
            row["top1"] += r == 0
            row["top2"] += r <= 1
    return rows


def _mean_ci95(vals: list[float]) -> tuple[float, float]:
    m = statistics.fmean(vals)
    if len(vals) < 2:
        return m, 0.0
    se = statistics.stdev(vals) / math.sqrt(len(vals))
    return m, 1.96 * se


def run(
    specs: list[str],
    seeds: int,
    seed_base: int,
    hanchan: bool,
    verbose: bool = True,
):
    per_seed = {
        t: [[] for _ in specs] for t in ("tenhou", "majsoul")
    }  # 每表：每 bot 每种子 pt 合计
    agg = [
        {"games": 0, "pts_tenhou": 0.0, "pts_majsoul": 0.0, "rank_sum": 0, "top1": 0, "top2": 0}
        for _ in specs
    ]
    t0 = time.time()
    for g in range(seeds):
        rows = play_seed(specs, seed_base + g, hanchan)
        for i, r in enumerate(rows):
            per_seed["tenhou"][i].append(r["pts_tenhou"])
            per_seed["majsoul"][i].append(r["pts_majsoul"])
            agg[i]["games"] += r["games"]
            agg[i]["pts_tenhou"] += r["pts_tenhou"]
            agg[i]["pts_majsoul"] += r["pts_majsoul"]
            agg[i]["rank_sum"] += r["rank_sum"]
            agg[i]["top1"] += r["top1"]
            agg[i]["top2"] += r["top2"]
        if verbose and (g + 1) % 25 == 0:
            print(f"[{(g + 1) * 4} games / {g + 1} seeds] {time.time() - t0:.0f}s", flush=True)
    return per_seed, agg, time.time() - t0


def table(per_seed, agg, specs, primary: str = "majsoul") -> list[dict]:
    out = []
    for i, spec in enumerate(specs):
        g = agg[i]["games"]
        row = {
            "bot": spec,
            "games": g,
            "top1_rate": round(agg[i]["top1"] / g, 4),
            "top2_rate": round(agg[i]["top2"] / g, 4),
            "avg_rank": round(agg[i]["rank_sum"] / g, 3),
        }
        for t in ("tenhou", "majsoul"):
            m, ci = _mean_ci95([x / 4 for x in per_seed[t][i]])
            row[f"pt_per_game_{t}"] = round(m, 3)
            row[f"ci95_{t}"] = round(ci, 3)
        out.append(row)
    return sorted(out, key=lambda r: -r[f"pt_per_game_{primary}"])


def main() -> int:
    # 报表含 →/± 等字符：重定向到管道/文件在 Windows 本地编码下不许炸
    import sys
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bots", required=True, help="逗号分隔 4 个席位，如 B1,B0,B0,B0")
    ap.add_argument("--games", type=int, default=100, help="种子数（实际对局 = x4）")
    ap.add_argument("--seed-base", type=int, default=20260921)
    ap.add_argument("--tonpuusen", action="store_true", help="东风战（默认半庄）")
    ap.add_argument("--pt-table", choices=("tenhou", "majsoul"), default="majsoul",
                    help="主排序口径（两表总是同场双计）：tenhou 顺位马 / majsoul 净点+±15/±5")
    ap.add_argument("--out", type=Path, default=None, help="报告 JSON 输出路径")
    args = ap.parse_args()

    specs = [s.strip() for s in args.bots.split(",")]
    if len(specs) != 4:  # 显式校验而非 assert：python -O 下 assert 会被剥离
        raise SystemExit(f"--bots 需要恰好 4 个席位，得 {len(specs)}")
    if args.games < 1:  # 0/负数种子 → table() 除零、fmean([]) 裸 traceback
        raise SystemExit(f"--games 须为 >=1 的种子数，得 {args.games}")
    per_seed, agg, secs = run(specs, args.games, args.seed_base, not args.tonpuusen)
    rows = table(per_seed, agg, specs, primary=args.pt_table)

    hdr = (f"{'bot':<12} {'games':>6} {'top1%':>7} {'top2%':>7} {'rank':>6} "
           f"{'pt/game MS':>10} {'±CI':>7} {'pt/game TH':>10} {'±CI':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['bot']:<12} {r['games']:>6} {r['top1_rate']*100:>6.1f}% {r['top2_rate']*100:>6.1f}% "
              f"{r['avg_rank']:>6.3f} {r['pt_per_game_majsoul']:>10.3f} {r['ci95_majsoul']:>7.3f} "
              f"{r['pt_per_game_tenhou']:>10.3f} {r['ci95_tenhou']:>7.3f}")
    print(f"\n{args.games} seeds x4 rotations, both pt tables (primary={args.pt_table}), {secs:.0f}s")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"bots": specs, "seeds": args.games, "hanchan": not args.tonpuusen,
             "pt_table": "both", "primary": args.pt_table,
             "seed_base": args.seed_base, "seconds": round(secs, 1), "results": rows},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
