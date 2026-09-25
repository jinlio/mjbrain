"""fp32↔fp16 平价审计：同核两档在同一场棋谱上的 top1 翻盘率与概率漂移。

背景：m3-final 发布成绩 top1 74.1% 以 arena B3（CUDA fp16）测得，而线上
advisor 历史口径是 fp32。两者现已共用 brain.infer 唯一前向核，只差精度
参数一个档。本脚本对每个决策窗各跑两档 topk，产出翻盘率/漂移统计，
作为"线上是否可切 fp16 口径"的测量依据。

规则口径与线上一致（雀魂 half）。非 CUDA 环境 fp16 档自动等同 fp32，
脚本会显式提示——真正有意义的审计要在 Windows（CUDA）主机上跑。

用法：
  python scripts/precision_parity.py --ckpt checkpoints/<run_id> \
      --events run_events.json [--seat 2]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    # 中文报表重定向到文件/管道在 Windows 本地编码下不许炸（全库统一模式）
    import sys as _sys
    for _s in (_sys.stdout, _sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", required=True, type=pathlib.Path,
                    help="单场完整 mjai 事件数组 JSON（与 /v1/react 输入同源）")
    ap.add_argument("--ckpt", required=True, type=pathlib.Path)
    ap.add_argument("--seat", type=int, default=-1, help="只统计该座位（-1 全座位）")
    ap.add_argument("--max-list", type=int, default=50, help="翻盘明细最多列几条")
    args = ap.parse_args(argv)

    events = json.loads(args.events.read_text(encoding="utf-8"))

    from riichienv import GameRule

    from advisor.recommend import Adviser
    from engine.replay import replay_decisions

    a32 = Adviser(str(args.ckpt), precision="fp32")
    a16 = Adviser(str(args.ckpt), precision="fp16")
    if a32._dev.type != "cuda":  # noqa: SLF001 — 审计脚本，报告口径本身
        print(f"[提示] 当前设备 {a32._dev} 无 fp16 路径（半精度仅 CUDA 生效），"
              f"两档必然逐位一致；本环境的翻盘率只说明脚本能跑通。")

    n = flips = 0
    drift: list[float] = []   # top3 内同 desc 的概率绝对漂移
    rows: list[str] = []
    worst: tuple[float, str] | None = None  # (fp32领先优势, 描述行)
    for d in replay_decisions(events, rule=GameRule.default_mjsoul()):
        if args.seat >= 0 and d.pid != args.seat:
            continue
        legal = list(d.legal)
        if len(legal) < 2 or d.ob is None:
            continue
        t32 = a32.topk(d.ob, legal, top=3)
        t16 = a16.topk(d.ob, legal, top=3)
        n += 1
        p16 = dict(t16)
        for desc, pr in t32:
            if desc in p16:
                drift.append(abs(pr - p16[desc]))
        if t32[0][0] != t16[0][0]:
            flips += 1
            margin = t32[0][1] - (t32[1][1] if len(t32) > 1 else 0.0)
            line = (f"  idx{d.idx} seat{d.pid}: fp32 {t32[0][0]} {t32[0][1]:.1%} "
                    f"→ fp16 {t16[0][0]} {t16[0][1]:.1%}（fp32 top1-top2 差 {margin:.1%}）")
            if worst is None or margin > worst[0]:
                worst = (margin, line)
            if len(rows) < args.max_list:
                rows.append(line)

    if n == 0:
        print("无可比决策窗（legal≥2）——检查 --seat/--events。")
        return 1
    dmax = max(drift) if drift else 0.0
    dmean = sum(drift) / len(drift) if drift else 0.0
    print(f"决策窗 {n} | top1 翻盘 {flips}（{flips / n:.2%}）")
    print(f"top3 概率漂移 |Δ|：均值 {dmean:.2%} 最大 {dmax:.2%}")
    if worst is not None:
        print("优势最大仍翻盘的一例（最难解释的情形）：")
        print(worst[1])
    if rows:
        print("翻盘明细：")
        print("\n".join(rows))
    print("\n判读：翻盘率≈0 且漂移百分之几以内 ⇒ 两档只差显示尾部，可把线上默认切到 "
          "fp16 对齐发布口径；若翻盘显著集中于大优势窗口 ⇒ 保持分档，仅文档注明口径。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
