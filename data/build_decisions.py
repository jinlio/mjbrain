"""构建微调语料（M1/M2）。

凤王 parquet(服务器视角 mjai) --engine.replay--> 决策点样本行，逐 shard 落
data/processed/decisions/*.parquet + profile.json 分层统计。

用法: conda activate mjbrain && python -m data.build_decisions \
        --max-games 20           # 试点；省略则整片
        --limit-per-shard 5000   # 每片最多取几局（控规模）
"""

from __future__ import annotations

import argparse
import bisect
import json
import pathlib
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from engine.replay import replay_decisions

_TABLE = pa.schema(
    [
        ("game_id", pa.string()),
        ("shard", pa.string()),
        ("idx", pa.int32()),
        ("seat", pa.int8()),
        ("bakaze", pa.string()),
        ("kyoku", pa.int8()),
        ("honba", pa.int8()),
        ("oya", pa.int8()),
        ("label", pa.string()),
        ("label_type", pa.string()),
        ("legal_json", pa.string()),
        ("implicit", pa.bool_()),
    ]
)


def _kyoku_marks(events: list[dict]):
    """[(idx, meta)] 升序；决策 idx 落在其后最近的 start_kyoku 语境里。"""
    marks = []
    for i, e in enumerate(events):
        if e.get("type") == "start_kyoku":
            marks.append((i, e))
    return marks


def process_game(events: list[dict], game_id: str, shard: str, stats: Counter):
    rows: list[dict] = []
    marks = _kyoku_marks(events)
    mark_idx = [m[0] for m in marks]
    for d in replay_decisions(events, stats=stats):
        pos = bisect.bisect_right(mark_idx, d.idx) - 1
        meta = marks[pos][1] if pos >= 0 else {}
        lab = json.loads(d.label)
        rows.append(
            {
                "game_id": game_id,
                "shard": shard,
                "idx": d.idx,
                "seat": d.pid,
                "bakaze": meta.get("bakaze", ""),
                "kyoku": meta.get("kyoku", -1),
                "honba": meta.get("honba", 0),
                "oya": meta.get("oya", -1),
                "label": d.label,
                "label_type": lab["type"],
                "legal_json": json.dumps([json.loads(s) for s in d.legal], separators=(",", ":")),
                "implicit": d.implicit,
            }
        )
        stats["decision"] += 1
        stats[f"labeltype::{lab['type']}"] += 1
        stats["implicit" if d.implicit else "explicit"] += 1
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/raw/tenhou_houou_mjai")
    ap.add_argument("--out-dir", default="data/processed/decisions")
    ap.add_argument("--max-games", type=int, default=0, help="全局上限（0=不限）")
    ap.add_argument("--limit-per-shard", type=int, default=0)
    ap.add_argument("--shard", default="", help="只处理指定 shard 文件名")
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(data_dir.glob("data/*.parquet"))
    if args.shard:
        shards = [s for s in shards if s.name == args.shard]
    assert shards, f"{data_dir}/data/*.parquet 没找到，先下载数据集"

    stats: Counter = Counter()
    n_done = 0
    for sh in shards:
        if args.max_games and n_done >= args.max_games:
            break
        pf = pq.ParquetFile(sh)
        kept_rows: list[dict] = []
        games_this_shard = 0
        for batch in pf.iter_batches(batch_size=256):
            if args.max_games and n_done >= args.max_games:
                break
            for row in batch.to_pylist():
                if args.max_games and n_done >= args.max_games:
                    break
                if args.limit_per_shard and games_this_shard >= args.limit_per_shard:
                    break
                try:
                    events = [json.loads(l) for l in row["events"].splitlines()]
                except Exception:  # noqa: BLE001 — 坏行弃本场，计数不崩
                    stats["games_unparseable"] += 1
                    continue
                before_total = sum(
                    stats.get(k, 0)
                    for k in ("desync_no_window", "desync_illegal_label", "apply_fail")
                )
                rows = process_game(events, row["game_id"], sh.name, stats)
                after_total = sum(
                    stats.get(k, 0)
                    for k in ("desync_no_window", "desync_illegal_label", "apply_fail")
                )
                games_this_shard += 1
                n_done += 1
                if after_total > before_total:
                    stats["games_dropped_desync"] += 1
                    continue
                stats["games_ok"] += 1
                kept_rows.extend(rows)
            if args.max_games and n_done >= args.max_games:
                break
        if kept_rows:
            cols = {f.name: [r[f.name] for r in kept_rows] for f in _TABLE}
            t = pa.table(cols, schema=_TABLE)
            out = out_dir / f"decisions-{sh.name.replace('.parquet', '')}.parquet"
            pq.write_table(t, out, compression="zstd")
            print(f"{sh.name}: {games_this_shard} 局 -> {len(kept_rows)} 决策 -> {out.name}")
    out = out_dir / "profile.json"
    prof = dict(sorted(stats.items()))
    out.write_text(json.dumps(prof, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n== 画像 ({n_done} 局) ==")
    for k, v in prof.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
