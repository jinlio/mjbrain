"""M2 核心训练：LAYA RLCD 蒸馏微调（单卡 4060 移植官方 notebook 环）。

与 reference/notebook(train_ddp.py) 的逐点对应：collate/高斯策略梯度/软 CE/
LBFGS 温度校准全部照搬，差异仅三处——
1. 单卡无 DDP（4060 8GB：fp16 autocast + encoder/head 梯度检查点，
   MICRO_BATCH 8 / GRAD_ACCUM 4，LR 2.5e-5/1e-4 与官方同参）；
2. target 是**人类 one-hot 与 B2 教师 softmax 的混合**（pairs 语料自带
   raw q，本端自选温度：teacher_probs = softmax(q/T)；重复实体牌条目均分
   人类质量）；
3. 校准按 **temp_bucket 分桶拟合**（agent 推理优先查 temperature_by_options，
   notebook 只更 temperature 列表会留陈旧桶参数被命中——移植时修正）。

items 在训练端由 pairs(结构化事实) × 原始牌谱重演 × brain/serialize 现算，
序列化器改版不需重做语料。

用法（conda activate mjbrain）：
  python -m train.rlcd_sft --pairs-dir data/processed/pairs_pilot --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch

_STATE_BUDGET = 512
_HEAD_BUDGET = 192


# ------------------------------------------------------------------ targets
def _human_vec(label_idx: int, entries: list[dict]) -> list[float]:
    """人类标签 -> legal 顺序概率。重复 (type,pai) 条目（同牌多实体）均分。"""
    lab = entries[label_idx]
    key = (lab["type"], lab.get("pai"))
    idxs = [i for i, a in enumerate(entries) if (a["type"], a.get("pai")) == key]
    if not idxs:
        idxs = [label_idx]
    v = [0.0] * len(entries)
    for i in idxs:
        v[i] = 1.0 / len(idxs)
    return v


def _teacher_vec(aligned_q: list, temp: float) -> list[float]:
    if not aligned_q:
        return []
    qs = [None if v is None else float(v) / temp for v in aligned_q]
    finite = [v for v in qs if v is not None]
    if not finite:
        return []
    m = max(finite)
    ex = [0.0 if v is None else math.exp(v - m) for v in qs]
    s = sum(ex)
    return [e / s for e in ex]


def mix_target(entries, aligned_q, label_idx: int, alpha: float, temp: float,
               stats: Counter) -> list[float] | None:
    h = _human_vec(label_idx, entries)
    t = _teacher_vec(aligned_q, temp)
    if t and len(t) == len(h):
        out = [alpha * tv + (1 - alpha) * hv for tv, hv in zip(t, h, strict=True)]
    else:
        stats["target_teacher_unavailable"] += 1
        out = h
    s = sum(out)
    if s <= 0:
        return None
    return [v / s for v in out]


# ------------------------------------------------------------------ items
def _is_val(game_id: str) -> bool:
    """held-out 按 game 划分（同局决策绝不跨集），~5%，跨运行稳定。"""
    import hashlib

    return int(hashlib.md5(game_id.encode()).hexdigest()[:8], 16) % 20 == 0


def build_items(pairs_dir: pathlib.Path, raw_dir: pathlib.Path, tok, *,
                alpha: float, temp: float, max_items: int, stats: Counter,
                disagree_alpha: float | None = None, w_meld: float = 1.0):
    """base（重演+tokenize，昂贵）走分池缓存；target 混合（便宜）每跑现算。"""
    base_tr, base_val = _build_base(pairs_dir, raw_dir, tok, max_items, stats)
    kw = {"disagree_alpha": disagree_alpha, "w_meld": w_meld}
    items = _materialize(base_tr, alpha, temp, stats, **kw)
    val_items = _materialize(base_val, alpha, temp, stats, **kw)
    stats["items"] = len(items)
    stats["val_items"] = len(val_items)
    return items, val_items


# M3 处方（PLAN M2-4 失败切片）：鸣牌类样本只占 ~3.5% 且 chi 层塌陷
# （top1 0.021）；教师分歧层人类风格与教师系统性冲突。v2 配方两板斧：
# --w-meld 给鸣牌类决策的 loss 上行加权；--disagree-alpha 让分歧层
# （agree=False，pairs 表已预计算）单独退到纯教师软标签。
MELD_TYPES = frozenset({"chi", "pon", "daiminkan", "kakan", "ankan"})


def _materialize(base: list[dict], alpha: float, temp: float, stats: Counter,
                 *, disagree_alpha: float | None = None, w_meld: float = 1.0):
    out = []
    for b in base:
        a = alpha
        if disagree_alpha is not None and b.get("agree") is False:
            a = disagree_alpha
            stats["layer_disagree"] += 1
        tq = mix_target(b["entries"], b["aligned_q"], b["label_idx"], a, temp, stats)
        if tq is None:
            stats["target_dead"] += 1
            continue
        w = w_meld if (w_meld != 1.0 and b.get("gt") in MELD_TYPES) else 1.0
        if w != 1.0:
            stats["w_meld_applied"] += 1
        out.append(
            {
                "ids": b["ids"],
                "markers": b["markers"],
                "qtype": b["qtype"],
                "w": w,
                "target": tq,
                # label = 人类选择下标（loss 不用它，held-out top-1 与
                # 诊断用；混合 target 的 argmax 可偏离人类）
                "label": b["label_idx"],
                "gt": b["gt"],
                # 诊断连接键（M2-4 失败切片用）：哪局哪手、教师是否同意人类
                "gid": b.get("gid", ""),
                "idx": b.get("idx", -1),
                "agree": b.get("agree"),
            }
        )
    return out


# v4：修两池截断语义（v2 的 val 被 batch 级 break 连坐清空；v3 的单局会垄断 val 配额）
_SER_TAG = "v4"


def _build_base(pairs_dir: pathlib.Path, raw_dir: pathlib.Path, tok,
                max_items: int, stats: Counter):
    import hashlib

    import pyarrow.parquet as pq
    from laya.common import QTYPES, build_sequence

    from brain.serialize import laya_question, state_text
    from engine.replay import replay_decisions

    # 兼容两种落盘：单文件 pairs.parquet 与增量分片 pairs-partNN.parquet
    files = sorted(pairs_dir.glob("pairs*.parquet"))
    assert files, f"{pairs_dir} 下没有 pairs*.parquet"
    fp = pathlib.Path("data/processed/items_cache")
    fp.mkdir(parents=True, exist_ok=True)
    sig = hashlib.sha1(
        "|".join(f"{f.name}:{f.stat().st_size}" for f in files).encode()
        + f"|{max_items}|ser={_SER_TAG}".encode()
    ).hexdigest()[:16]
    cache = fp / f"base-{pairs_dir.name}-{sig}.pt"
    if cache.exists():
        data = torch.load(cache, weights_only=True)
        stats["items_cache_hit"] += 1
        return data["train"], data["val"]

    rows = [r for f in files for r in pq.read_table(f).to_pylist()]
    by_game: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_game[r["game_id"]].append(r)
    wanted = set(by_game)

    base_tr: list[dict] = []
    base_val: list[dict] = []
    scan_done = False  # 两池皆满才许提前停扫（单池满继续访问，否则同批未及的 val 局会被连坐跳过）
    for f in sorted(raw_dir.glob("data/*.parquet")):
        if not wanted or scan_done:
            break
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
            if scan_done:
                break
            for rec in batch.to_pylist():
                if rec["game_id"] not in wanted:
                    continue
                gid = rec["game_id"]
                wanted.discard(gid)
                try:
                    events = [json.loads(ln) for ln in rec["events"].splitlines()]
                except Exception:  # noqa: BLE001 — 坏行弃
                    stats["game_unparseable"] += 1
                    continue
                decs = {(d.pid, d.idx): d for d in replay_decisions(events) if d.ob is not None}
                bucket = base_val if _is_val(gid) else base_tr
                vg_n = 0
                for row in by_game[gid]:
                    if len(base_tr) >= max_items and (
                        bucket is base_tr or len(base_val) >= max_items // 10
                    ):
                        break
                    if bucket is base_val and vg_n >= max(50, max_items // 80):
                        break  # 单局 val 配额：held-out 至少 ~8 局分摊，防一整局垄断
                    vg_n += 1
                    d = decs.get((row["seat"], row["idx"]))
                    if d is None:
                        stats["item_replay_miss"] += 1
                        continue
                    q = laya_question(d.legal)
                    seq, markers = build_sequence(
                        tok, state_text(d.ob), q, _STATE_BUDGET, _HEAD_BUDGET
                    )
                    if len(markers) != len(q["crit"]):
                        stats["item_truncated"] += 1
                        continue
                    bucket.append(
                        {
                            "ids": seq,
                            "markers": markers,
                            "qtype": QTYPES["choice"],
                            "entries": json.loads(row["legal_json"]),
                            "aligned_q": json.loads(row["q_json"]),
                            "label_idx": row["label_idx"],
                            "gt": row["label_type"],
                            "gid": gid,
                            "idx": row["idx"],
                            "agree": bool(row["agree"]),
                        }
                    )
                if len(base_tr) >= max_items and len(base_val) >= max_items // 10:
                    scan_done = True
                    break
    torch.save({"train": base_tr, "val": base_val, "ser": _SER_TAG}, cache)
    print(f"base items cached -> {cache.name}", flush=True)
    return base_tr, base_val


# ------------------------------------------------------------------ batch
def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    # kmax>=2：laya forward 的 act_head 特征要 p.topk(2)，整批单选项会炸；
    # 假位列被 marker_mask 掩成 -1e4（softmax≈0），不污染任何计算。
    kmax = max(2, max(len(it["markers"]) for it in items))
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "weight": torch.tensor([it.get("w", 1.0) for it in items], dtype=torch.float32),
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def evaluate(model, items, tok, device, *, batch: int = 16, dump_path=None) -> dict:
    """held-out 评估（模型 argmax vs 人类 label）。

    指标：top-1 总体+分决策类型；`conf1_{g}` = top-prob>0.7 的高置信错判率
    （M2-4 关注的"自信地错"）。`dump_path` 给定时把逐条预测写 JSONL
    （gid/idx/gt/human/pick/p_top/agree），供离线切片：防守/鸣牌错误分布、
    教师-人类分歧段的模型站队等。
    """
    model.eval()
    hits = Counter()
    tot = Counter()
    cw = Counter()
    rows: list[dict] = []
    with torch.no_grad():
        for i0 in range(0, len(items), batch):
            chunk = items[i0 : i0 + batch]
            cb = collate_train_batch(chunk, tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.float16):
                l, _ = model(
                    cb["input_ids"].to(device), cb["attention_mask"].to(device),
                    cb["marker_pos"].to(device), cb["marker_mask"].to(device),
                    cb["qtype"].to(device),
                )
            ln = l.float().cpu().numpy()
            for ri, it in enumerate(chunk):
                kk = len(it["markers"])
                z = ln[ri, :kk]
                p = np.exp(z - z.max())
                p_top = float(p[int(z.argmax())] / p.sum())
                pick = int(z.argmax())
                ok = pick == it["label"]
                tot[it["gt"]] += 1
                tot["ALL"] += 1
                if ok:
                    hits[it["gt"]] += 1
                    hits["ALL"] += 1
                elif p_top > 0.7:
                    cw[it["gt"]] += 1
                    cw["ALL"] += 1
                if dump_path is not None:
                    rows.append(
                        {
                            "gid": it.get("gid", ""), "idx": it.get("idx", -1),
                            "gt": it["gt"], "human": it["label"], "pick": pick,
                            "ok": ok, "p_top": round(p_top, 4),
                            "n_opts": kk, "agree": it.get("agree"),
                        }
                    )
    model.train()
    out = {f"top1_{g or 'ALL'}": round(hits[g] / max(1, tot[g]), 4) for g in tot}
    out.update(
        {f"conf1_{g or 'ALL'}": round(cw[g] / max(1, tot[g]), 4) for g in cw}
    )
    if dump_path is not None:
        dp = pathlib.Path(dump_path)
        dp.parent.mkdir(parents=True, exist_ok=True)
        with open(dp, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r_, ensure_ascii=False) + "\n" for r_ in rows)
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs-dir", default="data/processed/pairs_260")
    ap.add_argument("--raw-dir", default="data/raw/tenhou_houou_mjai")
    ap.add_argument("--model-dir", default="", help="laya checkpoint 目录（默认走 HF 缓存）")
    ap.add_argument("--alpha", type=float, default=0.5, help="教师软标签权重")
    ap.add_argument("--temp", type=float, default=0.3, help="教师 softmax 温度（meta_show 显示同值）")
    ap.add_argument("--disagree-alpha", type=float, default=None, dest="disagree_alpha",
                    help="M3 v2：教师分歧层(agree=False)单独用的教师权重，覆盖 --alpha；"
                         "典型 1.0=退纯教师软标签。不传则分歧层与全体同 alpha")
    ap.add_argument("--w-meld", type=float, default=1.0, dest="w_meld",
                    help="M3 v2：鸣牌类(chi/pon/kan)样本 loss 加权系数，>1 上采样其梯度；1=不加权")
    ap.add_argument("--epochs", type=int, default=0, help="0=冒烟2/正式4")
    ap.add_argument("--max-items", type=int, default=120000)
    ap.add_argument("--smoke", action="store_true", help="小样本端到端冒烟")
    ap.add_argument("--tag", default="rlcd-sft")
    ap.add_argument("--resume", default="",
                    help="checkpoints/<run_id>/resume.pt 断点续训（epoch 边界粒度；"
                         "items 缓存按 pairs+max_items 键控，重放同序）")
    ap.add_argument("--micro-batch", type=int, default=8, dest="micro_batch",
                    help="每步样本数；8GB 卡白天与桌面共卡时降到 4（配 --grad-accum 8 "
                         "保持有效 batch 32 与 lr 语义不变）")
    ap.add_argument("--grad-accum", type=int, default=4, dest="grad_accum")
    ap.add_argument("--save-every", type=int, default=0, dest="save_every",
                    help="每 N 个 optimizer update 落一次 resume.pt（0=仅 epoch 边界；"
                         "无人值守长跑建议 1000，OOM 重启最多丢该窗口进度）")
    ap.add_argument("--opt", choices=["adamw", "adamw8bit"], default="adamw",
                    help="adamw8bit=bitsandbytes 8-bit 优化状态：8GB 卡上底存约 -2.4GB，"
                         "退出 WDDM 分页区（bench 2026-09-21 夜实测选型）；"
                         "resume 时必须与落盘时的选择一致")
    args = ap.parse_args()

    from laya.common import build_model, proper_reward
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    from brain.runlog import RunLog

    device = torch.device("cuda")
    model_dir = pathlib.Path(args.model_dir) if args.model_dir else None
    if model_dir is None:
        # 离线优先：缓存完好即不碰 hub（断网/代理抖动下训练仍可复现）。
        # repo_id 形式的 from_pretrained 会先 HEAD 根 config.json，离线必挂，
        # 故一律解析成本地目录再加载。
        from huggingface_hub import snapshot_download

        try:
            model_dir = pathlib.Path(
                snapshot_download("convaiinnovations/laya", local_files_only=True)
            )
        except Exception:  # noqa: BLE001 — 本机无缓存才联网
            model_dir = pathlib.Path(snapshot_download("convaiinnovations/laya"))
    with open(model_dir / "rl_agent_config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    tok = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    pairs_dir = pathlib.Path(args.pairs_dir)
    max_items = 4000 if args.smoke else args.max_items
    epochs = args.epochs or (2 if args.smoke else 4)

    stats: Counter = Counter()
    t0 = time.time()
    items, val = build_items(pairs_dir, pathlib.Path(args.raw_dir), tok,
                             alpha=args.alpha, temp=args.temp, max_items=max_items,
                             stats=stats, disagree_alpha=args.disagree_alpha,
                             w_meld=args.w_meld)
    if not items:
        print("无可用 items", dict(stats))
        return 1

    run = RunLog(
        args.tag,
        config={
            "pairs_dir": str(pairs_dir),
            "raw_dir": args.raw_dir,
            "model_dir": str(model_dir),
            "base": "convaiinnovations/laya",
            "alpha": args.alpha,
            "disagree_alpha": args.disagree_alpha,
            "w_meld": args.w_meld,
            "teacher_temp": args.temp,
            "epochs": epochs,
            "micro_batch": args.micro_batch,
            "grad_accum": args.grad_accum,
            "group_size": 4,
            "lr_encoder": 2.5e-5,
            "lr_head": 1.0e-4,
            "sigma": [0.4, 0.1],
            "max_len": cfg["max_len"],
            "head_max_len": cfg["head_max_len"],
            "smoke": args.smoke,
            "resume": args.resume or None,
            "optimizer": args.opt,
        },
    )
    print(
        f"items={len(items)} ({time.time()-t0:.0f}s 构建) stats={dict(stats)}",
        flush=True,
    )

    cfg = dict(cfg)
    cfg["gradient_checkpointing"] = True
    model = build_model(cfg, encoder_dir=str(model_dir / "encoder"))
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.head_checkpointing = True
    model.to(device).train()

    MICRO_BATCH, GRAD_ACCUM, GROUP_SIZE = args.micro_batch, args.grad_accum, 4
    LR_ENC, LR_HEAD = 2.5e-5, 1.0e-4
    SIGMA_START, SIGMA_END = 0.4, 0.1
    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    # bench 2026-09-21 夜：8GB 卡上 fp32 AdamW 态(~4.8GB)+fp16 权重+梯度顶满
    # 显存进 WDDM 分页（44s/upd）；8-bit 态把底存拉到 ~4GB，实测 1.48s/upd。
    if args.opt == "adamw8bit":
        from bitsandbytes.optim import AdamW8bit

        opt_cls = AdamW8bit
    else:
        opt_cls = torch.optim.AdamW
    optimizer = opt_cls(
        [{"params": enc_params, "lr": LR_ENC}, {"params": head_params, "lr": LR_HEAD}],
        weight_decay=0.01,
    )
    updates = max(1, (len(items) // (MICRO_BATCH * GRAD_ACCUM)) * epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    print(f"total optimizer updates = {updates}", flush=True)

    # 断点续训：resume.pt 原子落盘——epoch 边界必有（batch_in_epoch=0），
    # --save-every 时另在 update 边界周期性落。items 走键控缓存、shuffle 用
    # 42+epoch 定种，因此同命令重放保证逐 epoch 数据序完全一致；epoch 内中断
    # 的续点是"已消费 items 条数"，重放时快进跳过（数据序不依赖 RNG 于 shuffle 后）。
    ck_dir = pathlib.Path(args.resume).parent if args.resume else pathlib.Path("checkpoints") / run.run_id
    ck_dir.mkdir(parents=True, exist_ok=True)

    def _save_resume(epoch_done: int, batch_in_epoch: int):
        tmp = ck_dir / "resume.pt.tmp"
        torch.save(
            {
                "model": {k: v.half().cpu() for k, v in model.state_dict().items()},
                "optim": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "steps": step,
                "epoch": epoch_done,
                "batch_in_epoch": batch_in_epoch,
                "micro_batch": MICRO_BATCH,
                "grad_accum": GRAD_ACCUM,
                "opt": args.opt,
            },
            tmp,
        )
        tmp.replace(ck_dir / "resume.pt")  # 原子替换：崩溃不会留半个档

    step = 0
    start_epoch = 0
    start_batch = 0
    if args.resume:
        st = torch.load(args.resume, weights_only=True)
        saved_opt = st.get("opt", "adamw")
        if saved_opt != args.opt:
            raise SystemExit(
                f"--resume 优化器不匹配：存档 {saved_opt}，当前 --opt {args.opt}"
                "（8bit/fp32 状态不可互载，须与首程同命令）"
            )
        start_batch = int(st.get("batch_in_epoch", 0))
        if start_batch and (int(st.get("micro_batch", MICRO_BATCH)) != MICRO_BATCH
                            or int(st.get("grad_accum", GRAD_ACCUM)) != GRAD_ACCUM):
            raise SystemExit(
                f"epoch 内续点须同 batch 配置（存档 micro={st.get('micro_batch')} "
                f"accum={st.get('grad_accum')}，当前 {MICRO_BATCH}/{GRAD_ACCUM}）")
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optim"])
        scaler.load_state_dict(st["scaler"])
        step = int(st["steps"])
        for _ in range(step):
            sched.step()
        start_epoch = int(st["epoch"])
        print(f"resumed: epoch {start_epoch}/{epochs}, step {step}, "
              f"batch_in_epoch {start_batch}", flush=True)
    status = "ok"
    last_val: dict = {}
    try:
        t_train = time.time()
        for epoch in range(start_epoch, epochs):
            random.seed(42 + epoch)
            random.shuffle(items)
            epoch_loss, n_batches = 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, epochs - 1))
            b_start = start_batch if epoch == start_epoch else 0
            for b_idx in range(b_start, len(items), MICRO_BATCH):
                chunk = items[b_idx : b_idx + MICRO_BATCH]
                if not chunk:
                    continue
                batch = collate_train_batch(chunk, tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits, act = model(
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                        batch["marker_pos"].to(device),
                        batch["marker_mask"].to(device),
                        batch["qtype"].to(device),
                    )
                logits = logits.float()
                mask = batch["marker_mask"].to(device)
                kcnt = mask.sum(-1, keepdim=True).float()
                target = batch["target"].to(device)

                eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
                eps = (eps - eps.sum(-1, keepdim=True) / kcnt) * mask
                z = logits.detach().unsqueeze(0) + eps
                q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
                with torch.no_grad():
                    r = proper_reward(
                        q, target.unsqueeze(0), batch["qtype"].to(device), mask,
                        w_sph=0.75, w_rps=1.0,
                    )
                    adv = r - r.mean(0, keepdim=True)
                    adv = adv / (adv.std() + 1e-6)
                logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
                # 加权均值（M3 --w-meld）：w 全 1 时与 .mean() 逐位相等。
                w = batch["weight"].to(device)
                loss_rl = -((adv * logp) * w).sum() / (w.sum() * logp.shape[0])
                ce_per = -(
                    target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
                ).sum(-1)
                loss_ce = (ce_per * w).sum() / w.sum()
                loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()
                scaler.scale(loss).backward()
                accum += 1
                if accum % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(items):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    sched.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    run.metric(
                        step,
                        loss=round(loss.item() * GRAD_ACCUM, 4),
                        rl=round(loss_rl.item(), 4),
                        ce=round(loss_ce.item(), 4),
                        reward=round(r.mean().item(), 4),
                        sigma=round(sigma, 3),
                        lr=sched.get_last_lr()[1],
                    )
                # 周期性续点：只可能在 update 边界触发（accum 已清零、无悬挂梯度）
                if args.save_every and step and step % args.save_every == 0:
                    _save_resume(epoch, b_idx + MICRO_BATCH)
                epoch_loss += loss.item() * GRAD_ACCUM
                n_batches += 1
                if n_batches % 50 == 0:
                    print(
                        f"ep{epoch+1}/{epochs} b{n_batches} loss={loss.item()*GRAD_ACCUM:.4f} "
                        f"r={r.mean().item():.3f} {time.time()-t_train:.0f}s",
                        flush=True,
                    )
            print(f"=== epoch {epoch+1} avg loss {epoch_loss/max(1,n_batches):.4f} ===", flush=True)
            if val:
                vm = evaluate(
                    model, val, tok, device,
                    dump_path=pathlib.Path("runs") / run.run_id / f"preds-val-ep{epoch+1}.jsonl",
                )
                run.metric(step, val_epoch=epoch + 1, **vm)
                print(f"    held-out: {vm}", flush=True)
                last_val = vm
            if epoch + 1 < epochs:  # 最后一个 epoch 由终盘保存接管
                _save_resume(epoch + 1, 0)

        # ---- 分桶温度校准（修正 notebook 只更 qtype 列表、旧 by_options 桶仍被
        # agent 命中的问题：这里按 temp_bucket 键逐桶拟合后整体替换）
        from laya.common import temp_bucket

        model.eval()
        calib = items[::15][:400]
        preds = []
        with torch.no_grad():
            for c0 in range(0, len(calib), 16):
                cb = collate_train_batch(calib[c0 : c0 + 16], tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    l, _ = model(
                        cb["input_ids"].to(device), cb["attention_mask"].to(device),
                        cb["marker_pos"].to(device), cb["marker_mask"].to(device),
                        cb["qtype"].to(device),
                    )
                ln = l.float().cpu().numpy()
                for ri, it in enumerate(calib[c0 : c0 + 16]):
                    kk = len(it["markers"])
                    preds.append((it["qtype"], ln[ri, :kk], it["target"]))
        by_bucket: dict[str, list] = defaultdict(list)
        for qt, z, t in preds:
            by_bucket[temp_bucket(qt, len(z))].append((z.tolist(), t))
        temps_by_bucket = {
            bkt: fit_one_temp(sel) for bkt, sel in sorted(by_bucket.items())
        }
        temps_by_qtype = [
            fit_one_temp([(z, t) for qt2, z, t in preds if qt2 == qt]) for qt in range(3)
        ]
        print("fitted temps:", temps_by_bucket, temps_by_qtype, flush=True)

        (ck_dir / "resume.pt").unlink(missing_ok=True)
        sd = {k2: v.half().contiguous().cpu() for k2, v in model.state_dict().items()}
        save_file(sd, str(ck_dir / "model.safetensors"))
        model.encoder.config.save_pretrained(str(ck_dir / "encoder"))
        tok.save_pretrained(str(ck_dir / "tokenizer"))
        out_cfg = dict(cfg)
        out_cfg.pop("gradient_checkpointing", None)
        out_cfg["fine_tuned"] = True
        out_cfg["model_name"] = f"mjbrain-{run.run_id}"
        out_cfg["temperature"] = temps_by_qtype
        out_cfg["temperature_by_options"] = {
            **{k: temps_by_bucket[k] for k in temps_by_bucket},
        }
        with open(ck_dir / "rl_agent_config.json", "w", encoding="utf-8") as f:
            json.dump(out_cfg, f, indent=2)
        summary = {
            "status": status,
            "items": len(items),
            "val_items": len(val),
            "heldout": last_val,
            "steps": step,
            "epochs": epochs,
            "train_seconds": round(time.time() - t_train, 1),
            "temps": temps_by_bucket,
            "ckpt": str(ck_dir),
            "sha256": __import__("hashlib").sha256(
                (ck_dir / "model.safetensors").read_bytes()
            ).hexdigest()[:16],
            "stats": dict(stats),
        }
    except Exception as ex:
        run.finish({"status": "failed", "error": repr(ex)[:400], "items": len(items), "steps": step})
        raise
    run.finish(summary)
    print("saved ->", ck_dir, "run ->", run.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
