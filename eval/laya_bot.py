"""B3 — 微调 LAYA 竞技场入口（M2 门槛候选）。

不自己重演牌局（step 制 sim 的牌山按 seed 播种，外部重演器无法复现），
而是走协议扩展：声明 `wants_ob=True`，engine/sim 直接把**决策时刻的真实
Observation** 递进来 —— 与训练语料完全同源的 obs（同为 get_observation
产物），state_text 零漂移。遮蔽由引擎负责，本模块零手工特征。

流程：state_text -> laya_question -> build_sequence -> 单次前向 ->
按校准温度（temp_bucket 优先，回退 qtype 表）取 argmax。
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter

_CACHE: dict[str, dict] = {}


def _load(ckpt: str) -> dict:
    """按目录缓存 (tok, model, cfg 温度)。多 bot 实例共享同一权重。"""
    import torch
    from laya.common import build_model
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    key = str(pathlib.Path(ckpt).resolve())
    if key in _CACHE:
        return _CACHE[key]
    p = pathlib.Path(key)
    with open(p / "rl_agent_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(p / "tokenizer")
    model = build_model(cfg, encoder_dir=str(p / "encoder"))
    model.load_state_dict(load_file(str(p / "model.safetensors")), strict=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # 外协 Mac（M 系）走 MPS；fp32，autocast 仅 cuda 开
    else:
        device = torch.device("cpu")
    model.to(device).eval()
    _CACHE[key] = {
        "tok": tok,
        "model": model,
        "device": device,
        "max_len": int(cfg.get("max_len", 512)),
        "head_max_len": int(cfg.get("head_max_len", 192)),
        "temps": cfg.get("temperature", [1.0, 1.0, 1.0]),
        "temps_by_opts": cfg.get("temperature_by_options", {}),
    }
    return _CACHE[key]


class LayaBot:
    name = "B3-laya"
    wants_ob = True  # 协议扩展：让 sim 把决策时刻 Observation 传进 react

    def __init__(self, seed: int = 0, ckpt: str | None = None) -> None:
        import os

        ck = ckpt or os.environ.get("MJ_LAYA_CKPT", "")
        if not ck or not pathlib.Path(ck).exists():
            raise RuntimeError("B3 需要 ckpt 目录或环境变量 MJ_LAYA_CKPT")
        self._L = _load(ck)
        self._seed = seed
        self.stats = Counter()

    def react(self, events, seat: int, legal_actions: list[str], *, ob=None) -> str:
        import numpy as np
        import torch
        from laya.common import build_sequence, temp_bucket

        from brain.serialize import laya_question, state_text

        def fallback(why: str) -> str:
            self.stats[why] += 1
            parsed = [json.loads(a) for a in legal_actions]
            pick = next((a for a in parsed if a["type"] == "none"), parsed[0])
            return json.dumps(pick, separators=(",", ":"))

        if ob is None:  # sim 未提供 obs（非 arena 路径调用）
            return fallback("no_ob")

        if len(legal_actions) == 1:
            # 强制决策（唯一合法动作）：零熵，且绕开 laya forward 里
            # act_head 特征 p.topk(2) 在 K=1 时的崩溃（arena 单样本批必炸）。
            self.stats["forced_single"] += 1
            return legal_actions[0]

        state = state_text(ob)
        q = laya_question(legal_actions)
        ids, markers = build_sequence(
            self._L["tok"], state, q, self._L["max_len"], self._L["head_max_len"]
        )
        if len(markers) != len(q["crit"]):
            return fallback("head_truncated")

        L, K = len(ids), len(markers)
        dev = self._L["device"]
        batch = (
            torch.tensor([ids], dtype=torch.long),
            torch.ones(1, L, dtype=torch.long),
            torch.tensor([markers], dtype=torch.long),
            torch.ones(1, K, dtype=torch.bool),
            torch.tensor([0], dtype=torch.long),
        )
        with (
            torch.no_grad(),
            torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"),
        ):
            logits, _ = self._L["model"](*(x.to(dev) for x in batch))
        z = logits.float().cpu().numpy()[0, :K]
        tkey = temp_bucket(0, K)
        t = self._L["temps_by_opts"].get(tkey, self._L["temps"][0])
        z = z / max(1e-3, float(t))
        self.stats["act"] += 1
        return legal_actions[int(np.argmax(z))]
