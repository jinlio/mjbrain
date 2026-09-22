"""推荐内核：给定 (ob, legal) 出 B3 top-k；手输/棋谱/服务三侧共用。

只读决策：本模块产出的只是建议文本，永不回写任何游戏通道
（零动作注入红线，见 docs/PLAN.md M4 定线）。
"""

from __future__ import annotations

import json


class Adviser:
    def __init__(self, ckpt: str, device: str | None = None) -> None:
        import torch

        from eval.laya_bot import _load

        self._L = _load(ckpt)
        if device:
            self._L["model"].to(torch.device(device))
            self._L["device"] = torch.device(device)

    def topk(self, ob, legal: list[str], top: int = 5) -> list[tuple[str, float]]:
        """返回 [(动作描述, 概率)]，按温度校准 softmax；概率和=1。"""
        import numpy as np
        import torch
        from laya.common import build_sequence, temp_bucket

        from brain.serialize import laya_question, state_text

        if len(legal) == 1:
            a = json.loads(legal[0])
            return [(self._desc(a), 1.0)]
        q = laya_question(legal)
        L = self._L
        ids, markers = build_sequence(
            L["tok"], state_text(ob), q, L["max_len"], L["head_max_len"])
        if len(markers) != len(q["crit"]):
            raise ValueError("选项被序列截断（加大 head_max_len）")
        dev = L["device"]
        batch = (
            torch.tensor([ids], dtype=torch.long),
            torch.ones(1, len(ids), dtype=torch.long),
            torch.tensor([markers], dtype=torch.long),
            torch.ones(1, len(markers), dtype=torch.bool),
            torch.tensor([0], dtype=torch.long),
        )
        with torch.no_grad():
            logits, _ = L["model"](*(x.to(dev) for x in batch))
        z = logits.float().cpu().numpy()[0, : len(markers)]
        t = L["temps_by_opts"].get(temp_bucket(0, len(markers)), L["temps"][0])
        zz = z / max(1e-3, float(t))
        p = np.exp(zz - zz.max())
        p /= p.sum()
        merged: dict[str, float] = {}
        for i in range(len(markers)):
            desc = self._desc(json.loads(legal[i]))
            merged[desc] = merged.get(desc, 0.0) + float(p[i])
        out = sorted(merged.items(), key=lambda x: -x[1])
        return out[:top]

    @staticmethod
    def _desc(a: dict) -> str:
        d = a["type"]
        if a.get("pai"):
            d += f":{a['pai']}"
        return d
