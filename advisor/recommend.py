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
        self._dev = self._L["device"]
        if device:
            want = torch.device(device)
            if want != self._dev:
                # 共享 _CACHE 条目：同进程已有别的持有者改道过设备时，
                # 第二例再就地 to(device) 会悄悄把 arena B3 换到别的设备/
                # 精度档上（fp16↔fp32 漂移，破坏可复现对局配）。显式拒绝。
                if self._L.get("device_overridden"):
                    raise RuntimeError(
                        f"ckpt 已在共享缓存被改道至 {self._dev}，"
                        f"拒绝再迁到 {want}（请分进程）")
                self._L["model"].to(want)
                self._L["device"] = self._dev = want
                self._L["device_overridden"] = True

    def topk(self, ob, legal: list[str], top: int = 5) -> list[tuple[str, float]]:
        """返回 [(动作描述, 概率)]，按温度校准 softmax；概率和=1。"""
        import numpy as np
        import torch
        from laya.common import build_sequence, temp_bucket

        from brain.serialize import laya_question, state_text

        if not legal:
            raise ValueError("topk：legal 为空，无可选动作")
        top = max(1, int(top))  # 负/0 会让 out[:top] 静默丢尾或丢全
        if len(legal) == 1:
            a = json.loads(legal[0])
            return [(self._desc(a), 1.0)]
        q = laya_question(legal)
        L = self._L
        ids, markers = build_sequence(
            L["tok"], state_text(ob), q, L["max_len"], L["head_max_len"])
        if len(markers) != len(q["crit"]):
            raise ValueError("选项被序列截断（加大 head_max_len）")
        dev = self._dev  # 实例快照：与 __init__ 迁移决策一致，防共享字典被改
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
