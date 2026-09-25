"""推荐内核：给定 (ob, legal) 出 B3 top-k；手输/棋谱/服务三侧共用。

只读决策：本模块产出的只是建议文本，永不回写任何游戏通道
（零动作注入红线，见 docs/PLAN.md M4 定线）。
"""

from __future__ import annotations

import json


class Adviser:
    def __init__(self, ckpt: str, device: str | None = None,
                 precision: str | None = None) -> None:
        """precision: "fp32"（默认=线上历史口径）| "fp16"（cuda 上半精度，
        与 arena B3 即发布成绩 74.1% 的实测策略逐位一致）。None 时读
        MJBRAIN_ADVISOR_PRECISION；非法值构造期即报错，不留到决策点。"""
        import os

        import torch

        from brain import infer
        from eval.laya_bot import _load

        self._prec = infer.check_precision(
            precision or os.environ.get("MJBRAIN_ADVISOR_PRECISION", "fp32"))
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
        """返回 [(动作描述, 概率)]，按温度校准 softmax；概率和=1。

        前向走 brain.infer 唯一核（与 arena B3 同一实现）；精度档 =
        self._prec。默认 fp32 与旧行为逐位一致（fp32 路径不建 autocast）。
        """
        import numpy as np

        from brain import infer

        if not legal:
            raise ValueError("topk：legal 为空，无可选动作")
        top = max(1, int(top))  # 负/0 会让 out[:top] 静默丢尾或丢全
        if len(legal) == 1:
            a = json.loads(legal[0])
            return [(self._desc(a), 1.0)]
        # dev 传实例快照：与 __init__ 迁移决策一致，防共享字典被改
        zz = infer.forward_scaled_logits(
            self._L, self._dev, ob, legal, self._prec)
        p = np.exp(zz - zz.max())
        p /= p.sum()
        merged: dict[str, float] = {}
        for i in range(len(zz)):
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
