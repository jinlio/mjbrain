"""M0 冒烟：检验 laya 安装与本机设备能力。

    conda activate mjbrain
    python scripts/smoke_laya.py            # 只查设备，不下载权重
    python scripts/smoke_laya.py --full     # 加载英文 root 权重跑一次 choice
"""

from __future__ import annotations

import argparse
import time


def devices() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch": False}
    out = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }
    if out["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        out["cuda_device"] = f"{p.name} ({p.total_memory / 2**30:.1f} GiB)"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="真正加载模型并预测（首次会下载权重）")
    args = ap.parse_args()

    print("== torch 设备探测 ==")
    for k, v in devices().items():
        print(f"  {k}: {v}")

    try:
        import laya
    except ImportError:
        print("laya 未安装：conda activate mjbrain && pip install laya（会拉 torch，注意网络与磁盘）")
        return 1

    print(f"\nlaya imported: {getattr(laya, '__version__', 'unknown')}")
    if not args.full:
        print("加 --full 做真实加载与一次 choice 预测")
        return 0

    state = {
        "hand": "123m456p789s 东东东 南",
        "prompt": "打哪张？选项含安全牌考量",
    }
    questions = {
        "discard": {
            "type": "choice",
            "instructions": "麻将决策：最优弃牌",
            "criteria": {"east": "打东风(客风安全)", "sou": "打四条(进张优先)", "man": "打一万"},
        }
    }
    t0 = time.perf_counter()
    agent = laya.load("convaiinnovations/laya")
    print(f"load: {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    res = agent.predict(state, questions)
    print(f"predict: {(time.perf_counter() - t0) * 1000:.1f}ms")
    print(res["answers"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
