"""B3 — 微调 LAYA 竞技场入口（M2 门槛候选）。

不自己重演牌局（step 制 sim 的牌山按 seed 播种，外部重演器无法复现），
而是走协议扩展：声明 `wants_ob=True`，engine/sim 直接把**决策时刻的真实
Observation** 递进来 —— 与训练语料完全同源的 obs（同为 get_observation
产物），state_text 零漂移。遮蔽由引擎负责，本模块零手工特征。

流程：brain.infer.forward_scaled_logits（与线上 advisor 共用的唯一前向核）
-> 按校准温度（temp_bucket 优先，回退 qtype 表）取 argmax。
"""

from __future__ import annotations

import json
import pathlib
import threading
from collections import Counter

_CACHE: dict[str, dict] = {}
_LOCK = threading.Lock()  # 多线程首载同 ckpt 会双份模型进显存


def _load(ckpt: str) -> dict:
    """按目录缓存 (tok, model, cfg 温度)。多 bot 实例共享同一权重。"""
    import torch
    from laya.common import build_model
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    # expanduser：cmd/PowerShell 不展开 ~，env 里写 ~/ckpt 要能认
    key = str(pathlib.Path(ckpt).expanduser().resolve())
    if key in _CACHE:
        return _CACHE[key]
    with _LOCK:
        if key in _CACHE:  # 等锁期间别人已载好
            return _CACHE[key]
        return _build_locked(key)


def _sha256_file(path: pathlib.Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _read_sums(path: pathlib.Path) -> dict[str, str]:
    """`<sha256>  <相对路径>` 行（shasum -c 格式）-> {rel: hash}。"""
    out: dict[str, str] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split(None, 1)
        if len(parts) != 2:
            raise RuntimeError(f"{path.name} 行格式坏：{ln!r}")
        out[parts[1].strip().lstrip("*").replace("\\", "/")] = parts[0].lower()
    return out


def _verify_ckpt_dir(ck: pathlib.Path) -> None:
    """权重完整性：载模型前逐条比对目录内 SHA256SUMS / VERIFY-SHA256.txt。

    威胁模型=Release 资产换包（advisor 被喂假权重=建议注入）。清单里列的
    每个文件必须存在且哈希一致，否则 RuntimeError；没有清单（训练 run 的
    原生 ckpt）静默跳过。逃生门 MJBRAIN_SKIP_CKPT_VERIFY=1（调试专用）。
    """
    import os

    if os.environ.get("MJBRAIN_SKIP_CKPT_VERIFY") == "1":
        print(f"[laya_bot] 警告：MJBRAIN_SKIP_CKPT_VERIFY=1，已跳过 {ck} 完整性校验")
        return
    manifest = None
    for name in ("SHA256SUMS", "VERIFY-SHA256.txt"):
        cand = ck / name
        if cand.exists():
            manifest = cand
            break
    if manifest is None:
        return
    for rel, want in _read_sums(manifest).items():
        target = (ck / rel).resolve()
        if not target.is_relative_to(ck.resolve()):  # 清单被塞 ../ 行也拦住
            raise RuntimeError(f"完整性清单路径越界：{rel!r}")
        if not target.exists():
            raise RuntimeError(f"权重文件缺失：{target}（清单 {manifest.name}）")
        got = _sha256_file(target)
        if got != want:
            raise RuntimeError(
                f"权重完整性失配：{rel}\n  期望 {want}\n  实得 {got}\n"
                f"  来源可疑（换包/截断）——勿用此 ckpt，重新从 Release 下载")


def _build_locked(key: str) -> dict:
    import torch
    from laya.common import build_model
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    p = pathlib.Path(key)
    with open(p / "rl_agent_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    tok_dir = p / "tokenizer"
    if not tok_dir.exists():
        # 不给 from_pretrained 兜底成 hub repo-id 的机会：离线机整挂起重试
        raise RuntimeError(f"tokenizer 目录缺失：{tok_dir}（Release 解压不全？）")
    enc_dir = p / "encoder"
    if not enc_dir.exists():
        # build_model 在 encoder 目录缺失时按 cfg["encoder"] 的 repo-id 联网
        # 拉 hub 模型——被投毒的 rl_agent_config.json 可借此执行远端代码。
        # 与 tokenizer 同口径硬检，彻底断网。
        raise RuntimeError(f"encoder 目录缺失：{enc_dir}（拒绝 hub 回退）")
    _verify_ckpt_dir(p)  # 载权重前最后一道闸：哈希对不上什么都不建
    # trust_remote_code=False：把"版本默认"变成"结构保证"
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=False)
    model = build_model(cfg, encoder_dir=str(enc_dir))
    model.load_state_dict(load_file(str(p / "model.safetensors")), strict=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")  # 外协 Mac（M 系）走 MPS；fp32，autocast 仅 cuda 开
        # getattr 守卫：老/非 mac torch 构建没有 backends.mps 属性，直取 AttributeError
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
        if not ck or not pathlib.Path(ck).expanduser().exists():
            raise RuntimeError("B3 需要 ckpt 目录或环境变量 MJ_LAYA_CKPT")
        self._L = _load(ck)
        self._seed = seed
        self.stats = Counter()

    def react(self, events, seat: int, legal_actions: list[str], *, ob=None) -> str:
        import brain.infer as infer

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

        try:  # 前向段（build_sequence/模型/tokenizer）任何异常：弃这一手
            # 过牌继续，数小时的 arena 长跑不许被一次 OOM/MPS 抖动打断；
            # 成功路径数值零改动（纯兜底，fallback 统计键 forward_error）。
            # "fp16" 即历史口径：cuda 上半精度，其余设备 fp32。
            z = infer.forward_scaled_logits(
                self._L, self._L["device"], ob, legal_actions, "fp16")
        except infer.Truncated:
            return fallback("head_truncated")
        except Exception:  # noqa: BLE001 — 见上：仅兜住原本会崩整跑的路径
            return fallback("forward_error")
        self.stats["act"] += 1
        return legal_actions[int(z.argmax())]
