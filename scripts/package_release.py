"""打包微调权重为 GitHub Release 附件。

用法:
    python scripts/package_release.py --ckpt checkpoints/<run_id> \
        [--run runs/<run_id>] [--tag <名字>] [--out dist]

产出 dist/mjbrain-weights-<tag>.zip：
    model.safetensors + encoder/ + tokenizer/ + rl_agent_config.json
    + MODEL_CARD.md（包内模型卡：出处/指标/使用限制）
    + SHA256SUMS（包内逐文件校验）
终端打印外层 zip 的 sha256，供 Release 页粘贴。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import zipfile

CARD = """# mjbrain-weights · {tag}

基于 [LAYA](https://huggingface.co/convaiinnovations/laya) 微调的立直麻将决策推荐
权重（4 人麻将，雀魂口径）。配套仓库提供规则引擎与推荐服务：**只推荐、不自动**——
合法动作集由确定性代码生成，权重只在合法子集内做带概率的选择。

- 训练 run：`{run_id}`
- 训练配置：items={items}、steps={steps}、epochs={epochs}、train={train_h}h
- 发布日期：{date}

## 预期用途与限制

- **研究 / 教育 / 学习参考**；不建议用于真实对局，模型输出不构成任何建议。
- 本包不含任何游戏协议、请求构造或自动化操作能力——推荐与实际出牌之间永远隔着人手。

## 出处与数据

| 项 | 内容 |
|---|---|
| 底座 | convaiinnovations/laya（Apache-2.0） |
| 微调语料 | [tenhou-houou-mjai](https://huggingface.co/datasets/hhim8826/tenhou-houou-mjai)（凤王牌譜；上游为个人研究/训练用途条款，**语料不随本包分发**） |
| 教师 | [Mortal](https://github.com/herentus/mortal) 输出软标签蒸馏（其代码/原权重 AGPL，**不随本包分发**） |
| 训练方法 | RLCD 风格 human × teacher 混合软标签，单卡 8-bit 微调（`train/rlcd_sft.py`） |

## Held-out 指标（{val_items} 决策点，与教师 top-1 一致率）

| 指标 | 值 |
|---|---|
{metrics_rows}

## 文件与校验

包内 `SHA256SUMS`（`sha256sum -c SHA256SUMS` 逐文件校验）；外层 zip 摘要见
Release 页正文。

## 许可

**Apache-2.0**（与配套仓库一致）。训练语料上游为个人研究/训练用途条款，
**再分发或商用前请自行复核上游条款**；本权重不附带任何真实对局使用许可。
"""


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics_rows(summary: dict) -> str:
    held = summary.get("heldout") or {}
    keys = [k for k in sorted(held) if k.startswith("top1_")]
    if "conf1_ALL" in held:
        keys.append("conf1_ALL")
    if not keys:
        return "| （summary 无 heldout） | — |"
    return "\n".join(f"| {k} | {held[k] * 100:.1f}% |" for k in keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=pathlib.Path, help="权重目录")
    ap.add_argument("--run", type=pathlib.Path, help="对应 runs/<id>，取 summary.json 填指标")
    ap.add_argument("--tag", help="发布名（默认取 ckpt 目录名）")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("dist"))
    args = ap.parse_args()

    ck = args.ckpt
    for req in ("model.safetensors", "encoder", "tokenizer"):
        if not (ck / req).exists():
            raise SystemExit(f"缺少 {ck / req}")
    tag = args.tag or ck.name
    args.out.mkdir(parents=True, exist_ok=True)
    zip_path = args.out / f"mjbrain-weights-{tag}.zip"

    summary = {}
    if args.run and (args.run / "summary.json").exists():
        summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    card = CARD.format(
        tag=tag,
        run_id=summary.get("run_id", ck.name),
        items=summary.get("items", "?"),
        steps=summary.get("steps", "?"),
        epochs=summary.get("epochs", "?"),
        train_h=round(summary.get("train_seconds", 0) / 3600, 1) if summary else "?",
        val_items=summary.get("val_items", "?"),
        date=datetime.datetime.now().astimezone().date().isoformat(),
        metrics_rows=metrics_rows(summary),
    )

    # 收集文件：三个必需项 + 可选 rl_agent_config.json
    entries: list[tuple[pathlib.Path, str]] = []
    for req in ("model.safetensors", "rl_agent_config.json"):
        p = ck / req
        if p.exists():
            entries.append((p, req))
    for sub in ("encoder", "tokenizer"):
        for p in sorted((ck / sub).rglob("*")):
            if p.is_file():
                entries.append((p, str(p.relative_to(ck)).replace("\\", "/")))

    sums = []
    for p, arc in entries:
        sums.append(f"{sha256_file(p)}  {arc}")

    tmp = zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        for p, arc in entries:
            # safetensors 已是紧凑二进制，压缩只烧 CPU 不减体积
            ct = (
                zipfile.ZIP_STORED
                if p.suffix == ".safetensors"
                else zipfile.ZIP_DEFLATED
            )
            zf.write(p, arc, compress_type=ct)
        zf.writestr("MODEL_CARD.md", card, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("SHA256SUMS", "\n".join(sums) + "\n", compress_type=zipfile.ZIP_DEFLATED)
    tmp.replace(zip_path)

    zsha = sha256_file(zip_path)
    (zip_path.with_suffix(zip_path.suffix + ".sha256")).write_text(
        f"{zsha}  {zip_path.name}\n", encoding="utf-8"
    )
    size_mb = zip_path.stat().st_size / (1 << 20)
    print(f"打包完成: {zip_path}  ({size_mb:.0f} MB)")
    print(f"sha256:  {zsha}")
    print("Release 页粘贴上面两行；MODEL_CARD.md 已入包。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
