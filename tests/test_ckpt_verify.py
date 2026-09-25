"""权重完整性校验纯函数（eval.laya_bot._verify_ckpt_dir 一族）。

全走 tmp_path 假目录，不触真权重/模型库。
"""

from __future__ import annotations

import hashlib

import pytest

from eval.laya_bot import _read_sums, _verify_ckpt_dir


def _mk(ck, name: str, data: bytes) -> str:
    p = ck / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_read_sums_parse(tmp_path):
    m = tmp_path / "SHA256SUMS"
    m.write_text("aa..  model.safetensors\n\nbb..  *encoder/config.json\n"
                 "cc..  tokenizer\\vocab.txt\n", encoding="utf-8")
    got = _read_sums(m)
    assert got == {"model.safetensors": "aa..", "encoder/config.json": "bb..",
                   "tokenizer/vocab.txt": "cc.."}
    bad = tmp_path / "SHA256SUMS"
    bad.write_text("no-separator-line\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="行格式"):
        _read_sums(bad)


def test_verify_pass_and_missing_file(tmp_path):
    h = _mk(tmp_path, "model.safetensors", b"w")
    (tmp_path / "VERIFY-SHA256.txt").write_text(
        f"{h}  model.safetensors\n{h}  encoder/config.json\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="缺失"):
        _verify_ckpt_dir(tmp_path)


def test_verify_tamper_raises(tmp_path):
    h = _mk(tmp_path, "model.safetensors", b"good")
    (tmp_path / "VERIFY-SHA256.txt").write_text(
        f"{h}  model.safetensors\n", encoding="utf-8")
    _verify_ckpt_dir(tmp_path)  # 基线须过
    (tmp_path / "model.safetensors").write_bytes(b"tampered!")
    with pytest.raises(RuntimeError, match="失配"):
        _verify_ckpt_dir(tmp_path)


def test_verify_rejects_manifest_escape(tmp_path):
    victim = tmp_path.parent / "victim"
    h = _mk(tmp_path, "model.safetensors", b"x")
    # 清单塞 ../ 行：校验逻辑不许变成任意文件读取器
    (tmp_path / "VERIFY-SHA256.txt").write_text(
        f"{h}  ../victim\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="越界"):
        _verify_ckpt_dir(tmp_path)
    _ = victim  # 仅示意：文件是否存在都无所谓，越界检查先于读取


def test_verify_no_manifest_is_silent(tmp_path):
    _mk(tmp_path, "model.safetensors", b"x")
    _verify_ckpt_dir(tmp_path)  # 训练 run 原生 ckpt 无清单：不炸


def test_verify_escape_hatch_env(tmp_path, monkeypatch, capsys):
    (tmp_path / "VERIFY-SHA256.txt").write_text(
        "deadbeef  model.safetensors\n", encoding="utf-8")  # 文件也不存在
    monkeypatch.setenv("MJBRAIN_SKIP_CKPT_VERIFY", "1")
    _verify_ckpt_dir(tmp_path)  # 逃生门：只警告不炸
    assert "已跳过" in capsys.readouterr().out
