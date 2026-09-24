"""scripts/run_demo.py 的纯逻辑：轮转、命令行、ckpt 链探测（不拉子进程）。"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "run_demo", pathlib.Path(__file__).parents[1] / "scripts" / "run_demo.py")
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)


def _args(**kw) -> argparse.Namespace:
    base = dict(url="https://game.maj-soul.com/1/", seat=-1, top=5,
                window_size="1600x900", window_position="", no_app=False,
                no_hud=False, device="", ckpt=None, dry=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_next_frames_rotation(tmp_path):
    d = tmp_path / "ms"
    f, a = rd.next_frames_path(d)
    assert f.name == "run1.jsonl" and a.name == "run1.advise.jsonl"
    f.touch()
    (d / "run2.jsonl").touch()
    (d / "run10.jsonl").touch()
    (d / "junk.jsonl").touch()          # 非 runN 不参与计数
    f2, a2 = rd.next_frames_path(d)
    assert f2.name == "run11.jsonl"     # max+1 而非线性找空位
    assert a2 == d / "run11.advise.jsonl"


def test_build_cmds_flags(tmp_path):
    cmds = rd.build_cmds(_args(window_position="80,60", no_app=True),
                         tmp_path / "ck", tmp_path / "r.jsonl",
                         tmp_path / "r.advise.jsonl", "cpu")
    cap = cmds["capture"]
    assert "--window-size" in cap and "1600x900" in cap
    assert "--window-position" in cap and "80,60" in cap
    assert "--no-app" in cap
    assert "--out" in cap and str(tmp_path / "r.jsonl") in cap
    live = cmds["live"]
    assert "--follow" in live and "-1" in live
    assert cmds["hud"][2:4] == ["hud.float", "--advise"]
    adv = cmds["advisor"]
    assert "--device" in adv and "cpu" in adv
    assert str(tmp_path / "ck") in adv


def test_build_cmds_defaults_use_app_window(tmp_path):
    cmds = rd.build_cmds(_args(), tmp_path / "ck", tmp_path / "r.jsonl",
                         tmp_path / "r.advise.jsonl", "mps")
    assert "--no-app" not in cmds["capture"]      # 缺省就是 app 独立窗
    assert "mps" == cmds["advisor"][cmds["advisor"].index("--device") + 1]


def test_resolve_ckpt_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "ROOT", tmp_path)
    # 链上全无 → None
    assert rd.resolve_ckpt(None) is None
    # 第二级出现 → 命中它
    d2 = tmp_path / rd.CKPT_CHAIN[1]
    d2.mkdir(parents=True)
    (d2 / "model.safetensors").touch()
    assert rd.resolve_ckpt(None) == d2
    # 显式指定优先（相对路径挂 ROOT）
    assert rd.resolve_ckpt("elsewhere") == tmp_path / "elsewhere"


def test_detect_device_returns_known():
    assert rd.detect_device() in ("cpu", "mps", "cuda")


def test_preflight_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "ROOT", tmp_path)
    rep = "\n".join(rd.preflight(_args(), None, tmp_path / "f.jsonl"))
    assert "ckpt:" in rep and "✗" in rep      # 显式 None：没找到
    assert "浏览器:" in rep
    assert "advisor 设备:" in rep
