"""hud.float 的非 GUI 部分：渲染纯函数 + 设置持久化（Tk 窗体属人工验收）。"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "hud_float", pathlib.Path(__file__).parents[1] / "hud" / "float.py")
hud = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hud)


def test_format_lines_full_record():
    rec = {"ts": "2026-09-24T19:44:07", "seat": 1, "event_n": 300,
           "zh_recommend": "切5饼", "zh_top": [["切5饼", 0.5], ["切9条", 0.3],
                                               ["切1万", 0.1], ["碰西", 0.05]],
           "zh_hint": "向听0 待[3万 6万] 宝[5饼]×2",
           "zh_reach": {"recommend": "切西", "top": [["切西", 0.58]]},
           "resp": {}}
    lines = hud.format_lines(rec)
    assert lines[0] == "座位1 19:44:07"
    assert lines[1] == "→ 切5饼"
    assert lines[2] == "切5饼 50%  切9条 30%  切1万 10%"  # top3 截断
    assert lines[3] == "向听0 待[3万 6万] 宝[5饼]×2"
    assert lines[4] == "若立直：宣言牌 切西"


def test_format_lines_no_window_and_missing():
    assert hud.format_lines({}) == ["等待建议…"]
    assert hud.format_lines(None) == ["等待建议…"]
    rec = {"ts": "x", "seat": None, "zh_recommend": None, "zh_top": None}
    lines = hud.format_lines(rec)
    assert lines[0] == "座位?"      # ts 不足 19 字符 → 不拼时间
    assert "（该事件自家未开窗）" in lines


def test_latest_record_skips_broken_lines():
    good = json.dumps({"seat": 2, "zh_recommend": "摸切"})
    assert hud.latest_record(["", "{trunc", good]) == {"seat": 2, "zh_recommend": "摸切"}
    assert hud.latest_record(["not json", "{7"]) is None
    assert hud.latest_record([]) is None


def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    assert hud.load_settings() == {}
    hud.save_settings({"x": 10, "y": 20, "alpha": 0.9})
    assert hud.load_settings() == {"x": 10, "y": 20, "alpha": 0.9}
    # 坏文件 → 空设置而非崩
    hud.settings_path().write_text("{broken", encoding="utf-8")
    assert hud.load_settings() == {}


# ---------- 窗口联动（跟随/相对位置/新鲜度） ----------

def _write_bounds(p, x, y, ts, w=1600, h=900):
    p.write_text(json.dumps({"x": x, "y": y, "w": w, "h": h,
                             "url": "game", "ts": ts}), encoding="utf-8")


def test_read_bounds_fresh_and_stale(tmp_path):
    bp = tmp_path / "browser-bounds.json"
    now = 1000.0
    _write_bounds(bp, 40, 60, now)
    assert hud.read_bounds(bp, max_age=5.0, now=now + 4) == {"x": 40, "y": 60}
    assert hud.read_bounds(bp, max_age=5.0, now=now + 6) is None   # 过期


def test_read_bounds_bad_inputs(tmp_path):
    bp = tmp_path / "b.json"
    assert hud.read_bounds(bp, now=0) is None                       # 不存在
    bp.write_text("{trunc", encoding="utf-8")
    assert hud.read_bounds(bp, now=0) is None                       # 坏 JSON
    _write_bounds(bp, 40, 60, 100.0)
    assert hud.read_bounds(bp, max_age=5.0, now=1000.0) is None     # ts 太老
    _write_bounds(bp, -32000, -32000, 0.0)                          # 最小化离屏
    assert hud.read_bounds(bp, max_age=9999, now=1) is None


def test_start_position_with_and_without_offset():
    s = {"x": 500, "y": 500, "off_x": 100, "off_y": -40}
    b = {"x": 250, "y": 160}
    assert hud.start_position(s, b) == (350, 120)      # 贴回浏览器窗口
    assert hud.start_position(s, None) == (500, 500)   # 浏览器不在→绝对位置
    assert hud.start_position({"x": 7, "y": 9}, b) == (7, 9)  # 没存过偏移


def test_follow_shift():
    assert hud.follow_shift(10, 10, {"x": 0, "y": 0}, {"x": 30, "y": -20}) == (40, -10)
    assert hud.follow_shift(10, 10, {"x": 0, "y": 0}, {"x": 0, "y": 0}) is None  # 没动
    assert hud.follow_shift(10, 10, None, {"x": 5, "y": 5}) is None              # 首次见
    assert hud.follow_shift(10, 10, {"x": 5, "y": 5}, None) is None              # 浏览器关了


def test_bounds_file_under_mjbrain_home(tmp_path, monkeypatch):
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    assert hud.bounds_file() == tmp_path / ".mjbrain" / "browser-bounds.json"
