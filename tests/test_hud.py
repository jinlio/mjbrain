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


def test_stale_lines_grace_keeps_normal_view():
    rec = {"ts": "2026-09-25T17:07:25", "seat": 3, "zh_recommend": "立直"}
    lines, stale = hud.stale_lines(rec, idle_s=89.9)
    assert stale is False
    assert lines == hud.format_lines(rec)


def test_stale_lines_stale_appends_note():
    rec = {"ts": "2026-09-25T17:07:25", "seat": 3, "zh_recommend": "立直"}
    lines, stale = hud.stale_lines(rec, idle_s=600)
    assert stale is True
    assert lines[:-1] == hud.format_lines(rec)  # 最后一条留档回看
    assert "无新事件 10 分钟" in lines[-1] and "17:07:25" in lines[-1]


def test_stale_lines_never_seen_record_not_stale():
    # 初始态（文件还没落第一行）不算"停更"，保持"等待建议…"亮色
    lines, stale = hud.stale_lines(None, idle_s=10**6)
    assert stale is False and lines == ["等待建议…"]


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
