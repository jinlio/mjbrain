"""窗口几何只读发布（CDP Browser.* -> bounds JSON）单测：不碰真浏览器。"""
import asyncio
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from capture.chromium import cdp_client


class FakeW(cdp_client.CdpWatcher):
    def __init__(self, pages, replies, bounds_path):
        super().__init__("ws://fake", lambda *a: None, bounds_path=bounds_path)
        self._pages = pages
        self._replies = list(replies)
        self.calls = []

    async def _send(self, method, params=None, session_id=None):
        self.calls.append((method, params))
        return self._replies.pop(0)


def _page():
    return cdp_client.PageRef("t1", "s1", "https://game.maj-soul.com/1/")


def test_publish_bounds_writes_atomic_json(tmp_path):
    bp = tmp_path / "browser-bounds.json"
    w = FakeW({"t1": _page()},
              [{"windowId": 7},
               {"bounds": {"left": 10, "top": 20, "width": 1600, "height": 900}}],
              bp)
    asyncio.run(w._publish_bounds())
    rec = json.loads(bp.read_text(encoding="utf-8"))
    assert (rec["x"], rec["y"], rec["w"], rec["h"]) == (10, 20, 1600, 900)
    assert rec["ts"] > 0 and "maj-soul" in rec["url"]
    assert w.calls == [("Browser.getWindowForTarget", {"targetId": "t1"}),
                       ("Browser.getWindowBounds", {"windowId": 7})]
    assert not (tmp_path / "browser-bounds.json.tmp").exists()  # 原子替换无残留


def test_publish_bounds_no_page_no_file(tmp_path):
    bp = tmp_path / "b.json"
    w = FakeW({}, [], bp)
    asyncio.run(w._publish_bounds())
    assert not bp.exists() and w.calls == []


def test_publish_bounds_incomplete_response_no_file(tmp_path):
    bp = tmp_path / "b.json"
    w = FakeW({"t1": _page()}, [{"windowId": 7}, {"bounds": {}}], bp)
    asyncio.run(w._publish_bounds())
    assert not bp.exists()


def test_bounds_path_defaults_none():
    w = cdp_client.CdpWatcher("ws://x", lambda *a: None)
    assert w._bounds_path is None
