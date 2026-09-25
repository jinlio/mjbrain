"""档1 帧管道离线链：capture JSONL 行 → parser/state → mjai 事件流。

真 protobuf 消息 → 真 wrapper 字节 → 真 base64 → run_capture 行格式 →
feed_lines 解出事件。这是 live_from_capture.py 的离线可验证部分；
/v1/react 的流→推荐链由 test_advisor_server 覆盖（同事件契约衔接），
两端拼起来即档1 全链，真机验收只剩灌流。
"""

from __future__ import annotations

import base64
import importlib.util
import json
import pathlib

import pytest
from google.protobuf import message_factory

from capture.liqi.decode import NOTIFY, wtf_decode

_gen = pathlib.Path(__file__).parents[1] / "capture" / "liqi" / "_gen" / "liqi_pb2.py"
# 只有走真 protobuf 的两条用例需要 _gen（生成物不进 git）；显示层用例纯格式化，
# 无生成码也能跑——故按用例打标而非整文件 skip。
needs_gen = pytest.mark.skipif(not _gen.exists(), reason="先跑 scripts/compile_liqi_proto.py")


def _load_glue():
    path = pathlib.Path(__file__).parents[1] / "scripts" / "live_from_capture.py"
    spec = importlib.util.spec_from_file_location("live_from_capture", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _varint(x: int) -> bytes:
    out = bytearray()
    while True:
        b = x & 0x7F
        x >>= 7
        out.append(b | (0x80 if x else 0))
        if not x:
            return bytes(out)


def _wrap(n: str, data: bytes) -> bytes:
    nb = n.encode()
    return b"\x0a" + _varint(len(nb)) + nb + b"\x12" + _varint(len(data)) + data


def _notify_action(pool, act_name: str, act_msg) -> bytes:
    """镜像雀魂 live notify：内层动作 XOR 后包 ActionPrototype。"""
    buf = bytearray(act_msg.SerializeToString())
    wtf_decode(buf)
    ap = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("lq.ActionPrototype"))()
    ap.step = 1
    ap.name = act_name
    ap.data = bytes(buf)
    return bytes([NOTIFY]) + _wrap("lq.ActionPrototype", ap.SerializeToString())


def _line(frame: bytes) -> str:
    return json.dumps({"t": 0.0, "dir": "recv", "page": "ws", "rid": "1",
                       "len": len(frame), "b64": base64.b64encode(frame).decode()})


@needs_gen
def test_jsonl_to_mjai_events(tmp_path):
    from capture.liqi import runtime

    pool = runtime._desc_pool()
    deal = message_factory.GetMessageClass(pool.FindMessageTypeByName("lq.ActionDealTile"))()
    deal.seat = 1
    deal.tile = "3p"
    disc = message_factory.GetMessageClass(pool.FindMessageTypeByName("lq.ActionDiscardTile"))()
    disc.seat = 1
    disc.tile = "3p"
    disc.moqie = True

    frames = [_notify_action(pool, "ActionDealTile", deal),
              _notify_action(pool, "ActionDiscardTile", disc)]
    lines = [_line(f) for f in frames]
    # 噪声行：非 JSON / 非协议字节 / 空行——必须被静默吞掉，不断链
    noise = ["{not json", _line(base64.b64decode("bm90LWxpcWktZnJhbWU=")), ""]

    glue = _load_glue()
    p, st = runtime.make_stack()
    st.seat = 1  # 无 AuthGame 帧时等价于 --seat 手工指定
    events: list[dict] = []
    assert glue.feed_lines(p, st, events, noise + lines[:1]) is True
    assert events == [{"type": "tsumo", "actor": 1, "pai": "3p"}]
    assert glue.feed_lines(p, st, events, lines[1:]) is True
    assert events[1:] == [{"type": "dahai", "actor": 1, "pai": "3p",
                           "tsumogiri": True}]


@needs_gen
def test_feed_lines_no_events_on_irrelevant_frame():
    from capture.liqi import runtime

    glue = _load_glue()
    p, st = runtime.make_stack()
    st.seat = 0
    events: list[dict] = []
    # 一条连 parser 都不认的字节流（type byte 未知）→ 无增长、无异常
    line = _line(bytes([0x7F]) + b"garbage")
    assert glue.feed_lines(p, st, events, [line]) is False
    assert events == []


def test_fmt_decision_marks_own_seat():
    """座位标注：自家座位加（自家），非自家/座位未知时不加（防"座位没变"误读）。"""
    glue = _load_glue()
    d = {"seat": 1, "legal_n": 3, "recommend": "dahai:5p",
         "top": [{"a": "dahai:5p", "p": 0.5}]}
    assert "座位1（自家） 候选3 → 切5饼" in glue.fmt_decision(d, my_seat=1)
    assert "（自家）" not in glue.fmt_decision(d, my_seat=0)
    assert "（自家）" not in glue.fmt_decision(d, None)
    assert glue.fmt_decision({"seat": 1, "window": False}, 1) == "座位1（自家）: 无决策窗"
    assert glue.fmt_decision({"seat": 2, "window": False}, 1) == "座位2: 无决策窗"


def test_fmt_decision_appends_hint():
    """hint 段：服务端附带时拼在行尾；无 hint（None/缺字段）不加竖线。"""
    glue = _load_glue()
    d = {"seat": 1, "legal_n": 3, "recommend": "dahai:5p",
         "top": [{"a": "dahai:5p", "p": 0.5}],
         "hint": {"shanten": 0, "waits": ["3m", "6m"], "dora": ["5p"],
                  "dora_in_hand": 0, "melded": False}}
    line = glue.fmt_decision(d, my_seat=1)
    assert line.endswith("| 向听0 待[3万 6万] 宝[5饼]")
    assert "候选3 → 切5饼" in line  # 竖线分隔，与 top 段并列
    for no_hint in (dict(d, hint=None), {k: v for k, v in d.items() if k != "hint"}):
        assert "向听" not in glue.fmt_decision(no_hint, my_seat=1)


def test_fmt_reach_chinese():
    """立直宣言牌建议的显示行：mjai 串经 advisor.zh → 中文，取 top3。"""
    glue = _load_glue()
    line = glue.fmt_reach({
        "legal_n": 3, "recommend": "dahai:W",
        "top": [{"a": "dahai:W", "p": 0.583}, {"a": "dahai:7p", "p": 0.417},
                {"a": "dahai:1s", "p": 0.05}, {"a": "dahai:9m", "p": 0.01}],
    })
    assert line.startswith("   ↳ 若立直：宣言牌 → 切西 ")
    assert "切西 58.3%" in line and "切7饼 41.7%" in line and "切1索 5.0%" in line
    assert "切9万" not in line  # 只显示 top3


def test_cdp_frame_decode_guards():
    from capture.chromium.cdp_client import decode_frame_payload as dec
    # 畸形/截断 base64（payloadData 半截）过去抛 binascii.Error 掀读循环
    assert dec({"opcode": 2, "payloadData": "!!!"}) is None
    assert dec({"opcode": 2, "payloadData": "aGVsbG8="}) == b"hello"
    assert dec({"opcode": 1, "payloadData": "文"}) == "文".encode()
    assert dec({"opcode": 9, "payloadData": "aGVsbG8="}) is None  # ping 丢弃
    assert dec({}) is None


def test_cdp_event_without_params_does_not_kill_loop():
    """缺 params 的事件按空处理：读循环容错的前提是分发器自身不抛。"""
    import asyncio

    from capture.chromium.cdp_client import CdpWatcher, PageRef

    w = CdpWatcher.__new__(CdpWatcher)
    w._pages = {"t1": PageRef("t1", "s1", "ws://game")}
    w._req_page = {}
    w._sink = lambda *a: None
    for m in ("Network.webSocketCreated", "Network.webSocketClosed",
              "Network.webSocketFrameReceived", "Network.webSocketFrameSent"):
        asyncio.run(w._on_event({"method": m, "sessionId": "s1"}))  # 不抛即过
    asyncio.run(w._on_event({"method": "Page.javascriptDialogOpening"}))
