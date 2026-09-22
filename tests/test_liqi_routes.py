"""真 descriptor 联测（移植 Akagi parser.rs 的 all_routes_resolve 不变量 +
合成真实帧 round-trip）。需要 scripts/compile_liqi_proto.py 的生成物；
未编译时整文件 skip（生成物不进 git）。"""

import pathlib

import pytest

_gen = pathlib.Path(__file__).parents[1] / "capture" / "liqi" / "_gen" / "liqi_pb2.py"
pytestmark = pytest.mark.skipif(not _gen.exists(), reason="先跑 scripts/compile_liqi_proto.py")


def test_all_routes_resolve():
    """每条 rpc 的 req/resp 类型必须能在 descriptor pool 解析——防 proto/liqi.json 漂移。"""
    from capture.liqi import runtime

    pool = runtime._desc_pool()
    routes = runtime.routes()
    assert len(routes) > 300, f"路由数可疑：{len(routes)}"
    for method, spec in routes.items():
        for role in ("req", "resp"):
            fqn = spec[role].lstrip(".")
            assert pool.FindMessageTypeByName(fqn) is not None, (
                f"{method} {role} {fqn} 不在 pool"
            )


def test_real_notify_roundtrip():
    from google.protobuf import message_factory

    from capture.liqi import runtime
    from capture.liqi.decode import NOTIFY

    pool = runtime._desc_pool()
    name = "lq.NotifyAccountUpdate"
    desc = pool.FindMessageTypeByName(name)
    cls = message_factory.GetMessageClass(desc)
    body = cls().SerializeToString()

    def _varint(n):
        out = bytearray()
        while True:
            b = n & 0x7F
            n >>= 7
            out.append(b | (0x80 if n else 0))
            if not n:
                return bytes(out)

    def _wrapper(nm, data):
        nb = nm.encode()
        return b"\x0a" + _varint(len(nb)) + nb + b"\x12" + _varint(len(data)) + data

    p = runtime.make_parser()
    # NotifyAccountUpdate{AccountUpdate update=1}：用真实字段 new_recharged_list(uint32 rep)
    upd = message_factory.GetMessageClass(pool.FindMessageTypeByName("lq.AccountUpdate"))()
    upd.new_recharged_list.append(4321)
    body = cls(update=upd).SerializeToString()
    fr = p.parse(bytes([NOTIFY]) + _wrapper(name, body))
    assert fr is not None and fr.method == "lq.NotifyAccountUpdate"
    assert fr.payload["update"]["new_recharged_list"] == [4321]


def test_real_request_response_pending():
    from google.protobuf import message_factory

    from capture.liqi import runtime
    from capture.liqi.decode import REQUEST, RESPONSE

    pool = runtime._desc_pool()
    routes = runtime.routes()
    method = ".lq.FastTest.authGame"
    spec = routes[method]

    def _varint(n):
        out = bytearray()
        while True:
            b = n & 0x7F
            n >>= 7
            out.append(b | (0x80 if n else 0))
            if not n:
                return bytes(out)

    def _wrapper(nm, data):
        nb = nm.encode()
        return b"\x0a" + _varint(len(nb)) + nb + b"\x12" + _varint(len(data)) + data

    req_cls = message_factory.GetMessageClass(pool.FindMessageTypeByName(spec["req"].lstrip(".")))
    resp_cls = message_factory.GetMessageClass(
        pool.FindMessageTypeByName(spec["resp"].lstrip("."))
    )
    req_body = req_cls().SerializeToString()
    resp_body = resp_cls().SerializeToString()

    p = runtime.make_parser()
    fr = p.parse(bytes([REQUEST]) + (7).to_bytes(2, "little") + _wrapper(method, req_body))
    assert fr.kind == REQUEST and fr.msg_id == 7
    fr2 = p.parse(bytes([RESPONSE]) + (7).to_bytes(2, "little") + _wrapper("", resp_body))
    assert fr2.kind == RESPONSE and fr2.method == method
    assert isinstance(fr2.payload, dict)


# ---------- 状态层整机（真描述符）：字段锁 / 二次解码 / GameRestore 金标 ----------

import base64

from capture.liqi.decode import NOTIFY, wtf_decode


def _wrapper(n, data):
    nb = n.encode()
    return b"\x0a" + bytes([len(nb)]) + nb + b"\x12" + bytes([len(data)]) + data if len(data) < 128 else None


def _varint(x):
    out = bytearray()
    while True:
        b = x & 0x7F
        x >>= 7
        out.append(b | (0x80 if x else 0))
        if not x:
            return bytes(out)


def _wrap(n, data):
    nb = n.encode()
    return b"\x0a" + _varint(len(nb)) + nb + b"\x12" + _varint(len(data)) + data


# parser.rs::gameplay_field_numbers_stable 镜像：映射层按 JSON 键读字段，
# 重排号会静默错值——协议漂移必须在字段号层拦住。
GAMEPLAY_FIELD_LOCK = [
    ("lq.ActionPrototype", {"step": 1, "name": 2, "data": 3}),
    ("lq.ActionNewRound", {"chang": 1, "ju": 2, "ben": 3, "tiles": 4,
                           "scores": 6, "liqibang": 8, "doras": 14}),
    ("lq.ActionDiscardTile", {"seat": 1, "tile": 2, "is_liqi": 3, "moqie": 5,
                              "doras": 8, "is_wliqi": 9}),
    ("lq.ActionDealTile", {"seat": 1, "tile": 2, "liqi": 5, "doras": 6}),
    ("lq.ActionChiPengGang", {"seat": 1, "type": 2, "tiles": 3, "froms": 4}),
    ("lq.ActionHule", {"hules": 1, "delta_scores": 3, "scores": 5}),
    ("lq.ReqSelfOperation", {"type": 1, "tile": 3, "moqie": 5}),
]


def test_gameplay_field_numbers_stable():
    from capture.liqi import runtime

    pool = runtime._desc_pool()
    for msg_name, fields in GAMEPLAY_FIELD_LOCK:
        desc = pool.FindMessageTypeByName(msg_name)
        for fname, number in fields.items():
            f = desc.fields_by_name[fname]
            assert f.number == number, f"{msg_name}.{fname} renumbered"


def test_live_action_prototype_second_decode():
    from google.protobuf import message_factory

    from capture.liqi import runtime

    pool = runtime._desc_pool()
    dt = message_factory.GetMessageClass(pool.FindMessageTypeByName("lq.ActionDiscardTile"))()
    dt.seat = 2
    dt.tile = "1m"
    dt.moqie = True
    inner = dt.SerializeToString()
    buf = bytearray(inner)
    wtf_decode(buf)  # live notify 带位置相关 XOR
    ap = message_factory.GetMessageClass(pool.FindMessageTypeByName("lq.ActionPrototype"))()
    ap.step = 9
    ap.name = "ActionDiscardTile"
    ap.data = bytes(buf)
    frame = bytes([NOTIFY]) + _wrap("lq.ActionPrototype", ap.SerializeToString())

    _p, st = runtime.make_stack()
    st.seat = 2
    fr = _p.parse(frame)
    assert fr.method == "lq.ActionPrototype"
    assert fr.payload["name"] == "ActionDiscardTile"
    assert isinstance(fr.payload["data"], dict)  # 已二次解码
    ev = st.dispatch(fr.kind, fr.method, fr.payload)
    assert ev == [{"type": "dahai", "actor": 2, "pai": "1m", "tsumogiri": True}]


# Akagi parser.rs 测试的实采 GameRestore ActionNewRound（13 枚手牌可读段）
RESTORE_NEW_ROUND_B64 = (
    "CAAQABgAIgI0cCICMW0iAjNwIgI0eiICN3oiAjZ6IgIwcCICM3MiAjBtIgI0cyICMXMiAjlzIgIyejIM"
    "qMMBqMMBqMMBqMMBQABYAGhFcgIxenoCCAB6AggBegIIAnoCCAOaAUA1NWQ2NzQ3MTRjNjAzODFhNGJj"
    "OTJmYzBmOWIwNWVjNDU4OWZlMzI0NTQ4OWVmOGY3NTc5NTVmMzIzZWIzZTE1qgFAYzFmNmY1YTQwOGFj"
    "MzgyZGEyZGE1MTA3OTUzYmUzYTg1N2M5OTdmNGNkMjg2M2JlZjczY2M3ZjE3MDllMDA4Mw=="
)


def test_restore_golden_actionnewround():
    from capture.liqi import runtime
    from capture.mjai.tile import ms_to_mjai, sort_pais

    _p, st = runtime.make_stack()
    st.seat = 1  # fixture 是东风局（chang/ju/ben 皆缺省=0），取非庄视角 13 枚
    ev = st.handle_game_restore({"game_restore": {"actions": [
        {"name": "ActionNewRound", "data": RESTORE_NEW_ROUND_B64}]}})
    assert ev[0]["type"] == "start_kyoku"
    sk = ev[0]
    assert sk["bakaze"] == "E" and sk["kyoku"] == 1 and sk["oya"] == 0
    assert sk["num_players"] == 4
    assert len(sk["scores"]) == 4 and all(s > 0 for s in sk["scores"])
    ms_hand = ["4p", "1m", "3p", "4z", "7z", "6z", "0p", "3s", "0m", "4s", "1s", "9s", "2z"]
    assert sk["tehais"][1] == sort_pais([ms_to_mjai(t) for t in ms_hand])
    assert ev[1] == {"type": "tsumo", "actor": 0, "pai": "?"}
    # 真字节 → 真描述符 → 真映射：整链金标（dora=1z→E 独立复核）
    raw = runtime.message_to_dict("lq.ActionNewRound",
                                  base64.b64decode(RESTORE_NEW_ROUND_B64))
    assert sk["dora_marker"] == ms_to_mjai(raw["doras"][0]) == "E"


def test_restore_unknown_action_is_noop():
    from capture.liqi import runtime

    _p, st = runtime.make_stack()
    st.seat = 0
    # ActionMJStart 有消息无 handler → 解码成功、零事件，不炸链
    ev = st.handle_game_restore({"game_restore": {"actions": [
        {"name": "ActionMJStart", "data": "CgJ1LTEYAg=="}]}})
    assert ev == []
