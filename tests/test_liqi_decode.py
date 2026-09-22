"""liqi 帧层单测：wtf_decode / Wrapper 手解 / 路由与 pending 配对。

合成帧用手工 wire bytes（不依赖 _gen 生成码）；message_to_dict 用 stub，
真实 descriptor 联测在 scripts/compile_liqi_proto.py 之后另行冒烟。
"""

from capture.liqi import decode as D


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _wrapper(name: str, data: bytes) -> bytes:
    nb = name.encode()
    return b"\x0a" + _varint(len(nb)) + nb + b"\x12" + _varint(len(data)) + data


def test_wrapper_roundtrip():
    w = D.decode_wrapper(_wrapper("lq.NotifyFlow", b"\x01\x02\x03"))
    assert w.name == "lq.NotifyFlow" and w.data == b"\x01\x02\x03"


def test_wrapper_long_varint():
    big = bytes(300)
    w = D.decode_wrapper(_wrapper("lq.NotifyX", big))
    assert len(w.data) == 300


def test_wtf_decode_matches_akagi_scheme():
    # 反向：对已知明文做 XOR 得密文，wtf_decode 应还原
    plain = bytearray(b"mahjong-liqi-action!!")
    key = bytes([0x84, 0x5E, 0x4E, 0x42, 0x39, 0xA2, 0x1F, 0x60, 0x1C])
    base = 23 ^ len(plain)
    enc = bytearray(plain)
    for i in range(len(enc)):
        enc[i] ^= (base + 5 * i + key[i % 9]) & 0xFF
    assert enc != plain
    D.wtf_decode(enc)
    assert enc == plain


def _stub_td(fqn: str, data: bytes) -> dict:
    return {"_fqn": fqn, "_raw": data}


def test_notify_frame_two_part_name():
    p = D.LiqiParser(message_to_dict=_stub_td)
    fr = p.parse(bytes([D.NOTIFY]) + _wrapper("lq.NotifyFlow", b"\x7f"))
    assert fr.kind == D.NOTIFY and fr.method == "lq.NotifyFlow" and fr.msg_id is None
    assert fr.payload == {"_fqn": "lq.NotifyFlow", "_raw": b"\x7f"}


def test_request_response_pending_pairing():
    routes = {".lq.FastTest.authGame": {"req": ".lq.ReqAuth", "resp": ".lq.ResAuth"}}
    p = D.LiqiParser(message_to_dict=_stub_td, routes=routes)
    req = bytes([D.REQUEST]) + (42).to_bytes(2, "little") + _wrapper(
        ".lq.FastTest.authGame", b"R"
    )
    fr = p.parse(req)
    assert fr.kind == D.REQUEST and fr.msg_id == 42
    resp = bytes([D.RESPONSE]) + (42).to_bytes(2, "little") + _wrapper("", b"S")
    fr2 = p.parse(resp)
    assert fr2.kind == D.RESPONSE
    assert fr2.method == ".lq.FastTest.authGame"
    assert fr2.payload == {"_fqn": "lq.ResAuth", "_raw": b"S"}
    # pending 消费掉：再来一发同 id 静默（attach 前请求不可见——Akagi 语义）
    assert p.parse(resp) is None


def test_unknown_request_route_silent():
    p = D.LiqiParser(message_to_dict=_stub_td, routes={})
    fr = p.parse(bytes([D.REQUEST]) + b"\x01\x00" + _wrapper(".lq.Lobby.ping", b"x"))
    assert fr is None


def test_restore_action_skips_xor():
    import base64

    raw = b"\x08\x01\x12\x03abc"
    b64 = base64.b64encode(raw).decode()
    assert D.decode_action_bytes(b64, restore=True) == raw
    assert D.decode_action_bytes(b64, restore=False) != raw  # 活体路径过 XOR


def test_decode_restore_vs_live_roundtrip():
    # 活体：XOR 后 base64，decode 应还原
    import base64

    plain = b"\x0a\x05hello"
    enc = bytearray(plain)
    key = D._KEYS
    base = 23 ^ len(enc)
    for i in range(len(enc)):
        enc[i] ^= (base + 5 * i + key[i % 9]) & 0xFF
    assert D.decode_action_bytes(base64.b64encode(enc).decode()) == plain
