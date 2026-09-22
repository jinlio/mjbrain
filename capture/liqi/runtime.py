"""真 descriptor 接线：编译后的 liqi 类型池 -> decode.LiqiParser。

用法：
    from capture.liqi import runtime
    p = runtime.make_parser()
    frame = p.parse(ws_bytes)     # -> Frame | None
"""

from __future__ import annotations

import pathlib

from google.protobuf import descriptor_pool, json_format, message_factory

from capture.liqi import decode

_ROUTES_PATH = (
    pathlib.Path(decode.__file__).parents[2] / "reference" / "akagi-v3" / "bridge" / "liqi.json"
)

_pool: descriptor_pool.DescriptorPool | None = None


def _desc_pool() -> descriptor_pool.DescriptorPool:
    global _pool
    if _pool is None:
        from capture.liqi._gen import liqi_pb2

        _pool = liqi_pb2.DESCRIPTOR.pool
    return _pool


def message_to_dict(fqn: str, data: bytes) -> dict:
    cls = message_factory.GetMessageClass(_desc_pool().FindMessageTypeByName(fqn))
    msg = cls()
    msg.ParseFromString(data)
    # 保留 proto 原字段名（snake_case），int64 出 int 而非 str——映射层语义更直白
    return json_format.MessageToDict(
        msg, preserving_proto_field_name=True, use_integers_for_enums=True
    )


_routes: dict | None = None


def routes() -> dict:
    global _routes
    if _routes is None:
        _routes = decode.load_routes(_ROUTES_PATH)
    return _routes


def make_parser() -> decode.LiqiParser:
    return decode.LiqiParser(message_to_dict=message_to_dict, routes=routes())


def make_stack():
    """只读全链整机：(LiQiParser, MajsoulState)。

    帧管道 `p.parse(ws_bytes)` → Frame；`st.dispatch(fr.msg_type, fr.method,
    fr.payload)` → mjai 事件列表。GameRestore 重放经注入的 decode_restore
    （免 XOR）走真描述符。per-WS 连接各配一套（msg_id pending-map 独立）。
    """
    from capture.mjai.state import MajsoulState

    p = make_parser()
    st = MajsoulState(
        decode_restore=lambda name, b64: p.decode_action(name, b64, restore=True)
    )
    return p, st
