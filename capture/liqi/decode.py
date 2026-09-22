"""liqi 帧解析（decode-only，无 encode 能力——只读红线的结构性保证）。

descriptor 来源：scripts/compile_liqi_proto.py 把 reference/akagi-v3/bridge/
liqi.proto 编成 capture/liqi/_gen/liqi_pb2.py（生成物，不进 git）。
路由表：reference/akagi-v3/bridge/liqi.json（`{".lq.Svc.method": {req,resp}}`）。

每 WS 连接一个 LiqiParser：Response 只带 msg_id，方法名/类型必须从本连接的
pending map 找（Akagi flow.rs 同构）。
"""

from __future__ import annotations

import base64
import json
import pathlib
from collections.abc import Callable
from dataclasses import dataclass, field

# ---------------- wtf_decode（Akagi parser.rs::wtf_decode 逐位移植） ----------------
_KEYS = bytes([0x84, 0x5E, 0x4E, 0x42, 0x39, 0xA2, 0x1F, 0x60, 0x1C])


def wtf_decode(data: bytearray) -> None:
    base = 23 ^ len(data)
    for i in range(len(data)):
        data[i] ^= (base + 5 * i + _KEYS[i % len(_KEYS)]) & 0xFF


def decode_action_bytes(b64: str, *, restore: bool = False) -> bytes:
    """ActionPrototype.data -> protobuf bytes。GameRestore 内嵌动作是纯 base64
    （restore=True，跑 XOR 会损坏字节——Akagi decode_restore_action 注释）。"""
    data = bytearray(base64.b64decode(b64))
    if not restore:
        wtf_decode(data)
    return bytes(data)


# ---------------- Wrapper ----------------
@dataclass(frozen=True)
class Wrapper:
    name: str
    data: bytes


def decode_wrapper(buf: bytes) -> Wrapper:
    """Wrapper{string name=1; bytes data=2;} 手解（不为一个两字段消息依赖生成码）。"""
    name = ""
    data = b""
    i = 0
    while i < len(buf):
        key = buf[i]
        i += 1
        fn, wt = key >> 3, key & 7
        if wt != 2:
            raise ValueError(f"Wrapper 字段 {fn} wiretype {wt} 非 LEN")
        ln = 0
        shift = 0
        while True:  # varint
            b = buf[i]
            i += 1
            ln |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
        chunk = buf[i : i + ln]
        i += ln
        if fn == 1:
            name = chunk.decode("utf-8", "replace")
        elif fn == 2:
            data = chunk
    return Wrapper(name, data)


# ---------------- 帧级路由 ----------------
NOTIFY, REQUEST, RESPONSE = 1, 2, 3


@dataclass(frozen=True)
class Frame:
    kind: int                 # NOTIFY/REQUEST/RESPONSE
    method: str               # notify: "lq.NotifyX"；req/resp: ".lq.Svc.method"
    msg_id: int | None
    payload: dict             # 已解码消息 -> JSON dict


@dataclass
class LiqiParser:
    """WS 单连接有状态解析器。pool/ROUTES 注入：pool 为 protobuf descriptor pool
    （_gen 产物），ROUTES 为 liqi.json 的 rpc-map。

    message_to_dict(pool, fqn, data) 由调用侧提供（解耦生成码 import 时机）。
    """

    message_to_dict: Callable[[str, bytes], dict]
    routes: dict = field(default_factory=dict)
    _pending: dict = field(default_factory=dict)  # msg_id -> (method, resp_fqn)

    def decode_action(self, name: str, b64: str, *, restore: bool = False) -> dict:
        """动作名（裸 `ActionX` 或 `.lq.ActionX`）+ base64 → 动作 dict。

        restore=True 用于 GameRestore 内嵌动作（纯 base64 免 XOR）。
        对齐 parser.rs::decode_named_action 的名字归一化：多段名取首段+末段。
        """
        raw = decode_action_bytes(b64, restore=restore)
        parts = [p for p in name.split(".") if p]
        if not parts:
            raise ValueError(f"empty action name: {name!r}")
        fqn = f"{parts[0]}.{parts[-1]}" if len(parts) > 1 else f"lq.{parts[0]}"
        return self.message_to_dict(fqn, raw)

    def maybe_decode_action(self, payload: dict) -> dict:
        """parser.rs::maybe_decode_action 镜像：payload 形如 {name:str, data:str}
        （即 ActionPrototype notify）时把 data 二次解码为动作 dict。
        非该形状原样返回。解码失败抛错=丢帧（同 Rust Err 传播）。"""
        name = payload.get("name")
        b64 = payload.get("data")
        if not isinstance(name, str) or not isinstance(b64, str):
            return payload
        return {**payload, "data": self.decode_action(name, b64)}

    def parse(self, buf: bytes) -> Frame | None:
        if not buf:
            return None
        t = buf[0]
        if t == NOTIFY:
            w = decode_wrapper(buf[1:])
            parts = [p for p in w.name.split(".") if p]
            if len(parts) != 2:
                raise ValueError(f"notify 名应两段 lq.X，得 {w.name!r}")
            fqn = f"{parts[0]}.{parts[1]}"
            payload = self.message_to_dict(fqn, w.data)
            return Frame(NOTIFY, fqn, None, self.maybe_decode_action(payload))
        if t in (REQUEST, RESPONSE):
            if len(buf) < 3:
                raise ValueError("frame too short")
            msg_id = int.from_bytes(buf[1:3], "little")
            w = decode_wrapper(buf[3:])
            if t == REQUEST:
                spec = self.routes.get(w.name)
                if spec is None:
                    return None  # 无路由=非游戏 RPC，静默（Akagi Err 分支同语义）
                payload = self.message_to_dict(spec["req"].lstrip("."), w.data)
                self._pending[msg_id] = (w.name, spec["resp"].lstrip("."))
                return Frame(REQUEST, w.name, msg_id, payload)
            if w.name:
                raise ValueError(f"response wrapper 不应带 name，得 {w.name!r}")
            pair = self._pending.pop(msg_id, None)
            if pair is None:
                return None  # 请求非本连接所见（attach 前发生）——静默
            method, resp_fqn = pair
            return Frame(RESPONSE, method, msg_id, self.message_to_dict(resp_fqn, w.data))
        return None  # 未知 type byte：非协议帧


def load_routes(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
