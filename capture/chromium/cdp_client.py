"""CDP 客户端：browser 级连接 + per-page Network 域订阅 + WS 帧解码。

移植改写自 Akagi v3 src/capture/chromium/cdp.rs（Apache-2.0）。
为什么 per-page 而非 browser 级订阅：Akagi 用 chromiumoxide 时 page 级事件
不上送 Browser::event_listener；我们手搓协议层可自验——统一走
Target.attachToTarget(flatten=True) 后事件带 sessionId，按 session 分发。

**只读**：仅 Network.enable / getResponseBody 一类观察命令；
无 Input.*、无 Fetch.*、不构造任何发往页面的请求。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

import websockets

log = logging.getLogger("mjbrain.capture.cdp")

# 帧回调签名：(direction "recv"/"sent", page_url, request_id, bytes)
FrameSink = Callable[[str, str, str, bytes], None]


@dataclass
class PageRef:
    target_id: str
    session_id: str
    url: str


class CdpWatcher:
    """附着 browser 端点，轮询页面集合 diff，每页 Network.enable 并转发 WS 帧。"""

    def __init__(self, browser_ws_url: str, sink: FrameSink, poll: float = 1.0):
        self._url = browser_ws_url
        self._sink = sink
        self._poll = poll
        self._ws: websockets.ClientConnection | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._pages: dict[str, PageRef] = {}  # target_id -> PageRef
        self._req_page: dict[str, str] = {}   # webSocket requestId -> target_id

    async def _send(self, method: str, params: dict | None = None,
                    session_id: str | None = None) -> dict:
        assert self._ws is not None
        self._next_id += 1
        mid = self._next_id
        msg: dict = {"id": mid, "method": method, "params": params or {}}
        if session_id:
            msg["sessionId"] = session_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[mid] = fut
        try:  # 超时/发送失败都要回销 pending，否则 future 逐次泄漏
            await self._ws.send(json.dumps(msg))
            return await asyncio.wait_for(fut, timeout=15)
        finally:
            self._pending.pop(mid, None)

    async def run(self, stop: asyncio.Event | None = None) -> None:
        self._ws = await websockets.connect(self._url, max_size=64 * 2**20)
        reader = asyncio.create_task(self._read_loop())
        try:
            while not (stop and stop.is_set()):
                await self._diff_pages()
                await asyncio.sleep(self._poll)
        finally:
            reader.cancel()
            await self._ws.close()

    async def _diff_pages(self) -> None:
        assert self._ws is not None
        res = await self._send("Target.getTargets")
        live = {
            t["targetId"]: t.get("url", "")
            for t in res["targetInfos"]
            if t["type"] == "page"
        }
        for tid, url in live.items():
            if tid in self._pages:
                self._pages[tid].url = url  # 同 tab 导航后刷新 url
                continue
            a = await self._send("Target.attachToTarget", {"targetId": tid, "flatten": True})
            sid = a["sessionId"]
            await self._send("Network.enable", {}, session_id=sid)
            self._pages[tid] = PageRef(tid, sid, url)
            log.info("attach page %s (%s)", tid[:8], url[:80])
        for tid in [t for t in self._pages if t not in live]:
            del self._pages[tid]  # 订阅随 session 消亡，无需清理

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                # 单条消息容错：坏 JSON/坏帧只丢这一条。曾把 try 架在整循环外，
                # 一帧畸形 payloadData 就掀翻读循环 → 捕获永久静默停摆
                # （连接级异常由 async for 自身抛出，照常终止，交上层重连）
                try:
                    msg = json.loads(raw)
                    if "id" in msg:
                        fut = self._pending.pop(msg["id"], None)
                        if fut and not fut.done():
                            fut.set_result(msg.get("result", {}))
                        continue
                    await self._on_event(msg)
                except Exception:
                    log.exception("cdp 单条消息处理失败（跳过，读循环存活）")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("cdp read loop died")

    async def _on_event(self, msg: dict) -> None:
        m = msg.get("method", "")
        if not m.startswith("Network.webSocket"):
            return
        sid = msg.get("sessionId", "")
        page = next((p for p in self._pages.values() if p.session_id == sid), None)
        if page is None:
            return
        p = msg.get("params") or {}  # 缺 params 的事件按空处理，不杀读循环
        rid = p.get("requestId", "")
        if m == "Network.webSocketCreated":
            self._req_page[rid] = page.target_id
            return
        if m == "Network.webSocketClosed":
            self._req_page.pop(rid, None)
            return
        if m in ("Network.webSocketFrameReceived", "Network.webSocketFrameSent"):
            direction = "recv" if m.endswith("Received") else "sent"
            payload = p.get("response", {})
            data = decode_frame_payload(payload)
            if data is not None:
                self._sink(direction, page.url, rid, data)


def decode_frame_payload(resp: dict) -> bytes | None:
    """CDP WebSocketFrameResponse -> bytes。opcode 2=二进制(payloadData 为 base64)，
    1=文本(UTF-8 直传)；continuation(0)/close(8)/ping(9)/pong(10) 丢（Akagi 同策略）。"""
    opcode = resp.get("opcode", -1)
    pd = resp.get("payloadData", "")
    if opcode == 2:
        try:  # validate=True：默认模式会静默丢弃非字母表字符（"!!!"→b''），
            # 畸形帧就伪装成一条合法的空事件流。binascii.Error 是 ValueError 子类
            return base64.b64decode(pd, validate=True)
        except ValueError:
            return None
    if opcode == 1:
        return pd.encode("utf-8", "replace")
    return None
