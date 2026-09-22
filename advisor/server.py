"""档1 推荐服务：POST /v1/react {events:[mjai...], seat?:int} → 当前开窗家的 B3 建议。

- 绑定 127.0.0.1：本机自用，零外暴露。
- 只读语义：输入捕获层的事件流，输出建议 JSON；服务不存在任何"代替玩家
  发动作"的路径（零动作注入红线）。
- 捕获侧（capture/）与训练/评估共用 brain.serialize 的同一 state_text。

启动：python -m advisor.server [--port 8765] [--ckpt DIR] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_DEFAULT_CKPT = "checkpoints/20260921T184331Z-rlcd-gate"

_adviser = None
# 可重入：do_POST 全程持锁（前向串行），_get_adviser 也在锁内取用
_lock = threading.RLock()


def _get_adviser(ckpt: str, device: str | None):
    global _adviser
    with _lock:
        if _adviser is None:
            from advisor.recommend import Adviser

            _adviser = Adviser(ckpt, device=device)
        return _adviser


def open_windows(env):
    """当前对所有座位可见的决策窗：pid -> [legal mjai 串]。"""
    out = {}
    for pid in range(env.num_players):
        try:
            ob = env.get_observation(pid)
        except Exception:  # noqa: BLE001,S112 —— 非行动帧无观测
            continue
        if ob is None:
            continue
        la = [str(a.to_mjai()) for a in ob.legal_actions()]
        if la:
            out[pid] = (ob, la)
    return out


def react(events: list[dict], seat: int | None, ckpt: str,
          device: str | None, top: int = 5) -> dict:
    from riichienv import GameRule, RiichiEnv

    env = RiichiEnv(game_mode="4p-red-half", rule=GameRule.default_mjsoul())
    applied = 0
    for ev in events:
        try:
            env.apply_event(ev)
            applied += 1
        except Exception as ex:  # noqa: BLE001 —— 停在可解释处并报给调用方
            return {"error": f"事件流在第 {applied} 行不可重放: {ex}",
                    "applied": applied}
    adv = _get_adviser(ckpt, device)
    wins = open_windows(env)
    seats = sorted(wins) if seat is None else [seat]
    out = {"applied": applied, "open_seats": seats, "decisions": []}
    for pid in seats:
        if pid not in wins:
            out["decisions"].append({"seat": pid, "window": False})
            continue
        ob, legal = wins[pid]
        try:
            ranked = adv.topk(ob, legal, top=top)
        except ValueError as ex:
            out["decisions"].append({"seat": pid, "error": str(ex)})
            continue
        out["decisions"].append({
            "seat": pid, "legal_n": len(legal),
            "recommend": ranked[0][0], "top": [{"a": a, "p": round(p, 4)}
                                               for a, p in ranked]})
    return out


class Handler(BaseHTTPRequestHandler):
    ckpt = _DEFAULT_CKPT
    device = None

    def log_message(self, *a) -> None:  # 静音默认 stderr 日志
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/v1/health":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/react":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            events = req["events"]
            seat = req.get("seat")
            top = int(req.get("top", 5))
        except Exception as ex:  # noqa: BLE001 —— 坏请求 400 而非崩服务
            self._send(400, {"error": f"请求体不合 JSON 或缺 events: {ex}"})
            return
        with _lock:
            self._send(200, react(events, seat, self.ckpt, self.device, top=top))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ckpt", default=_DEFAULT_CKPT)
    ap.add_argument("--device", default="cpu", help="cpu/cuda/mps")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    Handler.ckpt = a.ckpt
    Handler.device = a.device
    _get_adviser(a.ckpt, a.device)  # 启动即加载权重，首请求不卡
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[advisor] http://{a.host}:{a.port}/v1/react (ckpt={a.ckpt})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
