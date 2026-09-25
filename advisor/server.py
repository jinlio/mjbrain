"""档1 推荐服务：POST /v1/react {events:[mjai...], seat?:int} → 当前开窗家的 B3 建议。

- 绑定 127.0.0.1：本机自用，零外暴露。
- 只读语义：输入捕获层的事件流，输出建议 JSON；服务不存在任何"代替玩家
  发动作"的路径（零动作注入红线）。
- 捕获侧（capture/）与训练/评估共用 brain.serialize 的同一 state_text。
- 每个决策附带确定性牌况提示 hint（向听/待牌/宝牌，advisor.hints）——
  加分项：算不出置 None，绝不影响推荐主链。

启动：python -m advisor.server [--port 8765] [--ckpt DIR] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from advisor import hints

_DEFAULT_CKPT = "checkpoints/m3-final-0.7410"

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


def _has_reach(legal: list[str]) -> bool:
    for s in legal:
        try:
            if json.loads(s).get("type") == "reach":
                return True
        except ValueError:  # noqa: PERF203 —— 非 JSON 串（不该出现）当作无立直
            continue
    return False


def _reach_declare_window(events: list[dict], seat: int, adv, top: int) -> dict | None:
    """立直两段式的第二段：替该座补 apply 一次 reach，取"宣言牌"窗再问模型。

    riichienv 的 reach 是两步：先一个裸 reach 声明，再开一个只剩"保听打牌"的窗
    （实测：主窗 15 项=dahai×14+reach；声明窗只剩保听的 1~3 项）。真实对局的
    训练语料同构（reach 窗后紧跟宣言打牌窗，标的就是宣言牌），模型见过这类窗，
    故本段纯推理/显示增强，不动权重。拿不到宣言窗（异常/无窗）→ None，主建议
    不受影响；代价是每个立直可选窗多一次前向（CPU ~0.4s），立直窗本身很稀疏。
    """
    from riichienv import GameRule, RiichiEnv

    env = RiichiEnv(game_mode="4p-red-half", rule=GameRule.default_mjsoul())
    try:
        for ev in events:
            env.apply_event(ev)
        env.apply_event({"actor": seat, "type": "reach"})
        ob = env.get_observation(seat)
    except Exception:  # noqa: BLE001,S110 —— 宣言窗是加分项，拿不到就省去
        return None
    if ob is None:
        return None
    legal = [str(a.to_mjai()) for a in ob.legal_actions()]
    if not legal:
        return None
    ranked = adv.topk(ob, legal, top=top)
    return {"legal_n": len(legal), "recommend": ranked[0][0],
            "top": [{"a": a, "p": round(p, 4)} for a, p in ranked]}


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
        dec = {
            "seat": pid, "legal_n": len(legal),
            "recommend": ranked[0][0], "top": [{"a": a, "p": round(p, 4)}
                                               for a, p in ranked],
            "hint": hints.hint_from_obs(ob)}  # None=提示不可得，不影响推荐
        if _has_reach(legal):  # 立直可选 → 附带"若立直，宣言牌切哪张"
            fw = _reach_declare_window(events, pid, adv, top)
            if fw is not None:
                dec["reach"] = fw
        out["decisions"].append(dec)
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
        try:
            with _lock:
                result = react(events, seat, self.ckpt, self.device, top=top)
        except BaseException as ex:  # noqa: BLE001 —— 内核炸了也只回 500：
            # 单请求永不带崩服务（torch/内存类 RuntimeError 不是 ValueError
            # 那层能兜住的，2026-09-25 重连时段 advisor 静默死亡复盘）
            import traceback
            traceback.print_exc()
            try:
                self._send(500, {"error": f"react 内核异常: {type(ex).__name__}: {ex}"})
            except Exception:  # noqa: BLE001 —— 客户端已断开，无妨
                pass
            return
        self._send(200, result)


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
