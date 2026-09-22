#!/usr/bin/env python
"""M4 档1 实时推荐客户端：捕获 JSONL → mjai 事件流 → POST /v1/react → 打印。

读 scripts/run_capture.py 落盘的 JSONL（每行 {"dir":..,"b64":..}）：
  每行 base64 → capture.liqi runtime 的 parser.parse → Frame →
  MajsoulState.dispatch → mjai 事件累加；每当流有新事件，把**全量流**
  POST 给 advisor/server.py 的 /v1/react（服务端重放出决策窗），
  自家座位开窗时打印 B3 推荐 + top 候选。

前置：另开一个进程跑推荐服务
    python -m advisor.server            # 127.0.0.1:8765，启动即预热权重
和捕获
    python scripts/run_capture.py --url <雀魂页> --out data/raw/ms_frames/run1.jsonl

红线自查：本脚本只读 JSONL、只发本机 HTTP 建议请求；对雀魂客户端
零接触、零写入（动作注入禁止）。

--follow 尾随模式：文件边写边读，读到尾 sleep 重试；捕获进程追加帧即出推荐。
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request


def post_react(url: str, payload: dict, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(
        url + "/v1/react",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fmt_decision(d: dict) -> str:
    if d.get("window") is False:
        return f"座位{d['seat']}: 无决策窗"
    if "error" in d:
        return f"座位{d['seat']}: {d['error']}"
    top = "  ".join(f"{t['a']} {t['p'] * 100:.1f}%" for t in d.get("top", [])[:3])
    return f"[{time.strftime('%H:%M:%S')}] 座位{d['seat']} 候选{d['legal_n']} → {d['recommend']} | {top}"


def feed_lines(p, st, events: list[dict], lines) -> bool:
    """把若干 JSONL 行喂进 parser+state；返回事件流是否有增长（供调用侧节流）。"""
    grew = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            buf = base64.b64decode(rec["b64"])
        except Exception:  # noqa: BLE001,S112 —— 非协议行/截断行：静默跳过
            continue
        try:
            fr = p.parse(buf)
            if fr is None:
                continue
            evs = st.dispatch(fr.kind, fr.method, fr.payload)
        except Exception as ex:  # noqa: BLE001 —— 单帧解码失败=丢帧，不断链
            print(f"  ! 帧解码失败（跳过）: {type(ex).__name__}: {ex}", file=sys.stderr)
            continue
        if evs:
            events.extend(evs)
            grew = True
    return grew


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jsonl", required=True, type=pathlib.Path,
                    help="run_capture.py 落盘的帧 JSONL")
    ap.add_argument("--seat", type=int, default=-1,
                    help="座位号；缺省自动取状态机从 AuthGame 帧识别的自家座位")
    ap.add_argument("--url", default="http://127.0.0.1:8765",
                    help="advisor.server 地址（默认本机 8765）")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--follow", action="store_true",
                    help="尾随模式：读到文件尾不退出，等捕获追加")
    ap.add_argument("--poll", type=float, default=0.1, help="--follow 轮询秒")
    args = ap.parse_args(argv)

    while args.follow and not args.jsonl.exists():
        print(f"等 {args.jsonl} 出现…", file=sys.stderr)
        time.sleep(1.0)
    if not args.jsonl.exists():
        print(f"文件不存在: {args.jsonl}", file=sys.stderr)
        return 1

    from capture.liqi import runtime

    p, st = runtime.make_stack()
    events: list[dict] = []
    last_err = None
    with args.jsonl.open(encoding="utf-8") as fh:
        while True:
            lines = fh.readlines()
            if not lines:
                if not args.follow:
                    break
                time.sleep(args.poll)
                continue
            if not feed_lines(p, st, events, lines) and last_err is None:
                continue
            seat = args.seat if args.seat >= 0 else st.seat
            body = {"events": events, "top": args.top}
            if seat is not None:
                body["seat"] = int(seat)
            try:
                resp = post_react(args.url, body)
                last_err = None
            except (urllib.error.URLError, OSError) as ex:
                if last_err is None:
                    print(f"! 服务不可达（{args.url}）：{ex}；"
                          "确认 python -m advisor.server 在跑，持续重试", file=sys.stderr)
                    last_err = ex
                continue
            if "error" in resp:
                print(f"服务端：{resp['error']}（applied={resp.get('applied')}）",
                      file=sys.stderr)
                continue
            for d in resp.get("decisions", []):
                print(fmt_decision(d), flush=True)
    if events:
        print(f"共 {len(events)} 个 mjai 事件入流。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
