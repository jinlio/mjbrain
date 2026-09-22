"""M4 档1 · CDP 捕获跑手：拉起浏览器窗口 -> 订阅 WS 帧 -> 落 JSONL 原料。

用法（conda activate mjbrain）：
  python scripts/run_capture.py --url https://game.maj-soul.com/1/            # 雀魂 web
  python scripts/run_capture.py --url http://127.0.0.1:8123/ --headless       # 本地自测
  python scripts/run_capture.py --url ... --out data/raw/ms_frames/run1.jsonl

窗口打开后你正常登录/打牌；本脚本**只截帧不改任何东西**（liqi->mjai 解码是
下一步；现在先攒原料）。Ctrl-C 退出：保留浏览器进程（对局不断线，Akagi 语义）。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from capture.chromium import cdp_client, launch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", default="", help="帧 JSONL 落盘路径（默认临时目录）")
    ap.add_argument("--profile", default="", help="user-data-dir（默认临时目录）")
    ap.add_argument("--headless", action="store_true", help="无头（自测用；雀魂正式用有头）")
    ap.add_argument("--seconds", type=float, default=0, help="0=一直跑")
    args = ap.parse_args()

    out = pathlib.Path(args.out or tempfile.mkdtemp(prefix="ms_frames_"))
    out.parent.mkdir(parents=True, exist_ok=True)
    udd = pathlib.Path(args.profile or tempfile.mkdtemp(prefix="mjbrain-cdp-"))
    print(f"profile: {udd}  frames -> {out}")

    extra = ["--headless=new"] if args.headless else []
    proc, udd, port = launch.spawn_browser(args.url, user_data_dir=udd, extra_args=extra)
    print(f"browser pid={proc.pid} port={port}")
    # 我们总是显式传端口，直接 GET /json/version 即可（headless=new 不写
    # DevToolsActivePort 文件，read_devtools_port 只留给 --remote-debugging-port=0 场景）
    try:
        ws_url = launch.browser_ws_url(port)
    except TimeoutError as e:
        print("端点发现失败:", e)
        proc.terminate()
        return 1
    print("cdp:", ws_url)

    t0 = time.time()
    n = {"c": 0}
    with out.open("a", encoding="utf-8") as fh:

        def sink(direction: str, url: str, rid: str, data: bytes) -> None:
            n["c"] += 1
            fh.write(
                json.dumps(
                    {
                        "t": round(time.time() - t0, 3),
                        "dir": direction,
                        "page": url[:160],
                        "rid": rid,
                        "len": len(data),
                        "b64": base64.b64encode(data).decode(),
                    }
                )
                + "\n"
            )
            if n["c"] % 50 == 0:
                print(f"{n['c']} frames...", flush=True)

        async def run() -> None:
            stop = asyncio.Event()
            w = cdp_client.CdpWatcher(ws_url, sink)
            task = asyncio.create_task(w.run(stop))
            while not stop.is_set():
                await asyncio.sleep(0.2)
                if proc.poll() is not None:
                    print("浏览器进程退出，收工")
                    stop.set()
                if args.seconds and time.time() - t0 > args.seconds:
                    stop.set()
            await asyncio.wait_for(task, timeout=3)

        try:
            asyncio.run(run())
        except KeyboardInterrupt:
            pass
    print(f"total {n['c']} frames -> {out}")
    if proc.poll() is None and not args.headless:
        print("（浏览器保留运行中：只读截获已结束，对局不断线）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
