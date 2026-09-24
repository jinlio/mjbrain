"""M4 档1 · CDP 捕获跑手：拉起浏览器窗口 -> 订阅 WS 帧 -> 落 JSONL 原料。

用法（conda activate mjbrain）：
  python scripts/run_capture.py --url https://game.maj-soul.com/1/            # 雀魂 web
  python scripts/run_capture.py --url http://127.0.0.1:8123/ --headless       # 本地自测
  python scripts/run_capture.py --url ... --out data/raw/ms_frames/run1.jsonl

窗口打开后你正常登录/打牌；本脚本**只截帧不改任何东西**（liqi->mjai 解码是
下一步；现在先攒原料）。Ctrl-C 退出：保留浏览器进程（对局不断线，Akagi 语义），
下次运行经 try_attach 直接续连该实例。

登录状态：--profile 缺省用持久目录 ~/.mjbrain/browser-profile（不再每次新建
临时目录），雀魂登录 cookie 跨运行保留；profile 坏了手动删目录重新登录。
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

DEFAULT_WINDOW_SIZE = "1600x900"  # 16:9，雀魂画布按窗口等比缩放，基本无黑边


def _parse_size(s: str) -> tuple[int, int] | None:
    """'1600x900'（或 1600,900）-> (w,h)；空串=不指定交给浏览器。"""
    s = s.strip().lower().replace("x", ",")
    if not s:
        return None
    a, _, b = s.partition(",")
    try:
        w, h = int(a), int(b)
    except ValueError:
        raise SystemExit(f"--window-size 不合法: {s!r}（要 WxH，如 1600x900）")
    if not (200 <= w <= 7680 and 200 <= h <= 4320):
        raise SystemExit(f"--window-size 超出合理范围: {s!r}")
    return w, h


def _parse_pos(s: str) -> tuple[int, int] | None:
    """'80,60' -> (x,y)；空串=不指定。"""
    s = s.strip().replace("x", ",")
    if not s:
        return None
    a, _, b = s.partition(",")
    try:
        return int(a), int(b)
    except ValueError:
        raise SystemExit(f"--window-position 不合法: {s!r}（要 X,Y，如 80,60）")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", default="", help="帧 JSONL 落盘路径（默认临时目录）")
    ap.add_argument("--profile", default="",
                    help="user-data-dir（默认持久目录 ~/.mjbrain/browser-profile，"
                         "登录态跨运行保留；传一次性路径则不共享）")
    ap.add_argument("--headless", action="store_true", help="无头（自测用；雀魂正式用有头）")
    ap.add_argument("--seconds", type=float, default=0, help="0=一直跑")
    ap.add_argument("--window-size", default=DEFAULT_WINDOW_SIZE,
                    help=f"浏览器窗口 WxH（默认 {DEFAULT_WINDOW_SIZE}=16:9，"
                         "贴齐游戏画布比例；传空串用浏览器默认）")
    ap.add_argument("--window-position", default="",
                    help="窗口左上角 X,Y（缺省交给系统）")
    ap.add_argument("--no-app", action="store_true",
                    help="普通标签页模式（缺省用 --app 独立窗：无标签栏/地址栏）")
    args = ap.parse_args()

    win_size = _parse_size(args.window_size)
    win_pos = _parse_pos(args.window_position)

    out = pathlib.Path(args.out or tempfile.mkdtemp(prefix="ms_frames_"))
    out.parent.mkdir(parents=True, exist_ok=True)
    udd = pathlib.Path(args.profile) if args.profile else launch.default_user_data_dir()
    udd.mkdir(parents=True, exist_ok=True)
    print(f"profile: {udd}（持久，登录保留）  frames -> {out}")

    extra = ["--headless=new"] if args.headless else []
    proc = None
    # 已在运行的同 profile 实例直接续连（Ctrl-C 特意保留浏览器就是为这一步）；
    # attach 不中才 spawn。headless=new 不写 DevToolsActivePort，attach 自然落空。
    ws_url = None if args.headless else launch.try_attach(udd)
    if ws_url:
        print(f"复用已运行浏览器（profile={udd}），不再拉起新进程")
    else:
        # 我们总是显式传端口，直接 GET /json/version 即可（headless=new 不写
        # DevToolsActivePort 文件，read_devtools_port 只留给 --remote-debugging-port=0 场景）
        try:
            proc, udd, port = launch.spawn_browser(
                args.url, user_data_dir=udd, extra_args=extra,
                window_size=win_size, window_position=win_pos,
                app=not args.no_app and not args.headless)
            print(f"browser pid={proc.pid} port={port}")
            ws_url = launch.browser_ws_url(port)
        except TimeoutError as e:
            print("端点发现失败:", e)
            if proc is not None:
                proc.terminate()
            return 1
    print("cdp:", ws_url)

    t0 = time.time()
    n = {"c": 0}
    browser_alive = True  # attach 路径没有 proc 可查，别在窗口已关后谎报"保留运行"
    # 行缓冲：实时客户端（live_from_capture --follow）尾随读本文件，默认 8KB 块
    # 缓冲会让帧攒满一整块才落盘（实测滞后 5~96s），建议响应随之"极慢"
    with out.open("a", encoding="utf-8", buffering=1) as fh:

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
            nonlocal browser_alive
            stop = asyncio.Event()
            w = cdp_client.CdpWatcher(ws_url, sink)
            task = asyncio.create_task(w.run(stop))
            while not stop.is_set():
                await asyncio.sleep(0.2)
                if task.done():
                    # attach 路径没有 proc 可查：浏览器窗口被关=CDP 连接先死，收工
                    exc = task.exception()
                    if exc is not None:
                        browser_alive = False
                        print(f"CDP 连接断开（浏览器已关？）：{type(exc).__name__}: {exc}")
                    stop.set()
                    break
                if proc is not None and proc.poll() is not None:
                    browser_alive = False
                    print("浏览器进程退出，收工")
                    stop.set()
                    break
                if args.seconds and time.time() - t0 > args.seconds:
                    stop.set()
            if not task.done():
                await asyncio.wait_for(task, timeout=3)

        try:
            asyncio.run(run())
        except KeyboardInterrupt:
            pass
    print(f"total {n['c']} frames -> {out}")
    if not args.headless and browser_alive and (proc is None or proc.poll() is None):
        print("（浏览器保留运行中：只读截获已结束，对局不断线；下次运行自动续连）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
