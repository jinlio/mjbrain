"""CDP 只读捕获 · 浏览器 spawn 与端点发现（移植改写自 Akagi v3
src/capture/chromium/{launch,profile}.rs，Apache-2.0，见 LICENSES.md）。

零注入红线：本模块只拉起浏览器并发现调试端点，不发送任何 CDP 指令去
模拟输入（Input.*）或改写请求（Fetch.*）；用户在窗口里亲自操作。
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

# Akagi launch.rs 同款"少弹窗"flag；刻意不含任何 stealth/自动化伪装。
BASE_ARGS = [
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-sync",
    "--disable-features=Translate",
]


def find_browser() -> pathlib.Path:
    """探测系统浏览器（Chrome -> Edge），找不到则 RuntimeError。"""
    if sys.platform == "win32":
        cands = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
    elif sys.platform == "darwin":
        cands = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    else:  # linux：包名/发行版路径各异，硬编码候选 + PATH 兜底
        cands = [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/snap/bin/chromium",
        ]
    for c in cands:
        p = pathlib.Path(c)
        if p.exists():
            return p
    # 候选表没命中再按裸命令查 PATH（覆盖自装/非标准路径，win 上也有意义）
    for name in ("chrome", "google-chrome", "chromium", "msedge"):
        w = shutil.which(name)
        if w:
            return pathlib.Path(w)
    raise RuntimeError("找不到 Chrome/Edge：装一个或在 cands 里加路径")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def default_user_data_dir() -> pathlib.Path:
    """默认持久 profile：登录态跨运行/跨重启保留（雀魂 cookie 落在这里）。
    刻意不提供"自动清理"路径——删这个目录等于丢登录，只能人工决定。"""
    return pathlib.Path.home() / ".mjbrain" / "browser-profile"


def try_attach(profile: pathlib.Path) -> str | None:
    """该 profile 的浏览器若已在运行（上次 Ctrl-C 特意保留），返回其 CDP ws 地址。

    只读发现：读 profile/DevToolsActivePort 的端口 + GET /json/version 探活；
    不发任何 Input.*/Fetch.* 指令。任何一步失败都安静返回 None → 调用方走 spawn。
    前提：有头 Chrome 带 --remote-debugging-port 启动时会写该文件；headless=new
    不写（run_capture 的自测分支），attach 自然拿不到。
    """
    try:
        line = (profile / "DevToolsActivePort").read_text(encoding="utf-8")
        port = int(line.splitlines()[0].strip())
    except (OSError, ValueError, IndexError):
        return None
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/json/version", timeout=1.5
        ) as r:
            return json.load(r)["webSocketDebuggerUrl"]
    except Exception:  # noqa: BLE001 —— 端口死了/文件是陈旧残留都算"没在运行"
        return None


def spawn_browser(
    url: str,
    *,
    user_data_dir: str | os.PathLike,
    exe: pathlib.Path | None = None,
    port: int | None = None,
    extra_args: list[str] | None = None,
    window_size: tuple[int, int] | None = None,
    window_position: tuple[int, int] | None = None,
    app: bool = False,
) -> tuple[subprocess.Popen, pathlib.Path, int]:
    """拉起带 CDP 的浏览器进程。返回 (proc, profile_dir, debug_port)。

    独立 user-data-dir 是关键：不碰用户主 profile，且"杀遗留实例"
    （reclaim）可按该目录精确匹配（Akagi Windows 方案）。

    window_size=(w,h) -> --window-size；app=True 用 --app=<url> 独立窗
    （无标签栏/地址栏，客户区即页面，游戏画布比例=窗口比例）。
    """
    exe = exe or find_browser()
    udd = pathlib.Path(user_data_dir)
    udd.mkdir(parents=True, exist_ok=True)
    port = port or free_port()
    cmd = [
        str(exe),
        f"--user-data-dir={udd}",
        f"--remote-debugging-port={port}",
        *BASE_ARGS,
    ]
    if window_size:
        cmd.append(f"--window-size={window_size[0]},{window_size[1]}")
    if window_position:
        cmd.append(f"--window-position={window_position[0]},{window_position[1]}")
    cmd += list(extra_args or [])
    cmd.append(f"--app={url}" if app else url)
    proc = subprocess.Popen(cmd)
    return proc, udd, port


def read_devtools_port(profile: pathlib.Path, timeout: float = 20.0) -> int:
    """读 profile/DevToolsActivePort 第一行（端口），轮询到出现为止。"""
    f = profile / "DevToolsActivePort"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            line = f.read_text(encoding="utf-8").splitlines()[0].strip()
            if line:
                return int(line)
        except (OSError, ValueError, IndexError):
            # 竞态即本函数在等的事：文件半写（splitlines()[0] 越界）、
            # Windows 下 Chrome 持写锁（PermissionError）——都不能掀桌。
            # 与 try_attach 的捕获面对齐
            time.sleep(0.25)
    raise TimeoutError(f"{f} 未出现：浏览器可能没带 --remote-debugging-port 启动")


def browser_ws_url(port: int, timeout: float = 20.0) -> str:
    """GET /json/version -> webSocketDebuggerUrl（ws://127.0.0.1:P/devtools/browser/<uuid>）。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/version", timeout=2
            ) as r:
                return json.load(r)["webSocketDebuggerUrl"]
        except Exception:  # noqa: BLE001 — 端点未就绪期间连接被拒是常态
            time.sleep(0.25)
    raise TimeoutError(f"127.0.0.1:{port} 的 CDP 端点 {timeout}s 内未就绪")


def reclaim_stale(profile: pathlib.Path) -> list[int]:
    """杀掉命令行里带同一 user-data-dir 的遗留浏览器主进程（排除 --type= 子进程）。
    Windows-only 实现（M4 主平台）；非 Windows 安静返回空（宁漏不误杀），
    返回被杀 PID。"""
    if sys.platform != "win32":
        return []

    import subprocess

    needle = f"--user-data-dir={profile}".lower()
    ps = (
        "Get-CimInstance Win32_Process | Where-Object { "
        "$_.CommandLine -and $_.CommandLine.ToLower().Contains('"
        + needle.replace("'", "''")
        + "') -and -not $_.CommandLine.Contains('--type=') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $_.ProcessId }"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]
