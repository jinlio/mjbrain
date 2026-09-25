"""跨平台一键起档1 demo：advisor → 浏览器捕获 → live 推荐 → HUD 浮窗。

Windows 的 run_demo.bat / Mac 终端跑同一套（本脚本只依赖标准库）：

  python scripts/run_demo.py [--url https://game.maj-soul.com/1/] [--seat -1]
                             [--ckpt DIR] [--dry] [--no-hud]
                             [--window-size 1600x900] ...

四件事各归其位（子进程输出加前缀转印到本控制台）：
- advisor.server 常驻本机 8765（--device 自动：torch 可用 MPS/CUDA 就用，否则 CPU）；
- run_capture：持久 profile（登录保留）+ --app 独立窗 1600x900（默认，可覆盖），
  已在跑的浏览器实例自动续连（try_attach）；
- live_from_capture --follow：帧 → /v1/react → 终端 + `<frames>.advise.jsonl`；
- hud.float --wait：尾随该 JSONL 的悬浮小窗（不注入页面）。

权重缺省按链探测：checkpoints/m3-final-0.7410 → dist/rescue 原盘 →
m3-orig-0.7098 → rlcd-gate（--ckpt 可显式指定）。

Ctrl-C：收掉子进程；**浏览器保留**（对局不断线，下次运行续连）。
帧文件轮转 data/raw/ms_frames/run<N>.jsonl（与历史场次不串号）。
--dry 只做预检（ckpt/依赖/浏览器/轮转号），什么都不拉起。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://game.maj-soul.com/1/"
FRAMES_DIR = pathlib.Path("data/raw/ms_frames")
ADVISOR_URL = "http://127.0.0.1:8765"
ADV_MAX_RESTARTS = 5
CKPT_CHAIN = ("checkpoints/m3-final-0.7410",
              "dist/rescue-20260923/extracted/checkpoints",
              "checkpoints/m3-orig-0.7098",
              "checkpoints/20260921T184331Z-rlcd-gate")


def resolve_ckpt(explicit: str | None) -> pathlib.Path | None:
    """--ckpt 显式路径优先；否则按链找第一个含 model.safetensors 的目录。"""
    if explicit:
        p = pathlib.Path(explicit)
        return p if p.is_absolute() else ROOT / p
    for rel in CKPT_CHAIN:
        d = ROOT / rel
        if (d / "model.safetensors").exists():
            return d
    return None


def next_frames_path(frames_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """轮转到没被占用的 run<N>.jsonl。返回 (frames, advise)。"""
    frames_dir.mkdir(parents=True, exist_ok=True)
    used = [int(f.stem[3:]) for f in frames_dir.glob("run*.jsonl")
            if f.stem[3:].isdigit()]
    n = (max(used) + 1) if used else 1
    frames = frames_dir / f"run{n}.jsonl"
    return frames, frames.with_name(frames.stem + ".advise.jsonl")


def detect_device() -> str:
    """推荐服务设备：能 import torch 且有 MPS/CUDA 就用，否则 CPU。"""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "cpu"
    # getattr 守卫：老/非 mac 的 torch 构建没有 backends.mps 属性，直取 AttributeError
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_cmds(args, ckpt: pathlib.Path, frames: pathlib.Path,
               advise: pathlib.Path, device: str) -> dict[str, list[str]]:
    """四子进程的命令行（纯函数：--dry 与单测共用）。"""
    py = sys.executable
    capture = [py, str(ROOT / "scripts" / "run_capture.py"),
               "--url", args.url, "--out", str(frames)]
    if args.window_size:
        capture += ["--window-size", args.window_size]
    if args.window_position:
        capture += ["--window-position", args.window_position]
    if args.no_app:
        capture.append("--no-app")
    live = [py, str(ROOT / "scripts" / "live_from_capture.py"),
            "--jsonl", str(frames), "--follow", "--seat", str(args.seat)]
    hud = [py, "-m", "hud.float", "--advise", str(advise), "--wait"]
    adv = [py, "-m", "advisor.server", "--device", device, "--ckpt", str(ckpt),
           "--top", str(args.top)]
    return {"advisor": adv, "capture": capture, "live": live, "hud": hud}


def preflight(args, ckpt: pathlib.Path | None,
              frames: pathlib.Path) -> list[str]:
    """--dry 检查清单；返回逐行报告（不抛异常）。"""
    rep = [f"ckpt: {ckpt} {'✓' if ckpt and ckpt.exists() else '✗ 没找到（Release 解压或 --ckpt）'}"]
    for mod in ("riichienv", "websockets"):
        try:
            __import__(mod)
            rep.append(f"python 依赖: {mod} ✓")
        except ImportError:
            rep.append(f"python 依赖: {mod} ✗（pip install {mod}）")
    try:
        import tkinter  # noqa: F401

        rep.append("tkinter(HUD): ✓" if not args.no_hud else "tkinter(HUD): 本次不用")
    except ImportError:
        rep.append("tkinter(HUD): ✗ conda 环境缺 python-tk（--no-hud 可跳过）")
    try:
        from capture.chromium import launch

        rep.append(f"浏览器: ✓ {launch.find_browser()}")
    except Exception as ex:  # noqa: BLE001
        rep.append(f"浏览器: ✗ {ex}")
    rep.append(f"本轮帧文件: {frames.relative_to(ROOT)}")
    rep.append(f"advisor 设备: {args.device or detect_device()}")
    return rep


def wait_healthy(url: str, proc: subprocess.Popen, timeout: float = 120.0) -> bool:
    """轮询 advisor /v1/health；子进程提前退出判失败。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url + "/v1/health", timeout=2) as r:
                if json.loads(r.read().decode()).get("ok"):
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(0.5)
    return False


def _pump(name: str, proc: subprocess.Popen) -> None:
    """子进程 stdout/stderr 逐行加前缀转印（统一 utf-8，避免 Windows 乱码）。"""
    try:
        assert proc.stdout is not None
        for ln in proc.stdout:
            print(f"[{name}] {ln.rstrip()}")
    except (OSError, ValueError):  # 管道随 terminate 关闭
        pass


def spawn(cmds: dict[str, list[str]]) -> dict[str, subprocess.Popen]:
    """起子进程（管道 utf-8）。收工时全部 terminate——浏览器是孙进程
    （run_capture 所拉），terminate 不到它，保留语义自然成立。"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    procs: dict[str, subprocess.Popen] = {}
    for name, cmd in cmds.items():
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", env=env)
        procs[name] = p
        threading.Thread(target=_pump, args=(name, p), daemon=True).start()
    return procs


def main(argv=None) -> int:
    # stdout 重定向到管道/文件时按本地编码(cp936/cp1252)写：✓/✗ 等字符会
    # UnicodeEncodeError 掀掉整跑。errors=replace 只兜重定向，终端行为不变
    import sys as _sys
    for _s in (_sys.stdout, _sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--seat", type=int, default=-1,
                    help="-1=自动取 AuthGame 识别的自家座位")
    ap.add_argument("--ckpt", default=None, help="权重目录（缺省按链探测）")
    ap.add_argument("--device", default="", help="advisor --device（空=自动）")
    ap.add_argument("--top", type=int, default=5, help="advisor top-k")
    ap.add_argument("--window-size", default="1600x900")
    ap.add_argument("--window-position", default="")
    ap.add_argument("--no-app", action="store_true")
    ap.add_argument("--no-hud", action="store_true", help="不起悬浮窗（只看终端）")
    ap.add_argument("--dry", action="store_true", help="只预检不拉起")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    ckpt = resolve_ckpt(args.ckpt)
    frames, advise = next_frames_path(ROOT / FRAMES_DIR)
    if args.dry:
        for ln in preflight(args, ckpt, frames):
            print(ln)
        return 0
    if ckpt is None or not ckpt.exists():
        print("没找到权重：把 Release zip 解压到 checkpoints/m3-final-0.7410/ "
              "或 --ckpt 指定。", file=sys.stderr)
        return 1

    device = args.device or detect_device()
    cmds = build_cmds(args, ckpt, frames, advise, device)
    if args.no_hud:
        cmds.pop("hud")
    print(f"[run_demo] advisor={device}  ckpt={ckpt}\n[run_demo] 帧记录 -> {frames}")

    procs: dict[str, subprocess.Popen] = {}
    try:
        adv_cmds = {"advisor": cmds.pop("advisor")}
        procs.update(spawn(adv_cmds))
        if not wait_healthy(ADVISOR_URL, procs["advisor"]):
            print("! advisor 未就绪（上面 [advisor] 输出即原因：ckpt/端口？）")
            return 1
        procs.update(spawn(cmds))
        print("[run_demo] 就绪：浏览器里登录/打牌即可；Ctrl-C 收工（浏览器保留）。")
        adv_cmd = adv_cmds["advisor"]
        adv_restarts = 0
        while procs:
            time.sleep(0.5)
            for name in list(procs):
                p = procs[name]
                if p.poll() is not None:
                    print(f"[run_demo] {name} 退出（code={p.returncode}）")
                    procs.pop(name)
                    if name == "advisor" and adv_restarts < ADV_MAX_RESTARTS:
                        # advisor 是链的基石：死了≠收工，原地重拉。live 自带
                        # 持续重试，复活即续供（权重重载 ~20s，牌局照常进行）
                        adv_restarts += 1
                        print(f"[run_demo] ! advisor 死亡，自动重拉"
                              f"（第 {adv_restarts}/{ADV_MAX_RESTARTS} 次）", flush=True)
                        procs["advisor"] = spawn({"advisor": adv_cmd})["advisor"]
                        if wait_healthy(ADVISOR_URL, procs["advisor"]):
                            print("[run_demo] advisor 已复活，推荐续供", flush=True)
                        else:
                            print("[run_demo] advisor 本次重拉失败（见 [advisor] 输出）",
                                  flush=True)
    except KeyboardInterrupt:
        print("\n[run_demo] 收工（浏览器保留运行，对局不断线；下次运行自动续连）")
    finally:
        for name, p in procs.items():
            if p.poll() is None and name != "capture":
                p.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
