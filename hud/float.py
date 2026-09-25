"""OS 悬浮 HUD：Tkinter 无边框置顶小窗，浮在游戏窗口之上（不注入页面、不占布局）。

数据源是现成的推荐记录 `<frames>.advise.jsonl`（live_from_capture.py 行缓冲
随写随刷）——本窗**零网络、不碰 advisor 服务**（不与推理抢锁）、更不碰雀魂
页面（红线：无注入）。每条新记录到达即刷新渲染。

窗口行为：
- **可拖动 + 记住位置**：左键按住整窗移动；`hud.json` 存绝对位置，下次启动复原。
  （曾实现"跟随浏览器窗口移动"，2026-09-24 真机验收体验不佳，已移除。）
- **锁定/解锁**：右键菜单切换（或按 L 键）。锁定=不可拖，防误触。
- **停更降级**：>90s 无新记录 → 置灰 + "无新事件 N 分钟"行（局后不再把
  过期建议当现况）；下一条记录到达自动复原。

形态与差异：
- 全平台 overrideredirect 无边框 + -topmost 置顶 + -alpha 半透明。
  （macOS：曾在 Tk 8.6.13/conda-forge 实测 overrideredirect 帧高==内容高，
  真正无边框；`::tk::unsupported::MacWindowStyle plain none` 反而只改内部
  记账、原生标题栏不动，勿再走回头路。个别老 Tk 版若 overrideredirect
  不生效，退回带标题栏窗，行为不坏。）
- 没有 Tk（conda 缺包）→ 明确报"装 tkinter"并退出，不静默。

启动：python -m hud.float --advise data/raw/ms_frames/run1.advise.jsonl
（--wait 等文件出现；--no-topmost/--alpha/--geometry 覆盖本次启动设置，
窗口拖动或退出后仍会写入持久设置 ~/.mjbrain/hud.json）
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

DEFAULT_FONT_SIZE = 12
DEFAULT_ALPHA = 0.85

# 停更降级：超过宽限期没有新记录 → 置灰 + 追加"无新事件 N 分钟"一行。
# HUD 是被动尾随者，局后文件停止增长，不降级就会把几小时前的过期建议
# 当现况挂着（2026-09-25 午间局后 17:07 一条停一下午的复盘）。
STALE_GRACE_S = 90.0
FG_FRESH = "#dce8f4"
FG_STALE = "#5b6672"


# ---------- 渲染（纯函数，无 tk：可单测） ----------

def format_lines(rec: dict | None) -> list[str]:
    """advise JSONL 一条记录 → HUD 显示行（全中文，字段已在落盘前映射好）。"""
    if not isinstance(rec, dict) or not rec:
        return ["等待建议…"]
    seat = rec.get("seat")
    ts = str(rec.get("ts") or "")[11:19]
    head = f"座位{seat}" if seat is not None else "座位?"
    if ts:
        head += f" {ts}"
    lines = [head]
    if rec.get("zh_recommend"):
        lines.append(f"→ {rec['zh_recommend']}")
        tops = [f"{a} {p * 100:.0f}%"
                for a, p in (rec.get("zh_top") or [])[:3]]
        if tops:
            lines.append("  ".join(tops))
    else:
        lines.append("（该事件自家未开窗）")
    if rec.get("zh_hint"):
        lines.append(rec["zh_hint"])
    r = rec.get("zh_reach")
    if r and r.get("recommend"):
        lines.append(f"若立直：宣言牌 {r['recommend']}")
    return lines


def latest_record(lines: list[str]) -> dict | None:
    """尾随缓冲：取最后一条可解析 JSON 行（坏行=截断残留，跳过不崩）。"""
    for ln in reversed(lines):
        try:
            rec = json.loads(ln)
        except ValueError:
            continue
        if isinstance(rec, dict):
            return rec
    return None


def stale_lines(rec: dict | None, idle_s: float,
                *, grace: float = STALE_GRACE_S) -> tuple[list[str], bool]:
    """(显示行, 是否停更)。纯函数可单测：宽限期内=原样；停更后保留最后
    一条供回看、追加时长按置灰降级；从未有过记录（初始态）不判停更。"""
    lines = format_lines(rec)
    if rec is None or idle_s <= grace:
        return lines, False
    mm = int(idle_s // 60)
    t = str(rec.get("ts") or "")[11:19]
    return [*lines, f"（无新事件 {mm} 分钟，停更于 {t or '?'}）"], True


# ---------- 持久设置 ----------

def settings_path() -> pathlib.Path:
    return pathlib.Path.home() / ".mjbrain" / "hud.json"


def load_settings() -> dict:
    p = settings_path()
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(s, dict):
            return s
    except (OSError, ValueError):
        pass
    return {}


def save_settings(s: dict) -> None:
    p = settings_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # 设置落盘失败不影响显示


# ---------- Tk 浮窗 ----------

class HudApp:
    def __init__(self, advise: pathlib.Path, settings: dict,
                 wait: bool, poll_ms: int):
        import tkinter as tk  # 延迟到此才要求 GUI 栈

        self.advise = advise
        self.wait = wait
        self.poll_ms = poll_ms
        self.s = {
            "x": int(settings.get("x", 40)),
            "y": int(settings.get("y", 40)),
            "alpha": float(settings.get("alpha", DEFAULT_ALPHA)),
            "font": int(settings.get("font", DEFAULT_FONT_SIZE)),
            "topmost": bool(settings.get("topmost", True)),
            "locked": bool(settings.get("locked", False)),
        }
        self.pos = 0
        self.shown = None  # 当前显示的文本（避免无变化重绘）
        self.shown_stale = False  # 当前是否处于置灰降级态（fg 只在翻转时改）
        self.rec_seen = None  # 最后一条已显示记录（停更时留档回看）
        self.fresh_at = time.monotonic()  # 最后一次收到新记录的时刻

        self.root = tk.Tk()
        self.root.title("mjbrain HUD")
        # 全平台无边框：mac 在 Tk 8.6.13 实测 overrideredirect 即帧高==内容高
        # （MacWindowStyle 只改内部记账不改原生标题栏，已证伪，勿用）
        self.root.overrideredirect(True)
        self._try_alpha(self.s["alpha"])
        self.root.attributes("-topmost", self.s["topmost"])
        self.root.configure(bg="#101418")
        self.root.geometry(f"+{self.s['x']}+{self.s['y']}")

        fam = {"win32": "Microsoft YaHei",
               "darwin": "PingFang SC"}.get(sys.platform, "DejaVu Sans")
        self.label = tk.Label(self.root, text="等待建议…", justify="left",
                              anchor="w", fg=FG_FRESH, bg="#101418",
                              font=(fam, self.s["font"]))
        self.label.pack(padx=10, pady=6)

        # 拖动：无边框窗没有标题栏，左键按住整窗移动（锁定时忽略）
        self._drag = None
        for w in (self.root, self.label):
            w.bind("<Button-1>", self._drag_start)
            w.bind("<B1-Motion>", self._drag_move)
            w.bind("<Button-3>", self._menu)       # 右键：锁定/置顶/退出
            w.bind("<Escape>", lambda e: self.stop())
            w.bind("l", lambda e: self._toggle_lock())  # L 快捷切换锁定

    def _try_alpha(self, a: float) -> None:
        try:
            self.root.attributes("-alpha", max(0.3, min(1.0, a)))
        except Exception:  # noqa: BLE001 —— Aqua 等不支持时静默跳过
            pass

    def _drag_start(self, ev) -> None:
        if self.s["locked"]:
            return
        self._drag = (ev.x_root - self.s["x"], ev.y_root - self.s["y"])

    def _drag_move(self, ev) -> None:
        if not self._drag or self.s["locked"]:
            return
        self.s["x"], self.s["y"] = ev.x_root - self._drag[0], ev.y_root - self._drag[1]
        self.root.geometry(f"+{self.s['x']}+{self.s['y']}")

    def _menu(self, ev) -> None:
        import tkinter as tk

        m = tk.Menu(self.root, tearoff=0)
        m.add_command(
            label="解锁位置 (L)" if self.s["locked"] else "锁定位置 (L)",
            command=self._toggle_lock)
        m.add_command(
            label="钉住置顶" if not self.s["topmost"] else "取消置顶",
            command=self._toggle_top)
        m.add_command(label="退出 (Esc)", command=self.stop)
        try:
            m.tk_popup(ev.x_root, ev.y_root)
        finally:
            m.grab_release()

    def _save(self) -> None:
        save_settings(self.s)

    def _toggle_lock(self) -> None:
        self.s["locked"] = not self.s["locked"]
        self._save()

    def _toggle_top(self) -> None:
        self.s["topmost"] = not self.s["topmost"]
        self.root.attributes("-topmost", self.s["topmost"])
        self._save()

    def stop(self) -> None:
        self._save()
        self.root.destroy()

    def _read_new(self) -> dict | None:
        """尾随 advise JSONL（行缓冲文件；截断/轮转时回卷重读）。

        文件暂缺（live 还没落盘、或 --wait 场景开局前）安静返回 None，
        下个 tick 再试——main() 已对"不带 --wait 且文件不存在"fail-fast。
        """
        try:
            with self.advise.open(encoding="utf-8") as fh:
                fh.seek(self.pos)
                chunk = fh.read()
                self.pos = fh.tell()
        except OSError:  # FileNotFoundError 是子类：未出现/未就绪都走这
            return None
        if not chunk:
            try:
                if self.advise.stat().st_size < self.pos:  # 轮转重写
                    self.pos = 0
            except OSError:
                pass
            return None
        return latest_record(chunk.splitlines())

    def tick(self) -> None:
        rec = self._read_new()
        if rec is not None:  # 新记录：换页 + 回到"新鲜"计时起点
            self.rec_seen = rec
            self.fresh_at = time.monotonic()
        lines, stale = stale_lines(self.rec_seen, time.monotonic() - self.fresh_at)
        text = "\n".join(lines)
        if text != self.shown:
            self.shown = text
            self.label.configure(text=text)
        if stale != self.shown_stale:  # fg 只在置灰/复原翻转时改
            self.shown_stale = stale
            self.label.configure(fg=FG_STALE if stale else FG_FRESH)
        self.root.after(self.poll_ms, self.tick)

    def run(self) -> int:
        self.tick()
        self.root.mainloop()
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--advise", required=True, type=pathlib.Path,
                    help="live_from_capture 的 <frames>.advise.jsonl 路径")
    ap.add_argument("--wait", action="store_true",
                    help="文件暂未出现时不退出（live 起来后才落盘）")
    ap.add_argument("--poll", type=int, default=250, help="轮询毫秒")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--geometry", default=None,
                    help="本次起始位置 X,Y（覆盖持久设置的初值；拖动/退出后照常持久化）")
    ap.add_argument("--no-topmost", action="store_true")
    args = ap.parse_args(argv)

    s = load_settings()
    if args.alpha is not None:
        s["alpha"] = args.alpha
    if args.geometry:
        x, _, y = args.geometry.partition(",")
        try:
            s["x"], s["y"] = int(x), int(y)
        except ValueError:
            print(f"--geometry 要 X,Y 整数（如 40,40）：{args.geometry!r}", file=sys.stderr)
            return 2
    if args.no_topmost:
        s["topmost"] = False
    if not args.advise.exists() and not args.wait:
        print(f"文件不存在: {args.advise}（live_from_capture 跑起来才会生成；"
              "或加 --wait 等它出现）", file=sys.stderr)
        return 1
    try:
        app = HudApp(args.advise, s, wait=args.wait, poll_ms=args.poll)
    except ImportError as ex:  # tkinter 缺失的机器（典型：conda 精简 python）
        print(f"需要 Tkinter（GUI 栈）但导入失败：{ex}\n"
              "conda 环境：conda install -n <env> python-tk=3.12；"
              "Ubuntu：sudo apt install python3-tk；macOS 官方 python 自带。",
              file=sys.stderr)
        return 2
    except RuntimeError as ex:  # 无显示环境（headless）
        print(f"无法创建窗口（无图形会话？）：{ex}", file=sys.stderr)
        return 2
    return app.run()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
