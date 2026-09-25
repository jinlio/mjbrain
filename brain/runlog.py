"""runs/ 训练记录的最小实现（规范见 runs/README.md）。

用法（训练驱动脚本里）::

    from brain.runlog import RunLog
    log = RunLog("laya-ft-v1", config=cfg_dict)      # 建目录+config.yaml+env.txt
    for step in range(...):
        ...
        log.metric(step, loss=l, kl=k)               # 指标一行一条，实时 flush
    log.finish({"final_pt": -12.3, "ck_sha256": sha, "status": "ok"})
    # 失败路径同样 finish({"status": "failed", "reason": ...})
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import platform
import subprocess
import time


def _git_hash() -> str:
    root = pathlib.Path(__file__).resolve().parents[1]
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return r.stdout.strip() or "?"
    except Exception:  # noqa: BLE001 — git 缺失时记 '?'，绝不让记录器炸训练
        return "?"


class RunLog:
    def __init__(self, tag: str, config: dict, *, root: str | pathlib.Path | None = None) -> None:
        # run_id 会被 checkpoints/、runs/、预测 dump 三侧复用；tag 含路径
        # 分隔符或 .. 会让这些落盘点静默逃出预期目录（静态审计的 path
        # traversal 源头）。在唯一产出点拦住，调用方（含 trainer）无需改。
        if not tag or any(sep in tag for sep in ("/", "\\", "..")):
            raise ValueError(f"tag 须为非空路径安全串（无 / \\ ..）：{tag!r}")
        repo = pathlib.Path(__file__).resolve().parents[1]
        self.root = pathlib.Path(root) if root else repo / "runs"
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{ts}-{tag}"
        self.dir = self.root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)

        self._config_path = self.dir / "config.yaml"
        self._dump_config(config)

        try:
            import torch

            tinfo = f"torch {torch.__version__} | cuda {torch.version.cuda} | gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}"
        except Exception:  # noqa: BLE001 — torch 缺席时降级记录
            tinfo = "torch n/a"
        (self.dir / "env.txt").write_text(
            "\n".join(
                [
                    f"git={_git_hash()}",
                    f"python={platform.python_version()}",
                    f"conda_env={os.environ.get('CONDA_DEFAULT_ENV', '?')}",
                    f"platform={platform.platform()}",
                    tinfo,
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        self._metrics = (self.dir / "metrics.jsonl").open("a", encoding="utf-8")
        self._t0 = _dt.datetime.now(_dt.UTC)
        self._closed = False

    def _dump_config(self, config: dict) -> None:
        """无 PyYAML 依赖的降级序列化：json 是 yaml 的子集。"""
        self._config_path.write_text(
            json.dumps(config, indent=1, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

    def metric(self, step: int, **kv) -> None:
        if self._closed:  # finish 后再来 metric：静默丢，别让日志器炸训练
            return
        rec = {"step": step, "wall_s": (_dt.datetime.now(_dt.UTC) - self._t0).total_seconds(), **kv}
        self._metrics.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._metrics.flush()

    @staticmethod
    def _idx_lock(root: pathlib.Path, timeout: float = 10.0):
        """RUNS.md 读-改-写的跨进程锁（mkdir 原子，Win/POSIX 通用）。
        拿不到/陈旧（持锁进程死过 60s）→ 返回 None 直接写：单次写本身有
        tmp+replace 保证原子，最坏丢一行索引，绝不死锁。"""
        lock = root / "RUNS.md.lock"
        t0 = time.time()
        while True:
            try:
                lock.mkdir()
                return lock
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > 60:
                        lock.rmdir()  # 陈旧锁接管
                        continue
                except OSError:
                    pass
                if time.time() - t0 > timeout:
                    return None
                time.sleep(0.1)

    def finish(self, summary: dict) -> None:
        self._closed = True
        self._metrics.close()
        (self.dir / "summary.json").write_text(
            json.dumps({"run_id": self.run_id, **summary}, indent=1, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        idx = self.root / "RUNS.md"
        if not idx.exists():
            idx.write_text("# run 索引\n\n| 日期 | run_id | 目的 | 关键结果 |\n|---|---|---|---|\n", encoding="utf-8")
        date = self.run_id.split("T")[0]
        tag = self.run_id.split("-", 1)[1]
        key = "; ".join(f"{k}={v}" for k, v in summary.items() if k != "status") or json.dumps(summary, ensure_ascii=False)[:120]
        row = f"| {date} | {self.run_id} | {tag} | [{summary.get('status', 'ok')}] {key} |\n"
        # 插到表格末尾（最后一条 "|" 行之后），而不是文件末尾：
        # RUNS.md 可在表下另设"诊断记录"等小节，盲 append 会把行甩到小节后面。
        # 读-改-写全程持跨进程锁：并发训练同时 finish 不再互吞索引行。
        lock = self._idx_lock(self.root)
        try:
            lines = idx.read_text(encoding="utf-8").splitlines(keepends=True)
            tail = [i for i, ln in enumerate(lines) if ln.startswith("|")]
            at = tail[-1] + 1 if tail else 0
            lines.insert(at, row)
            tmp = idx.with_suffix(".md.tmp")
            tmp.write_text("".join(lines), encoding="utf-8")
            try:
                tmp.replace(idx)
            except PermissionError:
                # Windows：RUNS.md 被编辑器/杀软/tail 占用时 replace 会炸。
                # summary.json 已先落盘（真源在手），索引行降级 append 补记
                with idx.open("a", encoding="utf-8") as f:
                    f.write(row)
                tmp.unlink(missing_ok=True)
        finally:
            if lock is not None:
                lock.rmdir()
