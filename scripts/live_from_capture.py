#!/usr/bin/env python
"""M4 档1 实时推荐客户端：捕获 JSONL → mjai 事件流 → POST /v1/react → 打印+落盘。

读 scripts/run_capture.py 落盘的 JSONL（每行 {"dir":..,"b64":..}）：
  每行 base64 → capture.liqi runtime 的 parser.parse → Frame →
  MajsoulState.dispatch → mjai 事件累加；每当流有新事件，把**全量流**
  POST 给 advisor/server.py 的 /v1/react（服务端重放出决策窗），
  自家座位开窗时打印 B3 推荐 + top 候选（mjai 原串经 advisor.zh 映射成中文），
  行首标「座位N（自家）」；自家座位只在换局时变（换座打一行提示）。立直可选
  窗会多一行「↳ 若立直：宣言牌 → 切X」（服务端两段式跑宣言牌窗，见
  advisor.server._reach_declare_window）。

模型表现记录：每个成功响应追加一行到 `<输入名>.advise.jsonl`
（`--log` 改路径、`--no-log` 关闭），行= {"ts","event_n","seat","rtt_s",
"zh_recommend","zh_top","zh_reach","zh_hint","resp"}（zh_reach 仅立直可选窗有值：
"若立直，宣言牌切哪张"的第二段建议；zh_hint=向听/待牌/宝牌一行中文，见
advisor.hints），行缓冲随写随刷——事后与帧 JSONL
（实际打法）对照即得"模型推荐 vs 实际出牌"逐窗记录；缺位审计=数
window:true 的行。

前置：另开一个进程跑推荐服务
    python -m advisor.server            # 127.0.0.1:8765，启动即预热权重
和捕获
    python scripts/run_capture.py --url <雀魂页> --out data/raw/ms_frames/run1.jsonl

红线自查：本脚本只读 JSONL、只发本机 HTTP 建议请求；对雀魂客户端
零接触、零写入（动作注入禁止）。

--follow 尾随模式：文件边写边读，读到尾 sleep 重试；捕获进程追加帧即出推荐
（依赖捕获侧行缓冲落盘——run_capture.py 已 buffering=1，块缓冲会让帧滞后整块）。
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from advisor import zh  # noqa: E402 —— 显示层中文映射（日志 resp 仍存 mjai 原串）

# 推荐记录计数（顶层 finally 打印汇总用：Ctrl-C 退出也能看到行数）
STAT = {"path": None, "n": 0}


def post_react(url: str, payload: dict, timeout: float = 30.0, *,
               allow_remote: bool = False) -> dict:
    from brain import net
    # 协议规范化 + host 边界双闸都贴着 sink：本函数任何调用方（含未来
    # 新增）不依赖调用前先校验过；事件流离本机必须是显式决定
    url = net.normalize_http_base(url, "advisor 地址")
    if not allow_remote:
        net.require_loopback(url, "advisor 地址")
    req = urllib.request.Request(
        url + "/v1/react",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fmt_reach(r: dict) -> str:
    """立直宣言牌建议的第二行（模型在主窗可选立直时附带）。"""
    top = "  ".join(f"{zh.action_zh(t['a'])} {t['p'] * 100:.1f}%"
                    for t in r.get("top", [])[:3])
    return f"   ↳ 若立直：宣言牌 → {zh.action_zh(r['recommend'])} | {top}"


def fmt_decision(d: dict, my_seat: int | None = None) -> str:
    tag = "（自家）" if my_seat is not None and d.get("seat") == my_seat else ""
    if d.get("window") is False:
        return f"座位{d['seat']}{tag}: 无决策窗"
    if "error" in d:
        return f"座位{d['seat']}{tag}: {d['error']}"
    top = "  ".join(f"{zh.action_zh(t['a'])} {t['p'] * 100:.1f}%"
                    for t in d.get("top", [])[:3])
    line = (f"[{time.strftime('%H:%M:%S')}] 座位{d['seat']}{tag} 候选{d['legal_n']}"
            f" → {zh.action_zh(d['recommend'])} | {top}")
    hint = zh.hint_zh(d.get("hint"))
    return f"{line} | {hint}" if hint else line


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
    # 重定向到管道/文件时 ↳/→ 在 Windows 本地编码下会 UnicodeEncodeError；
    # errors=replace 只兜重定向，终端行为不变（全库 review 统一模式）
    import sys as _sys
    for _s in (_sys.stdout, _sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jsonl", required=True, type=pathlib.Path,
                    help="run_capture.py 落盘的帧 JSONL")
    ap.add_argument("--seat", type=int, default=-1,
                    help="座位号；缺省自动取状态机从 AuthGame 帧识别的自家座位")
    ap.add_argument("--url", default="http://127.0.0.1:8765",
                    help="advisor.server 地址（默认本机 8765）")
    ap.add_argument("--allow-remote", action="store_true",
                    help="放行 --url 指向非本机 advisor（默认关：mjai 事件流不出本机）")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--follow", action="store_true",
                    help="尾随模式：读到文件尾不退出，等捕获追加")
    ap.add_argument("--poll", type=float, default=0.1, help="--follow 轮询秒")
    ap.add_argument("--log", type=pathlib.Path, default=None,
                    help="推荐记录 JSONL 路径（默认 <输入名>.advise.jsonl）")
    ap.add_argument("--no-log", action="store_true", help="不落盘推荐记录")
    args = ap.parse_args(argv)

    # --url 是全链唯一动态服务地址：协议/host 校验后才进 urlopen；
    # host 边界 fail-closed——非本机必须 --allow-remote 显式放行
    from brain import net
    args.url = net.normalize_http_base(args.url, "--url")
    if not net.is_loopback(args.url):
        if not args.allow_remote:
            print(f"--url 指向非本机地址 {args.url}：需 --allow-remote 显式放行"
                  f"（注意：mjai 事件流将离开本机）", file=sys.stderr)
            return 1
        print(f"[net] 提示：请求发往非本机地址 {args.url}——事件流离开本机")

    while args.follow and not args.jsonl.exists():
        print(f"等 {args.jsonl} 出现…", file=sys.stderr)
        time.sleep(1.0)
    if not args.jsonl.exists():
        print(f"文件不存在: {args.jsonl}", file=sys.stderr)
        return 1

    log_fh = None
    if not args.no_log:
        log_path = args.log or args.jsonl.with_name(args.jsonl.stem + ".advise.jsonl")
        log_fh = log_path.open("a", encoding="utf-8", buffering=1)  # 行缓冲=随写随刷
        STAT["path"] = str(log_path)
        print(f"推荐记录 -> {log_path}")

    from capture.liqi import runtime

    p, st = runtime.make_stack()
    events: list[dict] = []
    LAST: dict = {}          # 各座位上次已打印的决策签名（同一决策窗重复回应不刷屏）
    last_err = None
    seat_seen = None         # 已报过的自家座位（换座=换局，只在新座位首次出现时报一次）
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
            if seat is not None and seat != seat_seen:
                # 座位只在换局时变（认证帧 seat_list 决定）；显示与模型输入同源
                if seat_seen is None:
                    print(f"— 自家座位 {seat}", flush=True)
                else:
                    print(f"— 换座 {seat_seen} → {seat}", flush=True)
                seat_seen = seat
            body = {"events": events, "top": args.top}
            if seat is not None:
                body["seat"] = int(seat)
            t_post = time.perf_counter()
            try:
                resp = post_react(args.url, body, allow_remote=args.allow_remote)
                last_err = None
            except (urllib.error.URLError, OSError) as ex:
                if last_err is None:
                    print(f"! 服务不可达（{args.url}）：{ex}；"
                          "确认 python -m advisor.server 在跑，持续重试", file=sys.stderr)
                    last_err = ex
                continue
            rtt = time.perf_counter() - t_post
            if log_fh is not None:
                rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "event_n": len(events),
                       "seat": body.get("seat"),
                       "rtt_s": round(rtt, 3),
                       "zh_recommend": None,
                       "zh_top": None,
                       "zh_reach": None,
                       "zh_hint": None,
                       "resp": resp}
                decs = [x for x in resp.get("decisions", []) if "recommend" in x]
                picked = next((x for x in decs if x.get("seat") == body.get("seat")),
                              decs[0] if decs else None)
                if picked is not None:
                    rec["zh_recommend"] = zh.action_zh(picked["recommend"])
                    rec["zh_top"] = [[zh.action_zh(t["a"]), t["p"]]
                                     for t in picked.get("top", [])]
                    rec["zh_hint"] = zh.hint_zh(picked.get("hint")) or None
                    rw = picked.get("reach")
                    if rw:
                        rec["zh_reach"] = {
                            "recommend": zh.action_zh(rw["recommend"]),
                            "top": [[zh.action_zh(t["a"]), t["p"]]
                                    for t in rw.get("top", [])]}
                log_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                STAT["n"] += 1
            if "error" in resp:
                print(f"服务端：{resp['error']}（applied={resp.get('applied')}）",
                      file=sys.stderr)
                continue
            for d in resp.get("decisions", []):
                seat_d = d.get("seat")
                if d.get("window") is False:
                    LAST.pop(seat_d, None)   # 窗口关闭：清记忆，下次开窗重新打印
                    continue
                if "error" in d:
                    print(f"座位{seat_d}: {d['error']}", file=sys.stderr)
                    continue
                h = d.get("hint") or {}
                sig = (d.get("recommend"), d.get("legal_n"),
                       tuple(t["a"] for t in d.get("top", [])),
                       (d.get("reach") or {}).get("recommend"),
                       (h.get("shanten"), tuple(h.get("waits") or []),
                        tuple(h.get("dora") or [])))
                if LAST.get(seat_d) == sig:  # 同一决策窗的重复回应不刷屏
                    continue
                LAST[seat_d] = sig
                print(f"{fmt_decision(d, body.get('seat'))} | 往返 {rtt:.2f}s", flush=True)
                if d.get("reach"):
                    print(fmt_reach(d["reach"]), flush=True)
    if events:
        print(f"共 {len(events)} 个 mjai 事件入流。")
    return 0


if __name__ == "__main__":
    code = 0
    try:
        code = main()
    except KeyboardInterrupt:
        pass  # Ctrl-C 静默退出；记录行缓冲已刷盘
    finally:
        if STAT["path"]:
            print(f"推荐记录汇总 -> {STAT['path']}（{STAT['n']} 行）")
    raise SystemExit(code)
