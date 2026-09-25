"""客户端 HTTP 地址规范器：advisor 服务地址的唯一校验口。

威胁模型里这不是真 SSRF（脚本都是操作员本机 CLI/常量，没有服务端被诱导
请求的场景），但校验口有二：① 把"地址必须 http(s)://host[:port]"从口头
约定变成代码保证（顺手把 127.0.0.1:8765 漏写 scheme 这类误用接住）；
② 动态 URL 过 urlsplit 白名单后再进 urlopen，静态审计口径下污点消解。
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_SCHEMEISH = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")  # `javascript:x` 式伪 host


def normalize_http_base(url: str, what: str = "advisor 地址") -> str:
    """规范化为 http(s)://host[:port]（无路径）。非法即 SystemExit。"""
    u = str(url).strip().rstrip("/")
    if u and "://" not in u and not _SCHEMEISH.match(u):
        u = "http://" + u  # 只给 `127.0.0.1:8765` 这类裸 host[:port] 补前缀；
        # `javascript:x` 保持原样交给 scheme 白名单毙掉，不许被洗成 http
    s = urlsplit(u)
    if s.scheme not in ("http", "https") or not s.netloc:
        raise SystemExit(f"非法{what}：{url!r}（须为 http(s)://host[:port]）")
    if s.path not in ("", "/") or s.query or s.fragment:
        raise SystemExit(f"非法{what}：{url!r}（只准 host[:port]，不带路径/参数）")
    return u


def host_of(base: str) -> str:
    return urlsplit(base).hostname or ""


def require_loopback(base: str, what: str = "服务地址") -> str:
    """字面 host 必须 ∈ 环回白名单：调用方自己拉起本机子进程服务的场景
    （run_demo → advisor）用这个，杜绝"健康检查打到别人家/元数据端点"。"""
    if host_of(base) not in _LOOPBACK:
        raise SystemExit(f"{what}只准指本机（127.0.0.1/::1/localhost），得 {base!r}")
    return base


def warn_if_not_loopback(base: str) -> None:
    """非本机 advisor 只提示不拦（LAN 部署合法），但数据流向要说清。"""
    h = host_of(base)
    if h not in _LOOPBACK:
        print(f"[net] 提示：请求发往非本机地址 {base}——事件流将离开本机")
