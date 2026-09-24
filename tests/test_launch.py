"""capture.chromium.launch：持久 profile 缺省 + try_attach 复用已运行实例。

全程不拉真浏览器：attach 探活用本机随机端口（关了=死端口）与一个最小
HTTP 假端点；spawn/命令行拼装类用例只 monkeypatch Popen。
"""

from __future__ import annotations

import http.server
import json
import pathlib
import socket
import threading

import pytest

from capture.chromium import launch


def test_default_profile_is_persistent_home(tmp_path, monkeypatch):
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    d = launch.default_user_data_dir()
    assert d == tmp_path / ".mjbrain" / "browser-profile"


def test_try_attach_no_file(tmp_path):
    assert launch.try_attach(tmp_path) is None


def test_try_attach_garbage_file(tmp_path):
    (tmp_path / "DevToolsActivePort").write_text("not-a-port\n", encoding="utf-8")
    assert launch.try_attach(tmp_path) is None


def test_try_attach_dead_port(tmp_path):
    # 借一个绑过又释放的端口：connect 必拒，等同陈旧文件
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    (tmp_path / "DevToolsActivePort").write_text(f"{port}\n/devtools/browser/x\n",
                                                 encoding="utf-8")
    assert launch.try_attach(tmp_path) is None


@pytest.fixture()
def fake_cdp_endpoint():
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = json.dumps({"webSocketDebuggerUrl": "ws://127.0.0.1:1/devtools/browser/fake"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        t.join(timeout=2)


def test_try_attach_live_instance(tmp_path, fake_cdp_endpoint):
    (tmp_path / "DevToolsActivePort").write_text(
        f"{fake_cdp_endpoint}\n/devtools/browser/uuid\n", encoding="utf-8")
    assert launch.try_attach(tmp_path) == "ws://127.0.0.1:1/devtools/browser/fake"


def test_spawn_browser_cmd_shape(tmp_path, monkeypatch):
    seen: dict = {}

    class FakePopen:
        pid = 4242

        def __init__(self, cmd, **kw):
            seen["cmd"] = [str(c) for c in cmd]

    monkeypatch.setattr(launch.subprocess, "Popen", FakePopen)
    proc, udd, port = launch.spawn_browser(
        "https://game.maj-soul.com/1/",
        user_data_dir=tmp_path / "prof",
        exe=pathlib.Path("/bin/chrome"),
    )
    cmd = seen["cmd"]
    assert proc.pid == 4242
    assert cmd[0] == str(pathlib.Path("/bin/chrome"))  # win 上 pathlib 会归一分隔符
    assert f"--user-data-dir={tmp_path / 'prof'}" in cmd
    assert f"--remote-debugging-port={port}" in cmd
    assert cmd[-1] == "https://game.maj-soul.com/1/"
    assert (tmp_path / "prof").is_dir()
