"""brain.net：advisor 地址校验口。"""

from __future__ import annotations

import pytest

from brain.net import is_loopback, normalize_http_base, require_loopback


def test_normalize_accepts():
    assert normalize_http_base("http://127.0.0.1:8765") == "http://127.0.0.1:8765"
    assert normalize_http_base("127.0.0.1:8765/") == "http://127.0.0.1:8765"
    assert normalize_http_base("https://advisor.local") == "https://advisor.local"


def test_normalize_rejects():
    for bad in ("", "ftp://h", "http://", "http://h:1/path", "http://h?x=1",
                "javascript:alert(1)"):
        with pytest.raises(SystemExit):
            normalize_http_base(bad)


def test_require_loopback():
    assert require_loopback("http://127.0.0.1:8765") == "http://127.0.0.1:8765"
    assert require_loopback("http://localhost:8765")
    with pytest.raises(SystemExit, match="只准指本机"):
        require_loopback("http://192.168.1.7:8765")
    with pytest.raises(SystemExit, match="只准指本机"):
        require_loopback("http://metadata.google.internal")


def test_is_loopback():
    assert is_loopback("http://127.0.0.1:8765")
    assert is_loopback("http://localhost:8765")
    assert not is_loopback("http://192.168.1.7:8765")
    assert not is_loopback("https://evil.example")
