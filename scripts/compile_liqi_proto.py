"""把 reference/akagi-v3/bridge/liqi.proto 编成 capture/liqi/_gen/liqi_pb2.py。

生成物不进 git（.gitignore capture/liqi/_gen/）。跑法：
  python scripts/compile_liqi_proto.py
proto 源变动（换版雀魂协议）= 重跑本脚本 + 跑 tests/test_liqi_routes.py。
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTO_DIR = ROOT / "reference" / "akagi-v3" / "bridge"
OUT = ROOT / "capture" / "liqi" / "_gen"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "__init__.py").touch()
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={OUT}",
        "liqi.proto",
    ]
    r = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    )
    if r.returncode:
        print(r.stdout, r.stderr)
        return 1
    gen = OUT / "liqi_pb2.py"
    print(f"{gen.name}: {gen.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
