"""最小可跑服务骨架：conda activate mjbrain && python -m uvicorn server.app:app（M4 前只回 501）。"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="mjbrain", version="0.0.1")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/v1/react")
def react() -> dict:
    # 契约形状照 Akagi /v3/react（PLAN 决策 D1）；实现在 M4
    from fastapi import HTTPException

    raise HTTPException(501, "M4: brain+engine wiring pending")
