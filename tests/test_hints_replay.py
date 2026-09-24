"""hint 全窗回放回归：run6 真机帧 → mjai 事件流 → 逐事件开窗 → hint 形状审计。

不跑模型（hint 是纯引擎特征），只验两件事：
1) 合法——每个窗口的 hint 字段/取值域全部合规（牌名限 34 种、向听 -1..13、
   melded 为 bool），全程 hint_from_obs 不返回 None（真 riichienv 观测没有
   算不出的理由）；
2) 覆盖——一整局的真实开窗数量可观（>50），且至少出现过听牌窗（存在
   waits 非空），否则等于没测到。

run6 帧文件与 liqi 生成码（_gen）都不进公开仓：缺任何一样即 skip。
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import time

import pytest

_ROOT = pathlib.Path(__file__).parents[1]
_FRAMES = _ROOT / "data" / "raw" / "ms_frames" / "run6.jsonl"
_gen = _ROOT / "capture" / "liqi" / "_gen" / "liqi_pb2.py"

needs_data = pytest.mark.skipif(not (_FRAMES.exists() and _gen.exists()),
                                reason="run6 帧或 _gen 生成码不在（公开仓常态）")

TILE_NAMES = {f"{n}{s}" for s in "mps" for n in range(1, 10)} | {"E", "S", "W", "N", "P", "F", "C"}


def _load_glue():
    path = _ROOT / "scripts" / "live_from_capture.py"
    spec = importlib.util.spec_from_file_location("live_from_capture", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_data
def test_run6_all_windows_hint_shapes():
    from advisor import hints
    from advisor.server import open_windows
    from capture.liqi import runtime
    from riichienv import GameRule, RiichiEnv

    glue = _load_glue()
    p, st = runtime.make_stack()
    st.seat = 0
    events: list[dict] = []
    with _FRAMES.open(encoding="utf-8") as fh:
        glue.feed_lines(p, st, events, fh.readlines())
    assert len(events) > 200, f"帧→事件流断链？仅 {len(events)} 事件"

    # 雀魂流的遮蔽语义：只自家座位的门前手牌是真实的（他座 tehais 全是 "?"，
    # RiichiEnv 静默按 1m 解析）→ 审计范围=自家窗，与生产请求（seat=自家）一致。
    assert st.seat is not None, "run6 缺 AuthGame 自家座位"
    env = RiichiEnv(game_mode="4p-red-half", rule=GameRule.default_mjsoul())
    applied = 0
    n_win = n_hint = n_tenpai = 0
    t0 = time.time()
    for ev in events:
        try:
            env.apply_event(ev)
            applied += 1
        except Exception:  # noqa: BLE001 —— 与 server.react 同语义：停在可解释处
            break
        for pid, (ob, legal) in open_windows(env).items():
            if pid != st.seat:
                continue
            n_win += 1
            h = hints.hint_from_obs(ob)
            assert h is not None, f"事件{applied} 座{pid}: 真观测下 hint 不应为 None"
            n_hint += 1
            assert set(h) == {"shanten", "waits", "dora", "dora_in_hand", "melded"}
            assert h["shanten"] is None or -1 <= int(h["shanten"]) <= 13
            assert isinstance(h["melded"], bool)
            assert all(t in TILE_NAMES for t in h["waits"]), h["waits"]
            assert all(t in TILE_NAMES for t in h["dora"]), h["dora"]
            assert 0 <= int(h["dora_in_hand"]) <= 14
            if h["waits"]:
                n_tenpai += 1
                assert h["shanten"] in (0, None)  # 有听牌线：向听要么 0 要么副露降级
    assert applied == len(events)  # 事件流应全程可重放（同 react 的 applied 语义）
    assert n_win == n_hint and n_win >= 50, (n_win, n_hint)
    assert n_tenpai >= 1
    assert time.time() - t0 < 60, "全回放 hint 不该慢过一分钟"


def test_illegal_hand_panic_is_swallowed():
    """回归护栏：物理 id 重复 14 次的非法牌面（雀魂 "?" 遮蔽手被 RiichiEnv
    静默解析即成此形状）会让 calculate_shanten 的 Rust 侧抛 PanicException——
    它不是 Exception 子类。hint_from_obs 必须接住并降级：向听/待牌置空，
    宝牌等不依赖牌面的字段照常输出。"""
    from riichienv import Action, ActionType

    from advisor import hints

    class GarbageOb:
        hand = [0] * 14                      # 1m 第一张物理 id 重复 14 次
        player_id = 0
        melds = [[], [], [], []]
        dora_indicators = [108]              # 东 → 宝牌南

        @staticmethod
        def legal_actions():
            return [Action(ActionType.DISCARD, tile=0)]

    h = hints.hint_from_obs(GarbageOb())    # 曾直接 panic 穿出
    assert h is not None and h["shanten"] is None and h["waits"] == []
    assert h["dora"] == ["S"]
