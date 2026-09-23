"""M4 出口工具（scripts/compare_advise.py）的纯函数测试：无需 ckpt/棋谱。

judge/next_self_action 是"推荐 vs 实际"对齐的核心判定，口径错会直接污染验收
数字，故逐条钉死（含赤 5 归一、跳过=摸牌、暗杠比 consumed、跨局边界）。
"""

from __future__ import annotations

import importlib.util
import pathlib

_path = pathlib.Path(__file__).parents[1] / "scripts" / "compare_advise.py"
_spec = importlib.util.spec_from_file_location("compare_advise", _path)
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def test_tile_and_desc_normalization():
    assert C.norm_tile("0m") == "5m" and C.norm_tile("5pr") == "5p"
    assert C.norm_tile("7s") == "7s"
    assert C.norm_desc("dahai:5sr") == "dahai:5s" and C.norm_desc("reach") == "reach"


def test_next_self_action_skips_others_and_names_actions():
    events = [
        {"type": "dahai", "actor": 2, "pai": "1m"},      # 他家
        {"type": "dora", "dora_marker": "9m"},           # 非动作事件
        {"type": "tsumo", "actor": 1, "pai": "?"},       # 自家摸牌（动作集内）
        {"type": "dahai", "actor": 1, "pai": "3p"},
    ]
    i, e = C.next_self_action(events, 0, 1)
    assert (i, e["type"]) == (2, "tsumo")
    i2, e2 = C.next_self_action(events, 3, 1)
    assert (i2, e2["pai"]) == (3, "3p")
    assert C.next_self_action(events, 0, 3) == (None, None)


def test_judge_dahai_match_and_mismatch():
    ev = {"type": "dahai", "actor": 0, "pai": "7p", "tsumogiri": False}
    ok, act, note = C.judge("dahai:7p", ev, 10, [], 0)
    assert ok is True and act == "dahai 7p" and note == ""
    ok2, _, note2 = C.judge("dahai:7p", {"type": "dahai", "pai": "7p",
                                         "tsumogiri": True}, 10, [], 0)
    assert ok2 is True and note2 == "摸切"
    ok3, act3, _ = C.judge("dahai:3s", ev, 10, [], 0)
    assert ok3 is False and act3 == "dahai 7p"
    # 赤 5 与普通 5 同张
    assert C.judge("dahai:5m", {"type": "dahai", "pai": "0m"}, 0, [], 0)[0] is True


def test_judge_pass_and_action_windows():
    tsumo = {"type": "tsumo", "actor": 1, "pai": "5s"}
    assert C.judge("none", tsumo, 5, [], 1)[0] is True            # 跳过=随后摸牌
    assert C.judge("none", {"type": "pon", "actor": 1, "pai": "9m"},
                   5, [], 1)[0] is False                          # 说跳过却鸣了
    # 打牌窗撞上"自家先摸牌"= 窗与回应错位（不是模型错），要点名对齐可疑
    ok, act, note = C.judge("dahai:1m", tsumo, 5, [], 1)
    assert ok is False and note == "对齐可疑" and act == "跳过（先摸牌）"
    assert C.judge("reach", {"type": "reach", "actor": 1}, 5, [], 1)[0] is True
    assert C.judge("hora", {"type": "hora", "actor": 1}, 5, [], 1)[0] is True


def test_judge_call_windows():
    ev = {"type": "pon", "actor": 1, "pai": "1z", "consumed": ["1z", "1z"]}
    assert C.judge("pon:1z", ev, 5, [], 1)[0] is True
    assert C.judge("pon:2z", ev, 5, [], 1)[0] is False
    chi = {"type": "chi", "actor": 1, "pai": "5p", "consumed": ["4p", "6p"]}
    assert C.judge("chi:5p", chi, 5, [], 1)[0] is True
    # 暗杠描述串只带一张，实际事件在 consumed 里
    ankan = {"type": "ankan", "actor": 1, "consumed": ["5mr", "5m", "5m", "5m"]}
    assert C.judge("ankan:5m", ankan, 5, [], 1)[0] is True
    assert C.judge("ankan:6m", ankan, 5, [], 1)[0] is False


def test_boundary_between_detects_kyoku_and_restore():
    assert C.boundary_between([{"type": "dahai"}, {"type": "start_kyoku"}], 0, 2)
    assert C.boundary_between([{"type": "dahai"}, {"type": "start_game"}], 0, 2)
    assert not C.boundary_between([{"type": "tsumo"}, {"type": "dahai"}], 0, 2)


def test_category_labels():
    assert C.category("dahai:1m") == "打牌" and C.category("none") == "跳过"
    assert C.category("reach") == "立直" and C.category("pon:1z") == "鸣牌"
    assert C.category("hora") == "和牌"