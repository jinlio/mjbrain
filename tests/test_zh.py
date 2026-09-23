"""advisor.zh 中文映射：覆盖 riichienv 实测词表（含赤5 'r' 后缀与 0m 两套记法）。"""

from __future__ import annotations

from advisor.zh import action_zh, tile_zh


def test_tile_numbered_and_honor():
    assert tile_zh("5s") == "5索"
    assert tile_zh("1m") == "1万"
    assert tile_zh("9p") == "9饼"
    assert tile_zh("E") == "东" and tile_zh("S") == "南" and tile_zh("W") == "西"
    assert tile_zh("N") == "北" and tile_zh("P") == "白" and tile_zh("F") == "发"
    assert tile_zh("C") == "中"


def test_red_five_both_notations():
    assert tile_zh("5mr") == "赤5万"
    assert tile_zh("5sr") == "赤5索"
    assert tile_zh("0p") == "赤5饼"


def test_actions_zh():
    assert action_zh("dahai:5s") == "切5索"
    assert action_zh("dahai:W") == "切西"
    assert action_zh("none") == "跳过"
    assert action_zh("pon:7s") == "碰7索"
    assert action_zh("chi:3s") == "吃3索"
    assert action_zh("daiminkan:1m") == "杠1万"
    assert action_zh("ankan:5mr") == "暗杠赤5万"
    assert action_zh("kakan:2p") == "加杠2饼"
    assert action_zh("hora") == "和"
    assert action_zh("tsumo") == "自摸"
    assert action_zh("reach") == "立直"


def test_unknown_passthrough():
    assert tile_zh("?") == "?"
    assert tile_zh("") == ""
    assert action_zh("weird_thing") == "weird_thing"