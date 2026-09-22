"""雀魂牌字符串 ↔ mjai 牌字符串 + mjai 规范序。

移植自 Akagi v3 `src/bridge/majsoul/tile.rs`（Apache-2.0，见 LICENSES.md）。
雀魂侧 `0m/0p/0s` 为赤牌；`1z..7z` 字牌。mjai 侧赤牌 `5mr/5pr/5sr`，
字牌 `E S W N P F C`，`?` 为未知牌占位（他家摸牌不可见时）。

未知输入一律抛错：让畸形的 liqi payload 显式失败，而不是悄悄污染 mjai 流。
"""

from __future__ import annotations

_MS_TO_MJAI: dict[str, str] = {}
for _suit in ("m", "p", "s"):
    for _n in range(1, 10):
        _MS_TO_MJAI[f"{_n}{_suit}"] = f"{_n}{_suit}"
    _MS_TO_MJAI[f"0{_suit}"] = f"5{_suit}r"
for _n, _z in zip(range(1, 8), ("E", "S", "W", "N", "P", "F", "C")):
    _MS_TO_MJAI[f"{_n}z"] = _z

UNKNOWN_TILE = "?"


class UnknownTileError(ValueError):
    """payload 里出现未知雀魂牌串。"""


def ms_to_mjai(ms: str) -> str:
    """雀魂牌串 → mjai 牌串；未知输入抛 UnknownTileError。"""
    try:
        return _MS_TO_MJAI[ms]
    except KeyError:
        raise UnknownTileError(f"unknown majsoul tile: {ms!r}") from None


# mjai 规范序（小到大）；未知串排在 `?` 之后一位
_ORDER: tuple[str, ...] = (
    "1m", "2m", "3m", "4m", "5mr", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5pr", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5sr", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C", "?",
)
_RANK = {t: i for i, t in enumerate(_ORDER)}
_LAST = len(_ORDER)


def pai_rank(pai: str) -> int:
    return _RANK.get(pai, _LAST)


def pai_lt(a: str, b: str) -> bool:
    """compare_pai(a,b) < 0 的等价判定。"""
    return pai_rank(a) < pai_rank(b)


def sort_pais(pais: list[str]) -> list[str]:
    """按 mjai 规范序稳定排序（dahai 后手牌升序用）。"""
    return sorted(pais, key=pai_rank)


def pai_has_red_form(pai: str) -> bool:
    """数牌 5（m/p/s）才有赤牌形态；字牌与非 5 没有（kakan/ankan 补牌判定用）。"""
    return pai in ("5m", "5p", "5s")
