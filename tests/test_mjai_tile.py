"""capture/mjai/tile.py 单测——镜像 tile.rs 的 5 项 Rust 测试 + 全表对拍。"""

from capture.mjai.tile import (
    UNKNOWN_TILE,
    UnknownTileError,
    ms_to_mjai,
    pai_rank,
    sort_pais,
)

# tile.rs 权威映射全表（34 雀魂串 → mjai），逐条对拍
_FULL_TABLE = {
    "1m": "1m", "2m": "2m", "3m": "3m", "4m": "4m", "5m": "5m",
    "6m": "6m", "7m": "7m", "8m": "8m", "9m": "9m", "0m": "5mr",
    "1p": "1p", "2p": "2p", "3p": "3p", "4p": "4p", "5p": "5p",
    "6p": "6p", "7p": "7p", "8p": "8p", "9p": "9p", "0p": "5pr",
    "1s": "1s", "2s": "2s", "3s": "3s", "4s": "4s", "5s": "5s",
    "6s": "6s", "7s": "7s", "8s": "8s", "9s": "9s", "0s": "5sr",
    "1z": "E", "2z": "S", "3z": "W", "4z": "N", "5z": "P", "6z": "F", "7z": "C",
}


def test_full_mapping_table():
    assert len(_FULL_TABLE) == 37  # 27 数牌 + 3 赤 + 7 字
    for ms, mj in _FULL_TABLE.items():
        assert ms_to_mjai(ms) == mj, ms


def test_red_fives_round_trip():
    assert ms_to_mjai("0m") == "5mr"
    assert ms_to_mjai("0p") == "5pr"
    assert ms_to_mjai("0s") == "5sr"


def test_honors_map_to_letters():
    assert ms_to_mjai("1z") == "E"
    assert ms_to_mjai("7z") == "C"


def test_unknown_tile_errors():
    for bad in ("8z", "garbage", "10m", ""):
        try:
            ms_to_mjai(bad)
        except UnknownTileError:
            continue
        raise AssertionError(f"{bad!r} should raise")


def test_red_five_sorts_before_normal_five():
    assert pai_rank("5mr") < pai_rank("5m")
    assert pai_rank("5pr") < pai_rank("5p")
    assert pai_rank("5sr") < pai_rank("5s")


def test_full_sort_matches_canonical_order():
    tiles = ["C", "1m", "5m", "5mr", "9p", "E", UNKNOWN_TILE, "3s"]
    assert sort_pais(tiles) == ["1m", "5mr", "5m", "9p", "3s", "E", "C", "?"]


def test_unknown_string_sorts_last():
    assert pai_rank("zz") > pai_rank("?")
