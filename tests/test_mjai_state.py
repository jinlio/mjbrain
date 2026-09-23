"""capture/mjai/state.py 行为测试——镜像 mod.rs 的语义不变量。

payload 全部手工构造为 MessageToDict 形状（proto3 缺省字段省略！这是与
prost 序列化的关键差异，测试特意在 seat=0/moqie=False 处用省略形态验证
语义默认）。
"""

from capture.liqi.decode import NOTIFY, REQUEST, RESPONSE
from capture.mjai.state import MajsoulState

ACCOUNT = 100001
OTHERS = [100002, 100003, 100004]


def _authed_state(seat=1):
    st = MajsoulState()
    st.account_id = ACCOUNT
    st.game_uuid = "u-abc"
    st.game_id = 123
    seat_list = [100002, ACCOUNT, 100003, 100004]
    ev = st.dispatch(RESPONSE, ".lq.FastTest.authGame", {
        "seat_list": seat_list,
        "players": [
            {"account_id": ACCOUNT, "nickname": "me"},
            {"account_id": 100004, "nickname": "d"},
        ],
        "game_config": {"meta": {"mode_id": 5, "room_id": 0}, "mode": {"mode": 2}},
    })
    assert ev[0]["type"] == "start_game" and ev[0]["id"] == str(seat)
    return st


def _proto(name, data):
    return {"step": 0, "name": name, "data": data}


def _events(st, name, data):
    return st.handle_action_prototype(_proto(name, data))


# ---------- authGame ----------

def test_auth_request_captures_identity():
    st = MajsoulState()
    assert st.dispatch(REQUEST, ".lq.FastTest.authGame",
                       {"account_id": ACCOUNT, "game_uuid": "u-1"}) == []
    assert st.account_id == ACCOUNT
    assert st.game_uuid == "u-1"
    assert st.game_id is not None  # FNV 稳定非零
    assert MajsoulState().game_id is None


def test_start_game_names_and_meta():
    st = _authed_state()
    ev = st.handle_auth_game_response({
        "seat_list": [100002, ACCOUNT, 100003, 100004],
        "players": [
            {"account_id": ACCOUNT, "nickname": "me"},
            {"account_id": 100004, "nickname": "d"},
        ],
        "game_config": {"meta": {"mode_id": 5, "room_id": 0}, "mode": {"mode": 2}},
    })
    assert ev[0]["names"] == ["", "me", "", "d"]  # 机器人无名 → 空串
    assert ev[0]["meta"]["mode_id"] == 5
    assert ev[0]["meta"]["room_id"] is None  # 0 → None（同 Akagi filter）


def test_auth_response_without_request_is_noop():
    assert MajsoulState().dispatch(RESPONSE, ".lq.FastTest.authGame",
                                   {"seat_list": [1, 2, 3, 4]}) == []


# ---------- start_kyoku ----------

_NEW_ROUND_BASE = {"chang": 0, "ju": 1, "ben": 0, "liqibang": 0,
                   "scores": [25000, 25000, 25000, 25000],
                   "doras": ["5m"], "dora": "5m"}


def test_start_kyoku_non_dealer_13_tiles():
    st = _authed_state(seat=1)  # oya=ju=1 → 我方是庄？换 seat 0 测非庄
    st.seat = 0
    tiles = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
             "1p", "0p", "1s", "1z"]
    ev = _events(st, "ActionNewRound", dict(_NEW_ROUND_BASE, tiles=tiles))
    sk = ev[0]
    assert sk["type"] == "start_kyoku"
    assert sk["bakaze"] == "E" and sk["kyoku"] == 2 and sk["oya"] == 1
    assert sk["dora_marker"] == "5m"
    assert sk["tehais"][0] == sorted_tiles(tiles)
    assert sk["tehais"][1] == ["?"] * 13
    assert ev[1] == {"type": "tsumo", "actor": 1, "pai": "?"}


def test_start_kyoku_dealer_14_tiles():
    st = _authed_state()
    st.seat = 1
    tiles = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
             "1p", "5p", "1s", "1z", "9s"]
    ev = _events(st, "ActionNewRound", dict(_NEW_ROUND_BASE, tiles=tiles))
    assert ev[0]["tehais"][1][:2] == ["1m", "2m"]
    assert ev[1] == {"type": "tsumo", "actor": 1, "pai": "9s"}


def sorted_tiles(ms_tiles):
    from capture.mjai.tile import ms_to_mjai, sort_pais
    return sort_pais([ms_to_mjai(t) for t in ms_tiles])


def test_start_kyoku_rejects_count_mismatch():
    st = _authed_state()
    st.seat = 0
    assert _events(st, "ActionNewRound", dict(_NEW_ROUND_BASE, tiles=["1m"] * 12)) == []


def test_new_kyoku_resets_deferred_doras():
    st = _authed_state()
    st.seat = 0  # oya=1，我方非庄 → 13 枚合法
    st.deferred_doras = ["9p"]  # 上局残留
    st.last_revealed_tile_actor = 2
    _events(st, "ActionNewRound", dict(_NEW_ROUND_BASE, tiles=["1m"] * 13))
    assert st.deferred_doras == [] and st.last_revealed_tile_actor is None


# ---------- tsumo / dahai ----------

def test_tsumo_unknown_for_other_seats():
    st = _authed_state()
    ev = _events(st, "ActionDealTile", {"seat": 2, "tile": "5p"})
    assert ev == [{"type": "tsumo", "actor": 2, "pai": "?"}]


def test_tsumo_own_draw_revealed():
    st = _authed_state()  # seat=1
    ev = _events(st, "ActionDealTile", {"seat": 1, "tile": "0s"})
    assert ev == [{"type": "tsumo", "actor": 1, "pai": "5sr"}]


def test_dahai_missing_seat_defaults_to_zero():
    st = _authed_state()
    # MessageToDict 省略 seat=0 / moqie=false —— proto3 缺省形态
    ev = _events(st, "ActionDiscardTile", {"tile": "1m"})
    assert ev == [{"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False}]


def test_riichi_queues_reach_accepted():
    st = _authed_state()
    ev = _events(st, "ActionDiscardTile", {"seat": 1, "tile": "9p", "is_liqi": True})
    assert ev == [{"type": "reach", "actor": 1},
                  {"type": "dahai", "actor": 1, "pai": "9p", "tsumogiri": False}]
    # 下一正常动作（摸牌）先吐 reach_accepted
    ev2 = _events(st, "ActionDealTile", {"seat": 2})
    assert ev2[0] == {"type": "reach_accepted", "actor": 1}
    assert ev2[1]["type"] == "tsumo"


def test_ron_on_declaration_tile_voids_riichi():
    st = _authed_state()
    _events(st, "ActionDiscardTile", {"seat": 1, "tile": "9p", "is_liqi": True})
    ev = _events(st, "ActionHule", {
        "hules": [{"seat": 2, "zimo": False}],
        "delta_scores": [-1000, -26000, 27000, 0]})
    assert all(e["type"] != "reach_accepted" for e in ev)
    assert st.pending_reach_accepted is None


def test_double_riichi_maps_wliqi():
    st = _authed_state()
    ev = _events(st, "ActionDiscardTile", {"seat": 3, "tile": "1z", "is_wliqi": True})
    assert ev[0]["type"] == "reach"


# ---------- 宝牌时机 ----------

def test_ankan_before_rinshan():
    st = _authed_state()
    st.doras = ["5m"]  # 模拟 start_kyoku 已播种（局内动作依赖该前置）
    ev = _events(st, "ActionAnGangAddGang", {"seat": 1, "type": 3, "tiles": "0p"})
    assert ev == [{"type": "ankan", "actor": 1,
                   "consumed": ["5pr", "5p", "5p", "5p"]}]
    # 岭打携带新标记 → dora 在 tsumo 前
    ev2 = _events(st, "ActionDealTile", {"seat": 1, "tile": "3s", "doras": ["5m", "9s"]})
    assert [e["type"] for e in ev2] == ["dora", "tsumo"]
    assert ev2[0]["dora_marker"] == "9s"


def test_kakan_after_rinshan_deferred_to_dahai():
    st = _authed_state()
    st.doras = ["5m"]  # 同上：start_kyoku 播种
    ev = _events(st, "ActionAnGangAddGang", {"seat": 1, "type": 2, "tiles": "5p"})
    assert ev == [{"type": "kakan", "actor": 1, "pai": "5p",
                   "consumed": ["5pr", "5p", "5p"]}]
    # 岭打先行，标记压 deferred
    ev2 = _events(st, "ActionDealTile", {"seat": 1, "doras": ["5m", "1p"]})
    assert [e["type"] for e in ev2] == ["tsumo"]
    assert st.deferred_doras == ["1p"]
    # 杠主 commit（dahai）时吐出
    ev3 = _events(st, "ActionDiscardTile", {"seat": 1, "tile": "7s", "moqie": True})
    assert [e["type"] for e in ev3] == ["dora", "dahai"]
    assert ev3[0]["dora_marker"] == "1p" and ev3[1]["tsumogiri"] is True


def test_daiminkan_sets_after_rinshan_timing():
    st = _authed_state()
    ev = _events(st, "ActionChiPengGang",
                 {"seat": 1, "type": 2, "tiles": ["5p", "0p", "5p", "5p"],
                  "froms": [1, 2, 1, 1]})
    assert ev[0]["type"] == "daiminkan"
    assert ev[0]["target"] == 2 and ev[0]["pai"] == "5pr"
    assert st.dora_timing == "pending_after_rinshan"


def test_consume_new_doras_no_double_emit():
    st = _authed_state()
    st.doras = ["5m"]
    # 未增长的数组 → 无新标记（#244 双 9m 回归）
    assert st.consume_new_doras({"doras": ["5m"]}) == []
    assert st.consume_new_doras({"doras": ["5m", "9p"]}) == ["9p"]
    assert st.doras == ["5m", "9p"]
    assert st.consume_new_doras({"doras": ["5m", "9p"]}) == []


def test_chankan_target_is_pon_tile_actor():
    st = _authed_state()
    _events(st, "ActionAnGangAddGang", {"seat": 2, "type": 3, "tiles": "3z"})
    ev = _events(st, "ActionHule", {
        "hules": [{"seat": 1, "zimo": False}],
        "delta_scores": [32000, -32000, 0, 0]})
    assert ev[0]["target"] == 2  # 国士抢暗杠挂在杠主身上


# ---------- 鸣牌 / kita / 流局 / 胡 ----------

def test_chi_pon_consumed_layout():
    st = _authed_state()
    ev = _events(st, "ActionChiPengGang",
                 {"seat": 1, "type": 0, "tiles": ["4p", "6p", "5p"],
                  "froms": [1, 1, 0]})
    assert ev[0]["pai"] == "5p" and ev[0]["target"] == 0
    assert ev[0]["consumed"] == ["4p", "6p"]
    ev2 = _events(st, "ActionChiPengGang",
                  {"seat": 1, "type": 1, "tiles": ["3s", "3s", "3s"], "froms": [1, 2, 1]})
    assert ev2[0]["consumed"] == ["3s", "3s"] and ev2[0]["target"] == 2


def test_mj_start_marker_is_silent_noop(caplog):
    """ActionMJStart 是空标记消息（proto 消息本身为空），非畸形：静默跳过、不告警。"""
    st = _authed_state()
    with caplog.at_level("WARNING", logger="mjbrain.capture.mjai"):
        assert _events(st, "ActionMJStart", None) == []
    assert "ActionPrototype missing" not in caplog.text
    # 真正的畸形（未知名字 + 缺 data）仍须告警
    with caplog.at_level("WARNING", logger="mjbrain.capture.mjai"):
        assert _events(st, "ActionWeird", None) == []
    assert "ActionPrototype missing" in caplog.text


def test_game_restore_skips_mj_start_marker(caplog):
    """GameRestore 里的 ActionMJStart 是空标记（连 data 键都没有），跳过且不告警。"""
    calls = []
    st = MajsoulState(decode_restore=lambda name, b64: calls.append(name) or {})
    with caplog.at_level("WARNING", logger="mjbrain.capture.mjai"):
        ev = st.handle_game_restore({"game_restore": {"actions": [
            {"name": "ActionMJStart", "step": 0},
        ]}})
    assert ev == [] and calls == []   # 标记不送去解码
    assert "GameRestore action missing" not in caplog.text


def test_omitted_zero_scalars_chi_and_pon():
    """proto3 零值省略：type=0（吃）、seat=0 时缺键即默认（run5 实测 payload 形状）。"""
    st = _authed_state()
    ev = _events(st, "ActionChiPengGang",
                 {"tiles": ["7s", "8s", "9s"], "froms": [0, 0, 1],
                  "tile_states": [0, 0]})   # seat=0、type=0 均省略
    assert ev == [{"type": "chi", "actor": 0, "target": 1,
                   "pai": "9s", "consumed": ["7s", "8s"]}]
    ev2 = _events(st, "ActionChiPengGang",
                  {"type": 1, "tiles": ["1z", "1z", "1z"], "froms": [0, 0, 3]})
    assert ev2 == [{"type": "pon", "actor": 0, "target": 3,
                    "pai": "E", "consumed": ["E", "E"]}]


def test_omitted_seat_zero_ankan_kakan_kita():
    st = _authed_state()
    st.doras = ["5m"]
    ev = _events(st, "ActionAnGangAddGang", {"type": 3, "tiles": "5m"})  # seat 省略
    assert ev == [{"type": "ankan", "actor": 0,
                   "consumed": ["5mr", "5m", "5m", "5m"]}]
    ev2 = _events(st, "ActionAnGangAddGang", {"type": 2, "tiles": "3z"})
    assert ev2 == [{"type": "kakan", "actor": 0, "pai": "W",
                    "consumed": ["W", "W", "W"]}]
    st.num_players = 3
    assert _events(st, "ActionBaBei", {}) == [{"type": "kita", "actor": 0, "pai": "N"}]


def test_call_missing_tiles_froms_drops_event():
    st = _authed_state()
    assert _events(st, "ActionChiPengGang",
                   {"seat": 1, "type": 1, "tiles": ["1z"]}) == []


def test_chi_dropped_in_sanma():
    st = _authed_state()
    st.num_players = 3
    assert _events(st, "ActionChiPengGang",
                   {"seat": 1, "type": 0, "tiles": ["4p", "6p", "5p"],
                    "froms": [1, 1, 0]}) == []


def test_kita_in_3p():
    st = _authed_state()
    st.num_players = 3
    ev = _events(st, "ActionBaBei", {"seat": 2})
    assert ev == [{"type": "kita", "actor": 2, "pai": "N"}]
    assert st.last_revealed_tile_actor == 2


def test_no_tile_sums_multi_entries():
    st = _authed_state()
    ev = _events(st, "ActionNoTile", {"liujumanguan": False, "scores": [
        {"seat": 0, "delta_scores": [1000, -1000, -1000, 1000]},
        {"seat": 1, "delta_scores": [4000, -4000, 0, 0]},
    ]})
    assert ev[0] == {"type": "ryukyoku", "deltas": [5000, -5000, -1000, 1000]}
    assert ev[1] == {"type": "end_kyoku"}
    # 无条目 → None（无变动 ≠ 全 0）
    assert _events(st, "ActionNoTile", {"liujumanguan": False})[0]["deltas"] is None


def test_liu_ju_generic_ryukyoku():
    st = _authed_state()
    ev = _events(st, "ActionLiuJu", {"type": 1, "seat": 0, "tiles": ["1z"] * 9})
    assert ev == [{"type": "ryukyoku", "deltas": None}, {"type": "end_kyoku"}]


def test_hule_multi_win_ura_and_target():
    st = _authed_state()
    _events(st, "ActionDiscardTile", {"seat": 2, "tile": "5m"})
    ev = _events(st, "ActionHule", {
        "hules": [
            {"seat": 1, "zimo": False, "liqi": True, "li_doras": ["1m"]},
            {"seat": 3, "zimo": False},
        ],
        "delta_scores": [-1000, 38000, -13000, -24000],
        "old_scores": [25000] * 4, "scores": [24000, 63000, 12000, 1000]})
    horas = [e for e in ev if e["type"] == "hora"]
    assert len(horas) == 2
    assert horas[0]["target"] == 2 and horas[0]["ura_markers"] == ["1m"]
    assert horas[1]["target"] == 2 and horas[1]["ura_markers"] is None
    assert horas[0]["deltas"] == [-1000, 38000, -13000, -24000]
    assert ev[-1] == {"type": "end_kyoku"}


def test_zimo_hora_targets_self():
    st = _authed_state()
    _events(st, "ActionDiscardTile", {"seat": 2, "tile": "5m"})  # 设干扰值
    ev = _events(st, "ActionHule", {
        "hules": [{"seat": 1, "zimo": True}],
        "delta_scores": [-1000, 38000, -13000, -24000]})
    assert ev[0]["target"] == 1  # zimo 自摸，target=actor


# ---------- end_game / restore ----------

def test_game_end_standings_to_seat_order():
    st = _authed_state()
    st.num_players = 4
    ev = st.dispatch(NOTIFY, "lq.NotifyGameEndResult", {
        "result": {"players": [
            {"seat": 3, "part_point_1": 60000},
            {"seat": 0, "part_point_1": 30000},
            {"seat": 2, "part_point_1": 10000},
            {"seat": 1, "part_point_1": -4000},
        ]}})
    assert ev[0]["scores"] == [30000, -4000, 10000, 60000]
    assert ev[0]["ranks"] == [2, 4, 3, 1]


def test_game_terminate_event():
    st = _authed_state()
    ev = st.dispatch(NOTIFY, "lq.NotifyGameTerminate", {})
    assert ev[0]["type"] == "end_game" and ev[0].get("terminated") is True


def test_game_restore_replay_with_decode_injection():
    calls = []

    def fake_decode_restore(name, b64):
        calls.append((name, b64))
        if name == "ActionNewRound":
            return dict(_NEW_ROUND_BASE, tiles=["1m"] * 13)
        if name == "ActionDiscardTile":
            return {"seat": 1, "tile": "9m", "moqie": True}
        return {}  # ActionMJStart：有消息无 handler → 解码成功、转换 no-op

    st = MajsoulState(decode_restore=fake_decode_restore)
    st.account_id = ACCOUNT
    st.dispatch(RESPONSE, ".lq.FastTest.authGame",
                {"seat_list": [ACCOUNT, 2, 3, 4]})
    st.seat = 0
    ev = st.dispatch(RESPONSE, ".lq.FastTest.syncGame", {"game_restore": {"actions": [
        {"name": "ActionNewRound", "data": "QUJD"},
        {"name": "ActionMJStart", "data": "WA=="},  # 无 handler → no-op
        {"name": "ActionDiscardTile", "data": "WmFk"},
    ], "passed_waiting_time": 12}})
    types = [e["type"] for e in ev]
    assert types == ["start_kyoku", "tsumo", "dahai"]
    assert calls == [("ActionNewRound", "QUJD"), ("ActionMJStart", "WA=="),
                     ("ActionDiscardTile", "WmFk")]


def test_unrouted_frames_are_noop():
    st = _authed_state()
    assert st.dispatch(NOTIFY, "lq.NotifyKeepAliveTick", {}) == []
    assert st.dispatch(REQUEST, ".lq.RoomTable.joinRoom", {}) == []
