"""雀魂 liqi 事件流 → mjai 事件流状态机（M4 档1 状态层）。

移植自 Akagi v3 `src/bridge/majsoul/mod.rs`（Apache-2.0，见 LICENSES.md），
剔除其 autoplay 专属部分（time_budget 决策窗、input_watch 点击验证、flow/session
日志）——只读建议形态不注入也不测延迟。

输入契约：capture/liqi.decode.LiqiParser 的 Frame（payload 已是 JSON dict，
ActionPrototype 的 data 经 maybe_decode_action 二次解码为动作 dict；GameRestore
内嵌动作经 decode_restore_action，纯 base64 免 XOR）。

与 Rust 原版的有意偏差：原版用 prost 序列化，标量字段恒在；我们走 protobuf
MessageToDict，proto3 未设置的标量会被整个省略（seat=0、moqie=False 等）。
因此凡原版 `.context("missing")` 处，此处按语义默认值 `.get(key, default)`
处理——"缺省"在本序列化下是合法值，不是协议畸形。repeated 字段行为一致
（空数组同样被省略，原版 `.get` 后判长度，两边等价）。

事件 dict 采用 mjai 标准型（type 字段 + 各事件载荷），与 data/ 层既有
tenhou→mjai 约定同构，下游 brain/review 零适配。
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from capture.mjai.tile import (
    UNKNOWN_TILE,
    ms_to_mjai,
    pai_has_red_form,
    sort_pais,
)

log = logging.getLogger("mjbrain.capture.mjai")

# —— 协议常量（mod.rs 头部）——
METHOD_AUTH_GAME = ".lq.FastTest.authGame"
METHOD_ACTION_PROTOTYPE = "lq.ActionPrototype"
METHOD_NOTIFY_GAME_END_RESULT = "lq.NotifyGameEndResult"
METHOD_NOTIFY_GAME_TERMINATE = "lq.NotifyGameTerminate"
METHOD_SYNC_GAME = ".lq.FastTest.syncGame"
METHOD_ENTER_GAME = ".lq.FastTest.enterGame"

ACTION_NEW_ROUND = "ActionNewRound"
ACTION_DEAL_TILE = "ActionDealTile"
ACTION_DISCARD_TILE = "ActionDiscardTile"
ACTION_CHI_PENG_GANG = "ActionChiPengGang"
ACTION_AN_GANG_ADD_GANG = "ActionAnGangAddGang"
ACTION_HULE = "ActionHule"
ACTION_NO_TILE = "ActionNoTile"
ACTION_LIU_JU = "ActionLiuJu"
ACTION_BA_BEI = "ActionBaBei"

CHI_PENG_GANG_CHI, CHI_PENG_GANG_PENG, CHI_PENG_GANG_GANG = 0, 1, 2
AN_GANG_ADD_GANG_AN, AN_GANG_ADD_GANG_ADD = 3, 2
TEHAI_SIZE, TSUMO_TEHAI_SIZE = 13, 14

PENDING_BEFORE_RINSHAN = "pending_before_rinshan"  # ankan 即乗り
PENDING_AFTER_RINSHAN = "pending_after_rinshan"    # kakan/daiminkan 後乗り


def stable_game_id(game_uuid: str) -> int:
    """FNV-1a 64bit：录制器重连去重键（原始 uuid 另存 MatchInfo）。"""
    h = 0xCBF29CE484222325
    for b in game_uuid.encode():
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _ms_tiles(arr: list) -> list[str]:
    return [ms_to_mjai(s) for s in arr]


def ankan_consumed(pai: str) -> list[str]:
    """mjai 赤牌规则：杠 4 枚至多 1 赤，赤置 index 0。"""
    normal = pai.rstrip("r")
    out = [normal] * 4
    if pai_has_red_form(normal):
        out[0] = normal + "r"
    return out


def kakan_consumed(pai: str) -> list[str]:
    """已有 pon 的 3 枚；pai 是新加的那枚（kakan 事件另行携带）。"""
    normal = pai.rstrip("r")
    out = [normal] * 3
    if pai_has_red_form(normal) and not pai.endswith("r"):
        out[0] = normal + "r"
    return out


def parse_scores(data: dict) -> list[int]:
    arr = data.get("scores")
    if arr is None:
        raise ValueError("ActionNewRound missing scores")
    return [int(v) for v in arr]


def parse_deltas(arr: list) -> list[int]:
    return [int(v) for v in arr]


def sum_delta_scores(data: dict, num_players: int) -> list[int] | None:
    """ActionNoTile 的多条 NoTileScoreInfo.delta_scores 按座求和。

    无任何条目 → None（无分数变动，与显式全 0 区分）；短数组零补齐（防御）。
    """
    arr = data.get("scores")
    if not arr:
        return None
    n = num_players
    total = [0] * n
    for entry in arr:
        deltas = entry.get("delta_scores")
        if not deltas:
            continue
        for i, v in enumerate(deltas[:n]):
            total[i] += int(v)
    return total


def fill_seat_row(tehais: list[list[str]], seat: int, row: list[str]) -> None:
    if seat >= len(tehais):
        raise ValueError(f"seat {seat} out of range")
    if len(row) != TEHAI_SIZE:
        raise ValueError(f"expected {TEHAI_SIZE} tiles for seat row, got {len(row)}")
    tehais[seat] = row


def names_from_payload(payload: dict, seat_list: list) -> list[str]:
    nick = {
        int(p["account_id"]): str(p.get("nickname", ""))
        for p in payload.get("players") or []
        if p.get("account_id") is not None
    }
    return [nick.get(int(v), "") for v in seat_list]


def parse_game_end_standings(payload: dict, num_players: int):
    """result.players 按名次序 → (seat 序 scores, seat 序 ranks) 或 None。"""
    n = num_players
    if n not in (3, 4):
        return None
    players = (payload.get("result") or {}).get("players")
    if not players or len(players) != n:
        return None
    scores: list[int | None] = [None] * n
    ranks: list[int | None] = [None] * n
    for rank0, p in enumerate(players):
        seat = int(p.get("seat", -1))
        if seat < 0 or seat >= n or scores[seat] is not None:
            return None
        scores[seat] = int(p.get("part_point_1", 0))
        ranks[seat] = rank0 + 1
    if any(s is None for s in scores):
        return None
    return [int(s) for s in scores], [int(r) for r in ranks]


class MajsoulState:
    """per-WS 连接的镜像态（Akagi：每条流各持一个实例）。

    实盘 ActionPrototype 的 `data` 由 capture/liqi 的 LiqiParser 在帧层已二次
    解码为 dict；GameRestore 内嵌动作经注入的 `decode_restore(name, b64)`
    （capture.liqi.runtime 提供；描述符映射留 liqi 层，状态层保持纯逻辑可单测）。
    """

    def __init__(
        self,
        *,
        decode_restore: Callable[[str, str], dict] | None = None,
    ):
        self._decode_restore = decode_restore
        self.account_id: int | None = None
        self.game_uuid: str | None = None
        self.game_id: int | None = None
        self.seat: int | None = None
        self.num_players: int = 4
        self.doras: list[str] = []
        self.dora_timing: str | None = None
        self.deferred_doras: list[str] = []
        self.last_revealed_tile_actor: int | None = None
        self.pending_reach_accepted: int | None = None

    # ---------- 入口 ----------

    def dispatch(self, msg_type: int, method: str, payload: dict) -> list[dict]:
        """Frame → mjai 事件列表。msg_type 用 capture.liqi.decode 的 NOTIFY/REQUEST/RESPONSE。"""
        from capture.liqi.decode import NOTIFY, REQUEST, RESPONSE

        if method == METHOD_AUTH_GAME:
            if msg_type == REQUEST:
                acc = payload.get("account_id")
                self.account_id = int(acc) if acc is not None else None
                uuid = payload.get("game_uuid")
                self.game_uuid = uuid if isinstance(uuid, str) else None
                self.game_id = stable_game_id(self.game_uuid) if self.game_uuid else None
                return []
            if msg_type == RESPONSE:
                return self.handle_auth_game_response(payload)
            return []
        if msg_type == NOTIFY and method == METHOD_ACTION_PROTOTYPE:
            return self.handle_action_prototype(payload)
        if msg_type == RESPONSE and method in (METHOD_SYNC_GAME, METHOD_ENTER_GAME):
            return self.handle_game_restore(payload)
        if msg_type == NOTIFY and method == METHOD_NOTIFY_GAME_END_RESULT:
            standings = parse_game_end_standings(payload, self.num_players)
            scores, ranks = standings if standings else (None, None)
            return [
                {"type": "end_game", "reached_end": False, "names": None,
                 "scores": scores, "ranks": ranks, "over": False}
            ]
        if msg_type == NOTIFY and method == METHOD_NOTIFY_GAME_TERMINATE:
            return [{"type": "end_game", "reached_end": False, "over": False,
                     "terminated": True}]
        return []

    def handle_auth_game_response(self, payload: dict) -> list[dict]:
        if self.account_id is None:
            log.warning("authGame response without request — seat unresolved")
            return []
        seat_list = payload.get("seat_list")
        if not seat_list:
            log.warning("authGame response missing seat_list")
            return []
        ids = [int(v) for v in seat_list]
        if self.account_id not in ids:
            log.warning("account_id %s not in seat_list %s", self.account_id, ids)
            return []
        self.seat = ids.index(self.account_id)
        detected = len(ids)
        if detected not in (3, 4):
            log.warning("unexpected seat_list length %s; defaulting num_players=4", detected)
            detected = 4
        self.num_players = detected
        meta = (payload.get("game_config") or {}).get("meta") or {}
        match = (payload.get("game_config") or {}).get("mode") or {}
        u32 = lambda k: (meta.get(k) or None) or None  # 0 → None，同 Akagi filter
        return [{
            "type": "start_game",
            "names": names_from_payload(payload, seat_list),
            "id": str(self.seat),  # mjai/riichienv 要字符串（Akagi Rust 侧整数 seat 在此转换）
            "num_players": self.num_players,
            "meta": {
                "game_id": self.game_id,
                "game_uuid": self.game_uuid,
                "mode_id": u32("mode_id"),
                "room_id": u32("room_id"),
                "contest_uid": u32("contest_uid"),
                "match_mode": match.get("mode"),
            },
        }]

    # ---------- ActionPrototype ----------

    def handle_action_prototype(self, payload: dict) -> list[dict]:
        name = payload.get("name")
        data = payload.get("data")
        if not isinstance(name, str) or data is None:
            log.warning("ActionPrototype missing name/data: %s", str(payload)[:200])
            return []
        # 胡牌/自家立直宣言牌被抢时，挂起的 reach_accepted 不清发（见 mod.rs 注释）
        pending_reach: int | None = None
        if name == ACTION_HULE:
            self.pending_reach_accepted = None
        elif name != ACTION_DISCARD_TILE:
            pending_reach = self.pending_reach_accepted
            self.pending_reach_accepted = None

        handler = {
            ACTION_NEW_ROUND: self.build_start_kyoku,
            ACTION_DEAL_TILE: self.build_tsumo,
            ACTION_DISCARD_TILE: self.build_dahai,
            ACTION_CHI_PENG_GANG: self.build_chi_peng_gang,
            ACTION_AN_GANG_ADD_GANG: self.build_an_gang_add_gang,
            ACTION_NO_TILE: self.build_no_tile,
            ACTION_LIU_JU: self.build_liu_ju,
            ACTION_HULE: self.build_hule,
            ACTION_BA_BEI: self.build_kita,
        }.get(name)
        events: list[dict] = []
        if handler is not None:
            try:
                events = handler(data)
            except Exception as e:  # noqa: BLE001 —— 单事件转换失败不断流，同 Rust
                log.warning("%s → mjai conversion failed: %s", name, e)
        if pending_reach is not None:
            events = [{"type": "reach_accepted", "actor": pending_reach}, *events]
        return events

    def consume_new_doras(self, data: dict) -> list[str]:
        """对比服务端 doras 数组与本方已报标记，返回新增（并同步全量尾）。

        全尾同步而非只取末位：本地列表绝不能落后服务端，否则后续未增长的
        数组仍比本地长导致重复发标（Akagi issue #244 双 9m 教训）。前缀被
        改写只 warn（mjai 无法撤回已发标记）。
        """
        arr = data.get("doras")
        if not arr:
            return []
        mapped = _ms_tiles(arr)
        prefix = min(len(self.doras), len(mapped))
        if self.doras[:prefix] != mapped[:prefix]:
            log.warning("server dora array %s does not extend reported %s; adopting tail",
                        mapped, self.doras)
        if len(mapped) <= len(self.doras):
            return []
        new_markers = mapped[len(self.doras):]
        self.doras = mapped
        return new_markers

    def build_tsumo(self, data: dict) -> list[dict]:
        if self.seat is None:
            raise ValueError("seat unresolved at ActionDealTile")
        actor = int(data.get("seat", 0))
        tile_raw = data.get("tile") or ""
        pai = ms_to_mjai(tile_raw) if (actor == self.seat and tile_raw) else UNKNOWN_TILE

        new_markers = self.consume_new_doras(data)
        timing = self.dora_timing
        self.dora_timing = None
        events: list[dict] = []
        if timing == PENDING_AFTER_RINSHAN:
            # kakan/daiminkan 後乗り：岭打 tsumo 先行，标记（若有）压待提交
            self.deferred_doras.extend(new_markers)
            events.append({"type": "tsumo", "actor": actor, "pai": pai})
        elif timing == PENDING_BEFORE_RINSHAN:
            # ankan 即乗り：先翻宝牌再岭打摸
            if not new_markers:
                log.warning("rinshan deal after ankan missing new dora marker")
            events.extend({"type": "dora", "dora_marker": m} for m in new_markers)
            events.append({"type": "tsumo", "actor": actor, "pai": pai})
        else:
            for m in new_markers:  # 非杠流程的意外宝牌增长：warn 后 tsumo 前发
                log.warning("new dora marker %s without preceding kan", m)
            events.extend({"type": "dora", "dora_marker": m} for m in new_markers)
            events.append({"type": "tsumo", "actor": actor, "pai": pai})
        return events

    def build_dahai(self, data: dict) -> list[dict]:
        actor = int(data.get("seat", 0))  # 庄家首巡省略 seat → 0，同 Rust
        tile_raw = data.get("tile")
        if not tile_raw:
            raise ValueError("ActionDiscardTile.tile is empty")
        pai = ms_to_mjai(tile_raw)
        tsumogiri = bool(data.get("moqie", False))
        is_riichi = bool(data.get("is_liqi", False)) or bool(data.get("is_wliqi", False))

        dora_markers = self.deferred_doras
        self.deferred_doras = []
        dora_markers.extend(self.consume_new_doras(data))

        events: list[dict] = [{"type": "dora", "dora_marker": m} for m in dora_markers]
        if is_riichi:
            events.append({"type": "reach", "actor": actor})
        events.append({"type": "dahai", "actor": actor, "pai": pai,
                       "tsumogiri": tsumogiri})
        if is_riichi:
            self.pending_reach_accepted = actor
        self.last_revealed_tile_actor = actor
        return events

    def build_chi_peng_gang(self, data: dict) -> list[dict]:
        actor = int(data["seat"])
        kind = int(data["type"])
        tiles = data["tiles"]
        froms = data["froms"]
        if len(tiles) != len(froms):
            raise ValueError(f"tiles/froms length mismatch {len(tiles)} vs {len(froms)}")
        target, pai, consumed = actor, "", []
        for tile_raw, from_seat in zip(tiles, froms):
            tile = ms_to_mjai(tile_raw)
            if int(from_seat) == actor:
                consumed.append(tile)
            else:
                target = int(from_seat)
                pai = tile
        if target == actor:
            raise ValueError("no foreign seat in froms")
        if not pai:
            raise ValueError("target tile not found")
        if kind == CHI_PENG_GANG_CHI:
            if self.num_players == 3:
                log.warning("ActionChiPengGang(chi) in 3p flow; dropping")
                return []
            if len(consumed) != 2:
                raise ValueError(f"chi expects 2 consumed, got {len(consumed)}")
            return [{"type": "chi", "actor": actor, "target": target,
                     "pai": pai, "consumed": consumed}]
        if kind == CHI_PENG_GANG_PENG:
            if len(consumed) != 2:
                raise ValueError(f"pon expects 2 consumed, got {len(consumed)}")
            return [{"type": "pon", "actor": actor, "target": target,
                     "pai": pai, "consumed": consumed}]
        if kind == CHI_PENG_GANG_GANG:
            if len(consumed) != 3:
                raise ValueError(f"daiminkan expects 3 consumed, got {len(consumed)}")
            self.dora_timing = PENDING_AFTER_RINSHAN  # 後乗り
            return [{"type": "daiminkan", "actor": actor, "target": target,
                     "pai": pai, "consumed": consumed}]
        raise ValueError(f"unknown ActionChiPengGang.type: {kind}")

    def build_an_gang_add_gang(self, data: dict) -> list[dict]:
        actor = int(data["seat"])
        kind = int(data["type"])
        pai = ms_to_mjai(data["tiles"])  # 单数串：ankan=杠体，kakan=新加枚
        new_markers = self.consume_new_doras(data)
        events: list[dict] = []
        if kind == AN_GANG_ADD_GANG_AN:
            events.append({"type": "ankan", "actor": actor,
                           "consumed": ankan_consumed(pai)})
            if new_markers:
                events.extend({"type": "dora", "dora_marker": m} for m in new_markers)
            else:
                self.dora_timing = PENDING_BEFORE_RINSHAN  # 即乗り：标记随岭打
        elif kind == AN_GANG_ADD_GANG_ADD:
            self.deferred_doras.extend(new_markers)
            self.dora_timing = PENDING_AFTER_RINSHAN
            events.append({"type": "kakan", "actor": actor, "pai": pai,
                           "consumed": kakan_consumed(pai)})
        else:
            raise ValueError(f"unknown ActionAnGangAddGang.type: {kind}")
        # 两种杠都亮出可被抢的牌（kakan→抢槓；ankan→国士抢暗杠）
        self.last_revealed_tile_actor = actor
        return events

    def build_no_tile(self, data: dict) -> list[dict]:
        deltas = sum_delta_scores(data, self.num_players)
        return [{"type": "ryukyoku", "deltas": deltas}, {"type": "end_kyoku"}]

    def build_liu_ju(self, data: dict) -> list[dict]:
        return [{"type": "ryukyoku", "deltas": None}, {"type": "end_kyoku"}]

    def build_hule(self, data: dict) -> list[dict]:
        hules = data.get("hules")
        if not hules:
            raise ValueError("ActionHule with empty hules")
        deltas = parse_deltas(data["delta_scores"]) if data.get("delta_scores") else None
        events: list[dict] = []
        for hule in hules:
            actor = int(hule.get("seat", 0))
            zimo = bool(hule.get("zimo", False))
            if zimo:
                target = actor
            else:
                if self.last_revealed_tile_actor is None:
                    raise ValueError("ron win without preceding tile-revealing action")
                target = self.last_revealed_tile_actor
            ura = None
            if bool(hule.get("liqi", False)):
                ura = _ms_tiles(hule.get("li_doras") or [])  # 空数组→Some([])
            events.append({"type": "hora", "actor": actor, "target": target,
                           "deltas": deltas, "ura_markers": ura})
        events.append({"type": "end_kyoku"})
        return events

    def build_kita(self, data: dict) -> list[dict]:
        actor = int(data["seat"])
        if self.num_players != 3:
            log.warning("ActionBaBei received in %sp flow", self.num_players)
        dora_markers = self.deferred_doras
        self.deferred_doras = []
        dora_markers.extend(self.consume_new_doras(data))
        events = [{"type": "dora", "dora_marker": m} for m in dora_markers]
        self.last_revealed_tile_actor = actor
        events.append({"type": "kita", "actor": actor, "pai": "N"})
        return events

    def build_start_kyoku(self, data: dict) -> list[dict]:
        seat = self.seat
        if seat is None:
            raise ValueError("seat unresolved at ActionNewRound")
        chang = int(data.get("chang", 0))
        ju = int(data.get("ju", 0))
        ben = int(data.get("ben", 0))
        liqibang = int(data.get("liqibang", 0))
        bakaze = ["E", "S", "W", "N"][chang] if chang < 4 else None
        if bakaze is None:
            raise ValueError(f"invalid chang value: {chang}")
        doras = data.get("doras")
        if not doras:
            raise ValueError("ActionNewRound missing doras[0]")
        dora_marker = ms_to_mjai(doras[0])
        scores = parse_scores(data)
        tiles_raw = data.get("tiles")
        if tiles_raw is None:
            raise ValueError("ActionNewRound missing tiles")
        my_tiles = _ms_tiles(tiles_raw)
        oya = ju
        n = self.num_players
        tehais: list[list[str]] = [[UNKNOWN_TILE] * TEHAI_SIZE for _ in range(n)]
        if len(my_tiles) == TEHAI_SIZE:
            # 非庄：我方 13 枚入位；庄的初摸对我不现
            fill_seat_row(tehais, seat, sort_pais(my_tiles))
            if oya == seat:
                raise ValueError("dealer must receive 14 tiles, got 13")
            tsumo_event = {"type": "tsumo", "actor": oya, "pai": UNKNOWN_TILE}
        elif len(my_tiles) == TSUMO_TEHAI_SIZE:
            if oya != seat:
                raise ValueError("non-dealer must receive 13 tiles, got 14")
            fill_seat_row(tehais, seat, sort_pais(my_tiles[:TEHAI_SIZE]))
            tsumo_event = {"type": "tsumo", "actor": seat, "pai": my_tiles[TEHAI_SIZE]}
        else:
            raise ValueError(
                f"unexpected tile count {len(my_tiles)} (expected 13 or 14)")
        # 新局：重置宝牌/立直/弃牌簿记，防上局 deferred 渗漏
        self.doras = [dora_marker]
        self.dora_timing = None
        self.deferred_doras = []
        self.last_revealed_tile_actor = None
        self.pending_reach_accepted = None
        return [{
            "type": "start_kyoku", "bakaze": bakaze, "dora_marker": dora_marker,
            "kyoku": oya + 1, "honba": ben, "kyotaku": liqibang, "oya": oya,
            "scores": scores, "tehais": tehais, "num_players": n,
        }, tsumo_event]

    # ---------- GameRestore 重演 ----------

    def handle_game_restore(self, payload: dict) -> list[dict]:
        """断线重连：重放 game_restore.actions[]（当前局全动作，免 XOR）。

        不发 start_game（前序 authGame 已是唯一重置点）；重放末尾挂起的
        reach_accepted 由首个后续实盘动作按常规消化。snapshot 字段忽略——
        动作重放本身重建全态。
        """
        actions = (payload.get("game_restore") or {}).get("actions")
        if not actions:
            return []
        if self._decode_restore is None:
            raise ValueError("handle_game_restore requires decode_restore callable")
        events: list[dict] = []
        for action in actions:
            name, b64 = action.get("name"), action.get("data")
            if not isinstance(name, str) or not isinstance(b64, str):
                log.warning("GameRestore action missing name/data: %s", str(action)[:120])
                continue
            try:
                decoded = self._decode_restore(name, b64)
            except Exception as e:  # noqa: BLE001
                log.warning("failed to decode GameRestore action %s: %s", name, e)
                continue
            events.extend(self.handle_action_prototype({"name": name, "data": decoded}))
        return events
