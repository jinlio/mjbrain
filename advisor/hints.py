"""确定性牌况提示（HUD 进阶内容）：向听 / 待牌 / 宝牌——零注入、零副作用。

口径（与 eval.B1 同源，数字必须可解释）：
- 向听：门前清时 = 对全部合法弃牌逐张 ``riichienv.calculate_shanten`` 的最小值
  （"最优打法下的向听数"）。calculate_shanten 不收副露（Rust 签名只 hand_tiles），
  有副露的向听**不硬算**：shanten=None 如实降级，与 B1 的降级方向一致。
- 待牌：Rust ``HandEvaluator(tiles, melds)`` 支持副露——听牌线（弃后向听 0，
  副露则逐张弃牌试 is_tenpai）的 get_waits 并集；无弃牌选项的 13 张窗（如门前清
  荣和窗）直接按当前手牌算。非听牌 → []。
- 宝牌：``ob.dora_indicators``（含杠后第二排，指示牌列表随开杠变长）逐张 +1
  组内循环：数牌 1-9 回环、风 E→S→W→N、箭 白→发→中；dora_in_hand 按牌种计数。

一切只读 Observation 属性（与 brain.serialize.state_text 同源，座位遮蔽由引擎
保证）。hint_from_obs 任何意外 → None；_hint 内部再把向听/待牌与宝牌分段兜底，
一条算不出不连坐另一条。
"""

from __future__ import annotations

import riichienv.convert as cvt

_SUIT_LEN = 9  # 数牌每组 9 种（type 0..26），风 27..30，箭 31..33


def _dora_of(indicator_type: int) -> int:
    """指示牌 type(0..33) -> 宝牌 type（标准 +1 组内循环）。"""
    t = int(indicator_type)
    if t < 27:
        s = t // _SUIT_LEN
        return s * _SUIT_LEN + (t % _SUIT_LEN + 1) % _SUIT_LEN
    if t < 31:
        return 27 + (t - 27 + 1) % 4
    return 31 + (t - 31 + 1) % 3


def _type_name(t: int) -> str:
    """牌种 type -> mjai 名（非赤代表张；t*4 若撞赤5id 则 +1 规避）。"""
    tid = t * 4
    if tid in (16, 52, 88):  # tid_to_mjai 对赤5牌号输出 5mr，取同种非赤张
        tid += 1
    return cvt.tid_to_mjai(tid)


def _is_discard(a) -> bool:
    atype = str(getattr(a, "action_type", "")).split(".")[-1]
    return atype.upper() == "DISCARD"


def _discard_types(acts) -> list[int]:
    """合法弃牌动作的牌种（去重升序）。Action.tile 是物理 id（0..135）。"""
    out = set()
    for a in acts:
        if not _is_discard(a):
            continue
        try:
            tile = int(a.tile)
        except (AttributeError, TypeError, ValueError):
            continue
        if 0 <= tile < 136:
            out.add(tile // 4)
    return sorted(out)


def _one_of_type(hand: list[int], t: int) -> int | None:
    return next((h for h in hand if h // 4 == t), None)


def _shanten_and_waits(hand, melds, melded, disc_types):
    """返回 (shanten:int|None, waits_types:set[int])。Rust 调用集中于此。"""
    from riichienv import HandEvaluator, calculate_shanten

    tenpai_13s: list[list[int]] = []  # 听牌线的 13 张（弃后）门前手牌+melds
    shanten: int | None = None
    if not melded:
        if disc_types:
            best = None
            for t in disc_types:
                tile = _one_of_type(hand, t)
                if tile is None:
                    continue  # 手牌重建失配：该张跳过（防御，不整体放弃）
                rest = hand[:]
                rest.remove(tile)
                s = calculate_shanten(rest)
                if best is None or s < best:
                    best = s
            shanten = best
            if best == 0:  # 存在弃后听牌的打法 → 收各线 waits
                for t in disc_types:
                    tile = _one_of_type(hand, t)
                    if tile is None:
                        continue
                    rest = hand[:]
                    rest.remove(tile)
                    if calculate_shanten(rest) == 0:
                        tenpai_13s.append(rest)
        elif len(hand) % 3 == 1:
            # 无弃牌窗（门前清荣和窗等）：13(+杠) 张直接算
            shanten = calculate_shanten(hand)
            if shanten == 0:
                tenpai_13s.append(hand)
    else:
        # 副露：向听如实降级 None；听牌判定走 HandEvaluator(tiles, melds)（支持副露）
        for t in disc_types:
            tile = _one_of_type(hand, t)
            if tile is None:
                continue
            rest = hand[:]
            rest.remove(tile)
            if HandEvaluator(rest, melds).is_tenpai():
                tenpai_13s.append(rest)
        if not disc_types and len(hand) % 3 == 1:
            try:
                if HandEvaluator(hand, melds).is_tenpai():
                    tenpai_13s.append(hand)
            except BaseException:  # noqa: BLE001 —— 牌数形状不对就省，不连坐宝牌
                pass

    waits_t: set[int] = set()
    for rest in tenpai_13s:
        # 实测：get_waits() 返回的是牌种 type id（0..33），不是物理 id——勿再 //4
        for w in HandEvaluator(rest, melds).get_waits():
            waits_t.add(int(w))
    return shanten, waits_t


def hint_from_obs(ob) -> dict | None:
    """决策窗 Observation -> {"shanten","waits","dora","dora_in_hand","melded"}。

    waits/dora 为中文可渲染的 mjai 牌名（advisor.zh.tile_zh 直吃）。
    任何意外 → None，调用方按缺省显示。必须接 BaseException：Rust 侧对
    非法牌面（如雀魂流里 "?" 遮蔽座被 RiichiEnv 静默解析成全 1m）抛的
    PanicException 不是 Exception 子类，漏接会掀翻请求线程。
    """
    try:
        return _hint(ob)
    except BaseException:  # noqa: BLE001 —— 加分项绝不让推荐链路崩（含 Rust panic）
        return None


def _hint(ob) -> dict:
    hand = sorted(int(t) for t in ob.hand)
    pid = int(ob.player_id)
    melds_all = list(ob.melds)
    melds = list(melds_all[pid]) if pid < len(melds_all) else []
    melded = bool(melds)
    disc_types = _discard_types(ob.legal_actions())

    shanten: int | None = None
    waits_t: set[int] = set()
    try:
        shanten, waits_t = _shanten_and_waits(hand, melds, melded, disc_types)
    except BaseException:  # noqa: BLE001 —— 向听/待牌拿不到只丢这两项，宝牌照出
        shanten, waits_t = None, set()

    doras_t = sorted({_dora_of(int(i) // 4) for i in ob.dora_indicators})
    dora_in_hand = sum(1 for h in hand if h // 4 in set(doras_t))

    return {
        "shanten": shanten,
        "waits": [_type_name(t) for t in sorted(waits_t)],
        "dora": [_type_name(t) for t in doras_t],
        "dora_in_hand": dora_in_hand,
        "melded": melded,
    }
