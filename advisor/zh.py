"""mjai 动作串 → 中文建议文本（显示层专用；日志/模型侧仍保留 mjai 原串）。

词表实测自 riichienv 的 legal_actions()：动作类型 dahai/none/pon/chi/daiminkan/
hora（另备 tsumo/reach/ankan/kakan/kan/ryukyoku/kyuushu_kyuuhai）；牌 token
1m-9m/1p-9p/1s-9s、赤5 用 'r' 后缀记法（5mr/5pr/5sr）、字牌 E/S/W/N/P/F/C。

映射口味按验收实例：dahai→切n万/切n饼/切n索（字牌切东/切白…）、none→跳过、
pon→碰n、chi→吃n、杠族→杠n（暗杠/加杠细分）、hora→和、tsumo→自摸。
"""

from __future__ import annotations

_SUIT = {"m": "万", "p": "饼", "s": "索"}
_HONOR = {"E": "东", "S": "南", "W": "西", "N": "北", "P": "白", "F": "发", "C": "中"}


def tile_zh(t: str) -> str:
    """'5s'→'5索'、'E'→'东'、'5mr'/'0m'→'赤5万'；无法识别或 '?' 原样返回。"""
    if not t or t == "?":
        return t
    red = t.endswith("r")          # riichienv 记法：5mr/5pr/5sr
    if red:
        t = t[:-1]
    if t[0] == "0":                # mjai 另一套赤5记法：0m/0p/0s
        red, t = True, "5" + t[1:]
    if t[0].isdigit():
        return f"{'赤' if red else ''}{t[0]}{_SUIT.get(t[1:], t[1:])}"
    return _HONOR.get(t, t)


def action_zh(a: str) -> str:
    """mjai 动作串 → 中文：'dahai:5s'→'切5索'、'none'→'跳过'、'hora'→'和'。"""
    kind, _, pai = a.partition(":")
    if kind == "none":
        return "跳过"
    if kind == "dahai":
        return "摸切" if pai == "?" else f"切{tile_zh(pai)}"
    if kind == "chi":
        return f"吃{tile_zh(pai)}"
    if kind == "pon":
        return f"碰{tile_zh(pai)}"
    if kind in ("kan", "daiminkan"):
        return f"杠{tile_zh(pai)}"
    if kind == "ankan":
        return f"暗杠{tile_zh(pai)}"
    if kind == "kakan":
        return f"加杠{tile_zh(pai)}"
    if kind == "hora":
        return "和"
    if kind == "tsumo":
        return "自摸"
    if kind == "reach":
        return "立直"
    if kind == "ryukyoku":
        return "流局"
    if kind == "kyuushu_kyuuhai":
        return "九种九牌"
    return a