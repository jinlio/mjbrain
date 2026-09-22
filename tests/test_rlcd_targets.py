import math
from collections import Counter

from train.rlcd_sft import MELD_TYPES, _human_vec, _materialize, _teacher_vec, mix_target


def _e(t, pai=None, **kw):
    d = {"type": t}
    if pai:
        d["pai"] = pai
    d.update(kw)
    return d


def test_human_vec_spreads_duplicates():
    entries = [_e("dahai", "5m"), _e("dahai", "5m"), _e("none")]
    v = _human_vec(0, entries)
    assert math.isclose(v[0], 0.5) and math.isclose(v[1], 0.5) and v[2] == 0.0
    v = _human_vec(2, entries)
    assert v == [0.0, 0.0, 1.0]


def test_teacher_vec_none_gets_zero_mass():
    v = _teacher_vec([1.0, None, 1.0], temp=1.0)
    assert v[1] == 0.0
    assert math.isclose(v[0], v[2])
    assert math.isclose(sum(v), 1.0)
    # 全 None -> 空表（调用侧按"教师不可用"回退人类标签）
    assert _teacher_vec([None, None], 1.0) == []
    assert _teacher_vec([], 1.0) == []


def test_mix_falls_back_to_human_without_teacher():
    entries = [_e("dahai", "5m"), _e("dahai", "5m"), _e("none")]
    stats = Counter()
    out = mix_target(entries, [None, None, None], 2, alpha=0.5, temp=0.3, stats=stats)
    assert out == [0.0, 0.0, 1.0]
    assert stats.get("target_teacher_unavailable") == 1


def test_mix_blends():
    entries = [_e("dahai", "5m"), _e("none")]
    out = mix_target(entries, [0.0, 0.0], 1, alpha=0.5, temp=1.0, stats=Counter())
    assert math.isclose(out[0], 0.25) and math.isclose(out[1], 0.75)
    assert math.isclose(sum(out), 1.0)


def _base(gt, agree, label_idx=1):
    """构造 _build_base 的最小 base item（materialize 只消费这些键）。"""
    return {
        "ids": [1, 2, 3],
        "markers": [1, 2],
        "qtype": 0,
        "entries": [_e("pon"), _e("none")],
        "aligned_q": [0.0, 0.0],
        "label_idx": label_idx,
        "gt": gt,
        "gid": "g",
        "idx": 0,
        "agree": agree,
    }


def test_materialize_disagree_alpha():
    # 分歧层（agree=False）退纯教师：target==teacher softmax；agree=True 用全局 alpha
    base = [_base("pon", False), _base("dahai", True)]
    st = Counter()
    out = _materialize(base, alpha=0.5, temp=1.0, stats=st, disagree_alpha=1.0)
    assert st["layer_disagree"] == 1
    pure_t = _teacher_vec([0.0, 0.0], 1.0)
    assert out[0]["target"] == pure_t          # 分歧层=纯教师
    assert not math.isclose(out[1]["target"][0], pure_t[0])  # 同意层仍是混合


def test_materialize_w_meld_only_hits_meld_types():
    # 只有鸣牌类样本带 w>1，其余 w=1；未开加权（w_meld=1）时全部 w=1
    base = [_base("chi", True), _base("pon", True), _base("dahai", True), _base("none", True)]
    st = Counter()
    out = _materialize(base, alpha=0.5, temp=1.0, stats=st, w_meld=2.0)
    ws = {it["gt"]: it["w"] for it in out}
    assert ws["chi"] == 2.0 and ws["pon"] == 2.0
    assert ws["dahai"] == 1.0 and ws["none"] == 1.0
    assert st["w_meld_applied"] == 2
    # 关掉加权 -> 全 1
    out2 = _materialize(base, alpha=0.5, temp=1.0, stats=Counter(), w_meld=1.0)
    assert all(it["w"] == 1.0 for it in out2)


def test_meld_types_coverage():
    assert {"chi", "pon", "daiminkan", "kakan", "ankan"} == set(MELD_TYPES)
