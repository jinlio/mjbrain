import json


def _capture_obs():
    from engine.sim import MahjongSim
    from eval import make_bot

    sim = MahjongSim(seed=11, hanchan=True)
    sim.reset()
    bots = [make_bot(f"B0#{k}", k) for k in range(4)]
    for _ in range(8):  # 打到中盘有河有副露附近
        acts = {}
        for pid in list(sim._obs):
            legal = sim.legal(pid)
            if not legal:
                continue
            acts[pid] = bots[pid].react(sim.events(), pid, legal)
        if not acts:
            break
        sim.step(acts)
    assert sim._obs, "没抓到观察"
    return sim, next(iter(sim._obs.values()))


def test_state_text_shape_and_leak():
    from brain.serialize import state_text

    _, ob = _capture_obs()
    text = state_text(ob)
    assert "我=" in text and "河0[" in text and "点数" in text
    # 零泄漏：对手手牌不在观察对象里（引擎已遮蔽），文本自然不含
    for pid in range(4):
        if pid != ob.player_id:
            assert list(ob.hands[pid] or []) == []


def test_determinism():
    from brain.serialize import state_text

    _, ob = _capture_obs()
    assert state_text(ob) == state_text(ob)


def test_options_1to1_and_mapping():
    from brain.serialize import laya_question, options_from_legal

    legal = [
        json.dumps({"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False}),
        json.dumps({"type": "dahai", "actor": 0, "pai": "5pr", "tsumogiri": False}),
        json.dumps({"type": "pon", "actor": 0, "pai": "5m", "consumed": ["5m", "5m"]}),
        json.dumps({"type": "chi", "actor": 0, "pai": "4s", "consumed": ["5s", "6s"]}),
        json.dumps({"type": "reach", "actor": 0}),
        json.dumps({"type": "hora", "actor": 0}),
        json.dumps({"type": "none", "actor": 0}),
    ]
    keys = options_from_legal(legal)
    assert len(keys) == len(legal)
    assert "5m" in keys and "5pr" in keys and "reach" in keys
    assert any(k.startswith("pon:5m") for k in keys)
    assert "chi:5s6s<-4s" in keys
    assert "pass" in keys and "hora" in keys
    q = laya_question(legal)
    assert q["t"] == "choice"
    # 消歧：同名 key 加后缀
    assert len(q["crit"]) == len(legal)
