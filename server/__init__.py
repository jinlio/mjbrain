"""server — 服务层：/v1/react（契约形状沿用 Akagi 云推理 [PLAN D1]）+ WS 推送。

{model, player_id, events[]} -> {reaction, candidates:[{action, prob}]}

安全纪律（调研 E-041 教训）：客户端不复检走法合法性 —— 本服务必须在
返回前用 engine 层验证 reaction ∈ 合法集，否则降级启发式 B1。
"""
