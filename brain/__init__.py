"""brain — LAYA 决策层：牌局快照 -> typed decision。

组件：
- serialize.py 状态→文本快照（受 token 预算约束，M2 核心研究点，调研 [C-022]）
- laya_agent.py LAYA 适配（Router/preload，CUDA/MPS/CPU）
- policy.py 组装合法子集为 choice options；防守辅助问（noul/score 并行一次前向）
"""
