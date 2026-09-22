"""liqi(雀魂) 帧 -> 结构化消息。移植改写自 Akagi v3 src/bridge/majsoul/parser.rs
（Apache-2.0，参考原文在 reference/akagi-v3/，见该 README 与其 LICENSES 标注）。

帧布局（与 Akagi 头注释同构，5 层）：
  [1B type][Request/Response 再 2B LE msg_id][Wrapper{name,data}]
  Wrapper.data -> DynamicMessage JSON；ActionPrototype.data -> base64(XOR(protobuf))。
type: 1=Notify 2=Request 3=Response。

**只读**：本包只 decode，从不 encode 请求、不持有任何发送能力。
"""
