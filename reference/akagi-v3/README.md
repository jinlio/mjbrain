# Akagi v3 参考源码摘录（移植用）

来源：https://github.com/shinkuan/Akagi （v3 分支，tag v3.7.1 附近，2026-09-22 经
raw.githubusercontent 拉取）。许可证：**Apache-2.0**（见 LICENSES.md 条目）。

用途：M4 档1 liqi→mjai 解码移植的对照实现（见 `docs/M4-CDP.md`）。
- `bridge/liqi.proto`、`bridge/liqi.json`：雀魂协议描述符与 rpc 路由表
- `bridge/parser.rs`：Wrapper 帧解析 + `wtf_decode`（ActionPrototype 位置相关 XOR）
- `bridge/mod.rs`：liqi 事件 -> mjai 映射（dispatch/handle_action_prototype）
- `bridge/tile.rs`：雀魂↔mjai 牌串映射 + mjai 规范序（已移植为 `capture/mjai/tile.py`；
  仓库原路径 `src/bridge/majsoul/tile.rs`，2026-09-22 补拉）
- `flow.rs`：per-WS 连接独立 bridge 实例与 msg_id pending-map 配对

移植产物（`capture/liqi/`）保留派作声明；这些 .rs 原文本身不参与 mjbrain 构建，
ruff 已排除 `reference/`。AGPL（Mortal）与本目录无关。
