# mjbrain

基于 [LAYA](https://github.com/NandhaKishorM/laya) 微调的立直麻将实时决策推荐系统（4 人麻将，雀魂优先）。**只推荐，不自动。**

## 写在前面

最近 Jev 非常火：一个只做决策的模型，相较穿透模型响应更快、资源占用更小。我在想它能在什么领域落地——个人很喜欢立直麻将，平时玩雀魂，看完别人的贪吃蛇 demo，就想着在麻将领域也试一次。

Jev 本身不开源、无法定制，于是这次尝试基于开源的 LAYA 做了微调，做出了这个 demo。模型性能目前比较一般，离传统立直麻将模型 [Mortal](https://github.com/herentus/mortal) 还很远；但作为"决策模型进入具体领域"的一次完整实践，从数据管线、蒸馏训练、规则引擎到实时推荐服务，全链路都开源了。**本项目仅供学习参考，不建议真实拿来打游戏。**

## 特性

- **只推荐，不自动**：零动作注入，不发出牌请求、不修改游戏状态，出牌永远由你自己按。实时数据走 web CDP 只读订阅 WebSocket 帧（无证书、无系统代理）。
- **档 0 · 手输局面**：CLI 输入当前局面（手牌/副露/牌河/宝牌），立即给出推荐动作 + 候选概率分布 + 向听/听牌标注，牌桌间隙查一手。
- **档 1 · 本地推荐服务**：`/v1/react` HTTP 服务接收 mjai 事件流，返回微调模型的推荐与候选分布；配套捕获管道（CDP → liqi 协议解码 → mjai 状态机 → 服务）可接入实时对局，也可离线重演棋谱。
- **模型管模糊判断，代码管精确规则**：合法动作集、向听、点数全部由确定性代码生成，模型只在合法子集上做带概率的选择，不会打出手牌里没有的牌。
- **完整训练管线**：牌谱重演 → 决策点提取 → 教师软标签蒸馏（RLCD 风格，human × teacher 混合目标）→ 单卡微调（8-bit 优化器 + 梯度检查点，8GB 显存可训）；配套竞技场评测框架（同一发牌种子、席位轮换、按种子聚类 95% CI）。

## 技术架构：与 Mortal 等传统模型的区别

传统模型大体两条路：[Mortal](https://github.com/herentus/mortal)、微软的 Suphx 是**端到端策略网络**——人工设计的多通道张量编码局面，深 ResNet 直接输出动作分布，靠自对局强化学习从零训起（Suphx 训练期还提供特权信息的 oracle 辅导，推理时移除）；akochan 是**搜索型**——在规则模拟器上跑 MCTS，评估函数管取舍。mjbrain 走第三条：**通用序列模型 + 外置规则引擎 + 离线蒸馏**。逐项对照：

| 维度 | 传统路线 | mjbrain |
|---|---|---|
| 模型族 | 专用网络：定型局面张量 → 深 ResNet（akochan 则以搜索替代网络） | 通用序列模型（[LAYA](https://github.com/NandhaKishorM/laya)）微调：局面渲染成 ≤512 token 文本，决策 = 在合法候选列表里做选择题 |
| 局面表示 | 手工设计的特征张量，向听等规则信息压缩在通道里，人读不出 | `state_text` 文本快照：向听、听牌、点数、宝牌显式写成可读特征 |
| 合法性保证 | 都由规则引擎负责（libriichi 状态机 / akochan 模拟器）——这点两边一样 | 同左；差别在耦合：riichienv 是独立环境层，同一份计算既喂模型文本，也喂复盘点数与 B1 兜底 |
| 训练范式 | 自对局强化学习：百万局量级、多卡、天级以上，奖励 = 终局得分 | 离线蒸馏：Mortal 当教师为人类牌谱决策打软标签，human × teacher 混合目标（RLCD 风格、分歧层退火、鸣牌加权），单卡 8-bit + 梯度检查点、几小时一轮 |
| 概率校准 | 输出原始策略概率 | 分桶温度逐跑重拟，top-N 连概率一起返回 |
| 推理栈 | Rust（libriichi）+ PyTorch C++ 扩展，毫秒级 | Python/PyTorch，CPU 约 1 秒/决策窗（慢是实话）；超时另有启发式 B1 兜底 |
| 训练复现 | 多卡集群起步，家用机重跑同等强度不现实 | 8GB 显存的游戏机跑通全管线 |

这笔架构交易换来三样：每步可解释（推荐自带向听/听牌/概率标注）、合法性与点数永远精确（规则只有一份代码）、训练管线家用可复现；付出的也有三样：推理吞吐不如 Rust 栈、没有搜索的事后修正、上限被教师水平和蒸馏数据量锁死——开头说的"离 Mortal 还很远"，主要就远在这三处。

## 快速开始

### 环境

依赖清单唯一入口是 [environment.yml](environment.yml)（conda 管理）：

```bash
conda env create -f environment.yml   # 首次：创建 mjbrain 环境（python 3.12 + 基础依赖）
conda activate mjbrain
pip install torch laya                # 推理侧必需（体积大，单独装更快）
pip install pandas pyarrow tqdm       # 训练数据管线（跑训练才需要）

python scripts/smoke_laya.py          # 设备冒烟：CUDA/MPS/CPU 检测
pytest -q                             # 测试（依赖棋谱样本的用例缺数据时自动跳过）
```

网络提示：HuggingFace 权重直连不稳可走镜像 `HF_ENDPOINT=https://hf-mirror.com`；pip 建议清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

### 权重

微调权重随 [Releases](../../releases) 单独分发（发布后补充直链）。下载 zip 解压到 `checkpoints/<run_id>/`（内含 `model.safetensors` + `encoder/` + `tokenizer/` + `rl_agent_config.json`），或启动时用 `--ckpt` 指向任意目录。包内 `MODEL_CARD.md` 记载出处、训练数据、held-out 指标与使用限制，`SHA256SUMS` 供逐文件校验。

> 权重许可与上游约束见包内 `MODEL_CARD.md` 与 [LICENSES.md](LICENSES.md)；语料、决策样本、Mortal 代码均**不随仓库分发**。

### 试一手（档 0 · 手输局面）

```bash
python scripts/live_advisor.py --seat 0 \
  --hand "2m3m4m5p6p7p2s3s4sEE1p9m" --draw "5s" \
  --rivers "9m|1s|E|P" --oya 0 --bakaze E --kyoku 1 --dora "1m" --topk 3
```

实测输出：

```
座位0 决策窗 14 项 → 推荐 dahai:9m
     94.0%  dahai:9m  未听
      5.8%  dahai:1p  未听
      0.1%  dahai:2s  未听
```

牌串格式：`1m`–`9m` / `1p`–`9p` / `1s`–`9s` / 字牌 `E S W N P F C`；赤五 `5mr`（或 `0m`）。`--last "pai<-frm"` 表达"上家刚打的牌"（吃/碰决策窗），`--furo` 可重复添加副露，完整参数 `--help`。

### 本地服务（档 1 · /v1/react）

```bash
python -m advisor.server --ckpt checkpoints/<run_id> --port 8765   # 启动即预热权重，仅绑 127.0.0.1
```

```bash
curl -s -X POST http://127.0.0.1:8765/v1/react \
  -H 'Content-Type: application/json' \
  -d '{"events": [/* mjai 事件数组，见 tests/test_advisor_server.py */], "seat": 0, "top": 5}'
# → {"applied": 89, "open_seats": [0],
#     "decisions": [{"seat": 0, "legal_n": 15, "recommend": "dahai:W", "top": [...]}]}
```

服务对输入只做重放与推荐：事件流原样重演出决策窗，在合法动作集上给出带概率的 top-N；服务端没有替玩家发动作的代码路径。健康检查 `GET /v1/health`。

### 实时捕获（可选，全程只读）

```bash
# 0) 首次使用捕获层：编译 liqi 协议描述符（生成 capture/liqi/_gen/，不入库）
python scripts/compile_liqi_proto.py
# 1) CDP 只读订阅雀魂页面的 WebSocket 帧 → JSONL（自动拉起独立 profile 浏览器窗口）
python scripts/run_capture.py --url https://game.maj-soul.com/1/ --out frames.jsonl
# 2) 尾随帧文件 → 解码为 mjai 事件流 → POST /v1/react → 打印实时推荐
#    每个决策窗的推荐同时落盘 frames.advise.jsonl（与帧文件对照即可复盘模型表现）
python scripts/live_from_capture.py --jsonl frames.jsonl --follow
```

> 懒人版：双击仓库根目录 `run_demo.bat` 一键开三个窗口（服务 / 捕获 / 推荐），
> 帧文件自动按局定名（`run1.jsonl`、`run2.jsonl`…），权重自动找 `checkpoints/` 下的解压位置。
```

## 架构

```
游戏流量 → capture（CDP → liqi 协议解码 → mjai 事件状态机，只读）
        → engine（规则重演 + 合法动作集/向听/点数，确定性代码）
        → brain（文本快照序列化，≤512 token 窗口）
        → advisor（LAYA 微调权重前向 + 分桶温度 → /v1/react 推荐）
超时兜底 = 启发式 B1（和牌/立直优先 + 现物防守 + 最小向听）
```

| 目录 | 职责 |
|---|---|
| `advisor/` | 推荐服务与局面合成（`/v1/react`、手输模式的状态重建） |
| `brain/` | state_text 序列化 + laya 序列构建（模型输入的唯一口径） |
| `capture/` | CDP 帧捕获、liqi protobuf 解码、mjai 状态机（Akagi 移植改写） |
| `engine/`、`environment/` | 牌谱重演、规则封装（riichienv） |
| `eval/` | 竞技场：B0 随机 / B1 启发式 / B3 = LAYA 微调，席位轮换 + 95% CI |
| `train/` | RLCD 蒸馏微调（软标签混合目标、单卡 8-bit） |
| `data/` | 牌谱 → 决策点提取 |
| `tests/` | 全链路测试：真棋谱往返、帧解码金标、服务契约 |

## 训练

1. `data/build_decisions.py`：牌谱（mjai 事件）→ 逐决策点样本；
2. 教师软标签蒸馏：`train/rlcd_sft.py`——human × teacher 混合目标、分歧层退火、鸣牌加权，单卡 8GB 可训；
3. `eval/arena` 竞技场出数，同一发牌种子 + 席位轮换，结论附 95% CI。

训练过程全记录在 `runs/`（`RunLog` 自动归档 config / env / metrics / summary）。语料与教师参照模型不随仓库分发（见下方致谢）。

## 致谢

- [LAYA](https://github.com/NandhaKishorM/laya) 与 [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)——决策模型底座与检查点（Apache-2.0）
- [Akagi](https://github.com/shinkuan/Akagi)——抓包/协议参考；`capture/` 为其 liqi/mjai 层的移植改写（Apache-2.0，声明见 [LICENSES.md](LICENSES.md)）
- [riichienv](https://pypi.org/project/riichienv/)——规则引擎：合法动作、向听、牌谱重演（Apache-2.0）
- [Mortal](https://github.com/herentus/mortal)——评估参照与蒸馏教师（研究用途，未随本仓库分发）
- [tenhou-houou-mjai](https://huggingface.co/datasets/hhim8826/tenhou-houou-mjai)——训练语料（研究用途，不随仓库分发）

## 许可证

本项目采用 **Apache License 2.0**，见 [LICENSE](LICENSE)。它与 Akagi / LAYA / riichienv 的许可兼容，可自由使用、修改、分发（保留声明）。

第三方组件的归属与声明汇总在 [LICENSES.md](LICENSES.md)；请在分发衍生作品前复核其中的约束项。

## 免责声明

- **关于雀魂实时推荐**：这个功能只为方便验证、复现而做，**不是游戏外挂**；作者本人仅在友人场验证过，并没有拿它参加过真实对局。
- 本项目**仅供学习参考，不建议真实拿来打游戏**；模型输出不构成任何建议，误操作后果自负。
- 系统对游戏客户端**零写入、零注入**（只读捕获、只出建议）；但只读截获本身仍受游戏服务条款约束，使用风险自担。
- 本仓库不含任何账号凭证、协议请求构造或自动化操作代码。
