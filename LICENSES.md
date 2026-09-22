# 第三方组件声明（LICENSES）

本项目主体许可见 [LICENSE](LICENSE)（Apache-2.0）。以下为随仓库分发的第三方组件及其许可与归属要求；**不随仓库分发**的组件列于文末，仅作研究用途的致谢，不构成再分发。

## 随仓库分发

| 组件 | 用途 | 许可 | 声明要求 |
|---|---|---|---|
| [Akagi](https://github.com/shinkuan/Akagi)（`capture/liqi`、`capture/mjai` 的移植改写；`reference/akagi-v3/` 为上游原文快照） | 抓包帧解码、liqi 协议、mjai 状态机参考 | Apache-2.0 | 保留本声明与上游版权信息；移植文件保留来源标注 |
| [LAYA](https://github.com/NandhaKishorM/laya)（推理/训练调用面） | 决策模型底座 | Apache-2.0 | 保留声明 |
| [ModernBERT](https://github.com/AnswerDotAI/ModernBERT)（本项目所用 LAYA 检查点的底座）/ [mmBERT](https://huggingface.co/jhu-clsp/mmBERT-base)（LAYA 多语变体底座） | 编码器 | Apache-2.0 / MIT | 各自保留声明；均已核验（GitHub LICENSE / HF 模型卡） |
| [riichienv](https://pypi.org/project/riichienv/) | 规则引擎（合法动作/模拟/牌谱重演） | Apache-2.0 | 保留声明 |
| [transformers](https://github.com/huggingface/transformers)、PyTorch 等运行依赖 | 训练/推理框架 | 各自许可（Apache-2.0 / BSD 等） | 见各包 LICENSE |

## 不随仓库分发（研究用途致谢）

| 组件 | 许可/约束 | 说明 |
|---|---|---|
| [Mortal](https://github.com/herentus/mortal) / libriichi | AGPL-3.0 | 仅作蒸馏教师与评估参照；**其代码、原权重均不随本仓库分发，亦未链接进任何产物**（本项目以其输出蒸馏出的权重另行随 Release 分发，见下节） |
| [tenhou-houou-mjai](https://huggingface.co/datasets/hhim8826/tenhou-houou-mjai) 语料 | 上游条款：个人研究/训练用途 | 语料与决策样本（pairs）**不随仓库分发**；训练所得权重随 Release 分发（见下节） |
| TypeSafe Jev | 闭源 | 仅作为灵感来源提及；**不进任何数据管线，本仓库与其无代码/数据往来** |

## 随 Release 分发（仓库外产物）

| 产物 | 许可 | 说明 |
|---|---|---|
| 微调权重 zip（GitHub Release 附件，内含 `MODEL_CARD.md`） | Apache-2.0 | 以 laya（Apache-2.0）为底座，在凤王牌譜研究语料上以 Mortal 输出为教师蒸馏训练；**不含语料原文、决策样本、Mortal 代码/原权重**。语料上游为个人研究/训练用途条款，**再分发或商用前请自行复核上游条款**；不附带真实对局使用许可。 |

更新纪律：新增任何依赖前先在此表登记；对外分发（Release/打包）前全表复核一次。
