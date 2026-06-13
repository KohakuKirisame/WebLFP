# WebLFP 🧠⚡

**WebLFP** 是一个本地优先的 LFP-only 神经表征推理与参考 spike 活动解码 Web
应用。它读取本地 LFP 记录，使用锁定的预训练统一权重提取 256 维 LFP feature，
并用同一份权重中的下游头估计每个窗口内窄波 / 非窄波 spike 的 `presence` 与
`count`。

运行时**不需要 Spike 输入**，**不会上传记录文件**，也**不包含训练或微调代码**。

📘 English documentation: [README.md](README.md)

## ✨ 功能

- 📂 读取 NumPy、MATLAB、raw binary、SpikeGLX、Open Ephys、Intan、Plexon、
  AlphaOmega 和 NWB 记录。
- 🔎 推理前检查元数据、数据流选项、通道 ID、时间范围、dtype 和采样率。
- 📈 运行模型前预览原始波形和 robust z-score 处理后波形。
- 🧩 使用统一预训练权重生成 256D LFP feature。
- 🎯 将生成的 LFP feature 直接接入内置窄波 / 非窄波 `presence` 与 `count`
  下游头。
- 🖼️ 可视化 PCA 轨迹、相邻窗口余弦相似度和下游任务时间序列结果。
- 🧾 保存本机 run 历史，并导出 `embeddings.npz` 和 `run.json`。
- 🖥️ 检查 CPU、CUDA、cuDNN、ROCm、内存、显存和 BF16 能力，并在设置页选择
  PyTorch 构建。

## 🔒 仅推理范围

本仓库是推理发布包，刻意不包含：

- 训练循环，
- 微调代码，
- optimizer / scheduler / scaler 状态，
- MAE masking 和 decoder 运行路径，
- 数据集标注工具，
- Spike sorting 流程，
- 用户账号、云服务和远程数据上传。

随仓库发布的模型文件被锁定为严格的 inference-only schema：

```text
models/spike-type-decoder/model.pt
  format_version
  model_type
  feature_extractor
  head
```

发布检查会拒绝包含训练专用条目的 checkpoint，例如 `optimizer`、`scheduler`、
`scaler`、`teacher`、`student`、`decoder`、`mask_token` 或其他非推理状态。

## 📦 获取代码和权重

模型权重使用 Git LFS 存储。克隆前请先安装 Git LFS：

```bash
git lfs install
git clone <repository-url> WebLFP
cd WebLFP
git lfs pull
```

启动脚本也会在 Git LFS 可用时运行 `git lfs pull`。

| 用途 | 文件 | SHA-256 |
| --- | --- | --- |
| 统一 LFP feature extractor + 窄波/非窄波解码器 | `models/spike-type-decoder/model.pt` | `0a68f7ce165eb52aaffda759146ea8be438de6b010aee731e19a36bbe8305809` |

如果 checkpoint 缺失、被替换或 SHA-256 不匹配，WebLFP 会拒绝运行推理。

## 🚀 运行要求

- Git 和 Git LFS
- Python 3.12 或 3.13
- [uv](https://docs.astral.sh/uv/)
- Node.js 和 npm

首次启动会创建本地 `.venv`，根据 lock 文件安装 Python 依赖，并构建前端。虚拟环境、
日志、本地 runs、记录文件、安装包和生成数组均被排除在 Git 之外。

## ⚙️ 根据 GPU 配置 PyTorch

首次处理数据前，请先打开**设置**，根据实际硬件配置 PyTorch。不要默认认为初始安装的
PyTorch 已经能够使用 GPU。

- NVIDIA GPU：确认当前驱动支持 CUDA 13.0 或更高版本，再选择 WebLFP 显示为兼容的
  CUDA PyTorch 构建。
- AMD GPU：使用受支持的 Linux 系统，并选择兼容的 ROCm 构建。
- 没有兼容 GPU，或不需要 GPU：选择 CPU 构建。

WebLFP 检测到所选后端后，点击**使用当前环境**。首次打开时，完成该确认前不能进入
工作区。

## 🪟 Windows

```bat
run.bat
```

如果 `uv` 不在 `PATH`：

```bat
set "UV_EXE=X:\path\to\uv.exe"
run.bat
```

## 🐧 Linux

```bash
bash ./run.sh
```

如果需要指定 `uv`：

```bash
UV_EXE=/path/to/uv bash ./run.sh
```

启动后打开 <http://127.0.0.1:8000>。

如果已经完成前端构建，可以设置 `SKIP_FRONTEND_BUILD=1` 跳过构建。GPU 用户可在设置页
选择 CPU、CUDA 或 Linux ROCm PyTorch 构建。

## 🧬 科学边界

WebLFP 运行时只使用 LFP。Spike 信息用于参考训练流程，但部署推理不需要 Spike 输入。

生成的 LFP feature 和参考解码器只在报告所述 KA 慢性癫痫小鼠海马记录和任务设置上
得到验证。输出不能重建 Spike 波形，不能识别真实单神经元细胞类型，也不应在缺少独立
验证时解释为生物医学结论。

如果你有兴趣，欢迎阅读我们先前的工作：Cao, F., Feng, Z., Zhang, J., & Shi, W.
(2026). HuiduRep: A Robust Self-Supervised Framework for Learning Neural
Representations from Extracellular Recordings. Proceedings of the AAAI
Conference on Artificial Intelligence, 40(21), 17374–17383.
https://doi.org/10.1609/aaai.v40i21.38790

## 📚 引用要求

任何在论文、预印本、学位论文、会议摘要、数据论文或其他公开成果中使用 WebLFP 代码、
界面、模型权重或生成结果的工作，都必须正确引用本仓库。当前尚无 WebLFP 专门论文；
请至少包含：

1. 仓库名称：`WebLFP`。
2. 实际使用的仓库 URL。
3. 精确的 release/tag；如果没有 release，则引用 Git commit SHA。
4. 访问日期。

可暂按以下格式引用：

```text
WebLFP contributors. WebLFP [Computer software].
<repository-url>, version <release-or-commit>, accessed <YYYY-MM-DD>.
```

## ⚖️ 许可证

代码和随仓库分发的模型权重采用
[GNU Affero General Public License v3.0 only](LICENSE)（`AGPL-3.0-only`）。

如果你通过网络向用户提供修改版 WebLFP 服务，AGPL-3.0 第 13 条要求你向这些用户提供
对应源代码。
