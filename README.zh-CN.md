# WebLFP

English documentation: [README.md](README.md)

WebLFP 是一个本地优先的 LFP 推理 Web 应用。它仅使用 LFP 输入生成 256 维
LFP feature，并用同一份统一权重提供原项目一致的窄波/非窄波 `presence` 与
`count` 窗口级解码。
运行时不需要 Spike 输入，记录文件不会上传到外部服务。

## 功能范围

- 读取 NumPy、MATLAB、raw binary、SpikeGLX、Open Ephys、Intan、Plexon、
  AlphaOmega 和 NWB 等记录。
- 检查元数据，选择数据流、通道和时间范围，预览原始及 robust z-score 波形。
- 按锁定配置重采样、切窗并生成统一权重的 LFP feature。
- 可视化 PCA、相邻窗口余弦相似度、窄波/非窄波 presence 与 count。
- 保存本机运行历史，导出 `embeddings.npz` 和 `run.json`。
- 检测 CPU、CUDA、cuDNN、ROCm 及硬件性能，并允许安装兼容的 PyTorch 构建。

本仓库是**纯推理发布包**，不包含训练、微调、MAE masking/decoder、优化器、
数据集标注或 Spike sorting 功能。

## 获取代码和权重

模型权重使用 Git LFS 存储。克隆前需要安装 [Git LFS](https://git-lfs.com/)。

```bash
git lfs install
git clone <repository-url> WebLFP
cd WebLFP
git lfs pull
```

正常的 `git clone`/`git pull` 会在 Git LFS 正确安装后取得权重；启动脚本也会再次
执行 `git lfs pull`。锁定权重为：

| 用途 | 文件 | SHA-256 |
| --- | --- | --- |
| LFP feature_extractor + 窄波/非窄波解码器 | `models/spike-type-decoder/model.pt` | `0a68f7ce165eb52aaffda759146ea8be438de6b010aee731e19a36bbe8305809` |

加载模型时会验证完整文件 SHA-256 和严格的 inference-only checkpoint 格式。
权重不匹配、被替换或缺失时，应用会拒绝推理。

## 运行要求

- Git 与 Git LFS
- Python 3.12 或 3.13
- [uv](https://docs.astral.sh/uv/)
- Node.js 与 npm

首次运行会创建本地 `.venv`、安装锁文件中的 Python 依赖并构建前端。虚拟环境、
安装包、`node_modules`、实验记录、运行结果和日志均不进入 Git。

## Windows 部署

在仓库根目录运行：

```bat
run.bat
```

如果 `uv` 不在 `PATH`，可以先设置其位置：

```bat
set "UV_EXE=X:\path\to\uv.exe"
run.bat
```

## Linux 部署

```bash
bash ./run.sh
```

如需指定 `uv`：

```bash
UV_EXE=/path/to/uv bash ./run.sh
```

服务启动后访问 <http://127.0.0.1:8000>。需要跳过已完成的前端构建时，可设置
`SKIP_FRONTEND_BUILD=1`。GPU 运行环境可在 Web 设置页选择 CPU、CUDA 13.0+
或 Linux ROCm 构建；应用安装后使用 `--no-sync` 重启，以保留所选 PyTorch 后端。

## 手动校验

```bash
uv lock --check
uv run ruff check .
uv run pytest

cd frontend
npm ci
npm run build
npm audit --audit-level=high
```

如果已经通过设置页安装了特定 CUDA/ROCm PyTorch，请使用虚拟环境中的 Python
运行测试，避免普通 `uv run` 重新同步默认构建。

## 数据与科学边界

WebLFP 默认只监听本机回环地址。输入记录、生成数组、历史运行和日志保存在本机，
并由 `.gitignore` 排除。

该 LFP feature 和参考解码器仅在报告所述 KA 慢性癫痫小鼠海马数据与任务上得到验证。
结果不能重建 Spike 波形，不能直接视为单神经元真实细胞类型，也不能替代独立的
组织学、电生理或分子验证。

## 出版物引用要求

任何在论文、预印本、学位论文、会议摘要、数据论文或其他出版物中使用 WebLFP
代码、界面、模型权重或输出结果的工作，都必须正确引用本仓库。当前尚无对应论文，
请至少提供：

1. 仓库名称：`WebLFP`。
2. 实际使用的仓库 URL。
3. 精确的 release/tag；若无 release，则引用 Git commit SHA。
4. 访问日期。

可暂按以下形式引用，发布正式论文后再补充论文引用：

```text
WebLFP contributors. WebLFP [Computer software].
<repository-url>, version <release-or-commit>, accessed <YYYY-MM-DD>.
```

## 许可证

代码和随仓库分发的模型权重采用
[GNU Affero General Public License v3.0 only](LICENSE)（AGPL-3.0-only）。
通过网络向用户提供修改版服务时，应按 AGPL-3.0 第 13 条向这些用户提供对应源代码。
出版物引用要求属于科研归属与可复现性要求，不替代许可证全文。
