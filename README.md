# WebLFP 🧠⚡

**WebLFP** is a local-first web application for LFP-only neural representation
inference and reference spike-activity decoding. It reads local LFP recordings,
extracts 256-dimensional LFP features with a locked pretrained checkpoint, and
uses the same checkpoint's downstream head to estimate narrow / non-narrow spike
`presence` and `count` for each window.

Runtime inference does **not** require Spike input, does **not** upload recording
files, and does **not** include training or fine-tuning code.

📘 中文文档：[README.zh-CN.md](README.zh-CN.md)

## ✨ What It Does

- 📂 Reads NumPy, MATLAB, raw binary, SpikeGLX, Open Ephys, Intan, Plexon,
  AlphaOmega, and NWB recordings.
- 🔎 Inspects metadata, stream options, channel IDs, time ranges, dtype, and
  sampling rate before inference.
- 📈 Previews raw and robust z-score processed traces before running the model.
- 🧠 Keeps the latest selected raw segment in memory so preview and inference do
  not read the same recording range twice; one preview may cover up to 300 s.
- 🧩 Generates 256D LFP features from the unified pretrained checkpoint.
- 🎯 Runs the bundled narrow / non-narrow `presence` and `count` downstream head
  directly on those generated LFP features.
- 🖼️ Visualizes PCA trajectories, adjacent-window cosine similarity, and
  downstream time-series results.
- 🧾 Saves local run history and exports `embeddings.npz` plus `run.json`.
- 🖥️ Checks CPU, CUDA, cuDNN, ROCm, memory, VRAM, and BF16 capability, with a
  settings page for choosing a PyTorch build.

## 🔒 Inference-Only Scope

This repository is an inference release package. It intentionally excludes:

- training loops,
- fine-tuning code,
- optimizer / scheduler / scaler state,
- MAE masking and decoder runtime,
- dataset annotation tools,
- Spike sorting workflows,
- user accounts, cloud services, and remote data upload.

The bundled model file is locked to a strict inference schema:

```text
models/spike-type-decoder/model.pt
  format_version
  model_type
  feature_extractor
  head
```

The release checks reject checkpoint files that contain training-only entries
such as `optimizer`, `scheduler`, `scaler`, `teacher`, `student`, `decoder`,
`mask_token`, or other non-inference state.

## 📦 Get The Code And Weights

Model weights are stored with Git LFS. Install Git LFS before cloning:

```bash
git lfs install
git clone <repository-url> WebLFP
cd WebLFP
git lfs pull
```

The launch scripts also run `git lfs pull` when Git LFS is available.

| Purpose | File | SHA-256 |
| --- | --- | --- |
| Unified LFP feature extractor + narrow/non-narrow decoder | `models/spike-type-decoder/model.pt` | `0a68f7ce165eb52aaffda759146ea8be438de6b010aee731e19a36bbe8305809` |

If the checkpoint is missing, replaced, or hash-mismatched, WebLFP refuses to
run inference.

## 🚀 Requirements

- Git and Git LFS
- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)
- Node.js and npm

The first launch creates a local `.venv`, installs Python dependencies from the
lock file, and builds the frontend. Virtual environments, logs, local runs,
recordings, installers, and generated arrays are excluded from Git.

## ⚙️ Configure PyTorch For Your GPU

Before processing data for the first time, open **Settings** and configure
PyTorch for the actual hardware. Do not assume that the default build can use
the GPU.

- NVIDIA GPU: confirm that the installed driver supports CUDA 13.0 or newer,
  then choose a compatible CUDA PyTorch build shown by WebLFP.
- AMD GPU: use a supported Linux system and choose a compatible ROCm build.
- No compatible GPU, or GPU use is not required: choose the CPU build.

After WebLFP detects the selected backend, click **Use current environment**.
The workspace remains unavailable on first open until this confirmation is
completed.

## 🪟 Windows

```bat
run.bat
```

If `uv` is not on `PATH`:

```bat
set "UV_EXE=X:\path\to\uv.exe"
run.bat
```

## 🐧 Linux

```bash
bash ./run.sh
```

If you need to specify `uv`:

```bash
UV_EXE=/path/to/uv bash ./run.sh
```

After startup, open <http://127.0.0.1:8000>.

Set `SKIP_FRONTEND_BUILD=1` to skip a completed frontend build. GPU users can
choose CPU, CUDA, or Linux ROCm PyTorch builds from the settings page.

## 🧬 Scientific Boundary

WebLFP uses LFP only at runtime. Spike information was used in the reference
training workflow, but Spike input is not required for deployment.

The generated LFP features and reference decoder were validated on the reported
KA chronic epilepsy mouse hippocampal recordings and task setup. The output does
not reconstruct Spike waveforms, does not identify true single-neuron cell type,
and should not be interpreted as a biomedical conclusion without independent
validation.

If you are interested, you are welcome to read our prior work: Cao, F., Feng,
Z., Zhang, J., & Shi, W. (2026). HuiduRep: A Robust Self-Supervised Framework
for Learning Neural Representations from Extracellular Recordings. Proceedings
of the AAAI Conference on Artificial Intelligence, 40(21), 17374–17383.
https://doi.org/10.1609/aaai.v40i21.38790

## 📚 Citation Requirement

Any publication, preprint, thesis, conference abstract, data paper, or other
public work that uses WebLFP code, UI, model weights, or generated outputs must
properly cite this repository. There is no dedicated WebLFP paper yet; please
include at least:

1. Repository name: `WebLFP`.
2. The repository URL you used.
3. The exact release/tag, or the Git commit SHA if no release is available.
4. Access date.

Temporary citation format:

```text
WebLFP contributors. WebLFP [Computer software].
<repository-url>, version <release-or-commit>, accessed <YYYY-MM-DD>.
```

## ⚖️ License

Code and bundled model weights are distributed under
[GNU Affero General Public License v3.0 only](LICENSE) (`AGPL-3.0-only`).

If you provide a modified WebLFP service over a network, AGPL-3.0 section 13
requires you to provide the corresponding source code to those users.
