import { useEffect, useMemo, useState } from "react";
import type { Language } from "./i18n";

export type ThemeMode = "auto" | "dark" | "light";

type SystemInfo = {
  platform: {
    system: string;
    release: string;
    machine: string;
    python_version: string;
    processor: string;
  };
  memory: {
    total_gib: number | null;
    installed_gib: number | null;
    configured_speed_mt_s: number | null;
    modules: Array<{
      manufacturer: string;
      part_number: string;
      capacity_bytes: number;
      rated_speed_mt_s: number | null;
      configured_speed_mt_s: number | null;
    }>;
  };
  nvidia: {
    available: boolean;
    gpus: Array<{ name: string; driver_version: string; memory_mib: number }>;
    driver_cuda_version: string | null;
    toolkit_cuda_version: string | null;
    minimum_cuda_version: string;
  };
  rocm: {
    platform_supported: boolean;
    detected: boolean;
    version: string | null;
  };
  pytorch: {
    installed: boolean;
    version: string | null;
    backend: "none" | "cpu" | "cuda" | "rocm";
    cuda_build_version: string | null;
    hip_build_version: string | null;
    cuda_available: boolean;
    device_count: number;
    devices: string[];
    device_memory_mib: number[];
    cudnn_available: boolean;
    cudnn_version: string | null;
  };
  cuda_bf16: {
    evaluated: boolean;
    supported: boolean | null;
    measured_tflops: number | null;
    reference: string;
    reference_floor_tflops: number;
    passes: boolean | null;
    reason: string;
  };
  performance: {
    acceptable: boolean;
    backend: "cpu" | "cuda" | "rocm";
    warnings: Array<{ code: string; message: string }>;
    thresholds: {
      system_memory_gib: number;
      memory_speed_mt_s: number;
      gpu_memory_gb: number;
      cuda_bf16_tflops: number;
    };
  };
  system_cudnn_version: string | null;
  system_cudnn_files: string[];
};

type PyTorchOption = {
  id: string;
  torch_version: string;
  backend: "cpu" | "cuda" | "rocm";
  runtime_version: string | null;
  index_url: string;
  label: string;
  recommended: boolean;
  compatible: boolean;
  compatibility_reason: string;
};

type InstallStatus = {
  state: "idle" | "waiting" | "installing" | "completed" | "failed" | "unknown";
  message: string;
  job_id?: string;
  verification?: Record<string, unknown>;
  elapsed_sec?: number;
  log_tail?: string[];
};

type UpdateStatus = {
  repository_url: string;
  branch: string;
  status: "up_to_date" | "update_available" | "local_ahead" | "diverged" | "unavailable";
  update_available: boolean | null;
  local_commit: string | null;
  remote_commit: string | null;
  latest_commit_url: string | null;
  detail: string;
};

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail ?? `Request failed: ${response.status}`);
  return payload as T;
}

function formatMemory(mebibytes: number): string {
  return `${(mebibytes / 1024).toFixed(1)} GiB`;
}

function formatElapsed(seconds = 0, language: Language): string {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  if (language === "zh") return minutes ? `${minutes} 分 ${remainder} 秒` : `${remainder} 秒`;
  return minutes ? `${minutes} min ${remainder} sec` : `${remainder} sec`;
}

export default function Settings({
  theme,
  language,
  environmentConfirmed,
  onThemeChange,
  onLanguageChange,
  onEnvironmentConfirm,
  onBack,
}: {
  theme: ThemeMode;
  language: Language;
  environmentConfirmed: boolean;
  onThemeChange: (theme: ThemeMode) => void;
  onLanguageChange: (language: Language) => void;
  onEnvironmentConfirm: () => void;
  onBack: () => void;
}) {
  const chinese = language === "zh";
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [options, setOptions] = useState<PyTorchOption[]>([]);
  const [selectedOption, setSelectedOption] = useState("");
  const [status, setStatus] = useState<InstallStatus | null>(null);
  const [update, setUpdate] = useState<UpdateStatus | null>(null);
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function refresh(force = false) {
    setLoading(true);
    setError("");
    try {
      const systemValue = await getJson<SystemInfo>(
        `/api/settings/system${force ? "?refresh=true" : ""}`,
      );
      const [optionValue, statusValue] = await Promise.all([
        getJson<{ minimum_cuda_version: string; options: PyTorchOption[] }>(
          "/api/settings/pytorch-options",
        ),
        getJson<InstallStatus>("/api/settings/pytorch-install"),
      ]);
      setSystem(systemValue);
      setOptions(optionValue.options);
      setStatus(statusValue);
      if (!selectedOption) {
        const compatible = optionValue.options.filter((option) => option.compatible);
        const preferred =
          compatible
            .filter((option) => option.recommended && option.backend === "cuda")
            .sort((a, b) => (b.runtime_version ?? "").localeCompare(a.runtime_version ?? ""))[0] ??
          compatible.find((option) => option.recommended) ??
          compatible[0];
        setSelectedOption(preferred?.id ?? "");
      }
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function checkUpdate() {
    setCheckingUpdate(true);
    setError("");
    try {
      setUpdate(await getJson<UpdateStatus>("/api/settings/update"));
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setCheckingUpdate(false);
    }
  }

  useEffect(() => {
    void refresh();
    void checkUpdate();
  }, []);

  useEffect(() => {
    if (!status || !["waiting", "installing"].includes(status.state)) return;
    const timer = window.setInterval(async () => {
      try {
        const value = await getJson<InstallStatus>("/api/settings/pytorch-install");
        setStatus(value);
        if (["completed", "failed"].includes(value.state)) {
          window.clearInterval(timer);
          await refresh();
        }
      } catch {
        setStatus((current) => ({
          state: current?.state === "installing" ? "installing" : "waiting",
          message: chinese ? "WebLFP 已停止，正在安装 PyTorch 并等待服务重启…" : "WebLFP has stopped while PyTorch is installed. Waiting for the service to restart…",
          elapsed_sec: (current?.elapsed_sec ?? 0) + 1.5,
          log_tail: current?.log_tail,
        }));
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [status?.state]);

  const selected = options.find((option) => option.id === selectedOption);
  const grouped = useMemo(
    () => ({
      cuda: options.filter((option) => option.backend === "cuda"),
      cpu: options.filter((option) => option.backend === "cpu"),
      rocm: options.filter((option) => option.backend === "rocm"),
    }),
    [options],
  );

  async function install() {
    if (!selected?.compatible) return;
    const confirmed = window.confirm(
      chinese
        ? `安装 ${selected.label}？\n\nWebLFP 将停止，替换当前 PyTorch 后自动重启。独立进度窗口会显示安装日志，关闭该窗口不会中断后台安装。`
        : `Install ${selected.label}?\n\nWebLFP will stop, replace PyTorch, and restart automatically. A separate progress window shows the log; closing it does not stop the background installation.`,
    );
    if (!confirmed) return;
    setError("");
    setStatus({ state: "waiting", message: chinese ? "正在安排安装任务…" : "Scheduling the installation…" });
    try {
      const value = await getJson<InstallStatus>("/api/settings/pytorch-install", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ option_id: selected.id, confirmed: true }),
      });
      setStatus(value);
    } catch (reason) {
      const message = (reason as Error).message;
      if (message.includes("fetch")) {
        setStatus({ state: "waiting", message: chinese ? "服务正在停止，安装即将开始…" : "The service is stopping. Installation will begin shortly…" });
      } else {
        setError(message);
        setStatus(null);
      }
    }
  }

  return (
    <div className="settings-shell">
      <aside className="settings-sidebar">
        <button className="back-button" onClick={onBack}>← {chinese ? "返回工作区" : "Back to workspace"}</button>
        <span className="settings-label">SETTINGS</span>
        <a href="#compute">{chinese ? "计算环境" : "Compute environment"}</a>
        <a href="#updates">{chinese ? "检查更新" : "Check for updates"}</a>
        <a href="#language">{chinese ? "语言" : "Language"}</a>
        <a href="#appearance">{chinese ? "外观" : "Appearance"}</a>
        <div className="settings-sidebar-note">
          {chinese ? "安装仅使用 PyTorch 官方 wheel 索引，不接受自定义命令或下载地址。" : "Installation uses official PyTorch wheel indexes only. Custom commands and download URLs are not accepted."}
        </div>
      </aside>

      <main className="settings-content">
        <header className="settings-heading">
          <div><span>APPLICATION SETTINGS</span><h1>{chinese ? "设置" : "Settings"}</h1></div>
          <button className="secondary-button compact" onClick={() => void refresh(true)} disabled={loading}>
            {loading ? (chinese ? "检测中…" : "Detecting…") : (chinese ? "重新检测" : "Detect again")}
          </button>
        </header>

        {error && <div className="error-banner">{error}</div>}

        <section id="compute" className="settings-section">
          <div className="settings-section-title">
            <div><span>COMPUTE ENVIRONMENT</span><h2>{chinese ? "计算环境" : "Compute environment"}</h2></div>
            <p>{chinese ? "CUDA wheel 最低为 13.0；ROCm 官方 wheel 仅支持 Linux。" : "CUDA wheels require 13.0 or newer. Official ROCm wheels are Linux-only."}</p>
          </div>

          {system && (!system.pytorch.installed ? (
            <div className="performance-alert pending">
              <div><span>RUNTIME REQUIRED</span><strong>{chinese ? "尚未安装 PyTorch" : "PyTorch is not installed"}</strong></div>
              <p>{chinese ? "请在下方选择适合当前硬件的 CPU、CUDA 或 ROCm 版本。安装并重新检测后才能确认运行环境。" : "Choose a CPU, CUDA, or ROCm build for this hardware below. Install it and detect again before confirming the environment."}</p>
            </div>
          ) : system.performance.warnings.length ? (
            <div className="performance-alert poor">
              <div><span>PERFORMANCE WARNING</span><strong>{chinese ? "机器性能过差" : "Hardware below recommended performance"}</strong></div>
              <ul>{system.performance.warnings.map((warning) => <li key={warning.code}>{warning.message}</li>)}</ul>
            </div>
          ) : (
            <div className="performance-alert good">
              <div><span>PERFORMANCE CHECK</span><strong>{chinese ? "当前性能门槛已通过" : "Current performance thresholds passed"}</strong></div>
              <p>{system.pytorch.backend === "cpu" ? (chinese ? "当前为 CPU 模式，已按规则忽略显存和 CUDA BF16 项。" : "CPU mode is active; VRAM and CUDA BF16 checks are ignored.") : (chinese ? "内存、显存和当前计算后端均未触发性能告警。" : "Memory, VRAM, and the active compute backend produced no performance warning.")}</p>
            </div>
          ))}

          <div className="hardware-grid">
            <div className="hardware-card surface">
              <span>SYSTEM MEMORY</span>
              <strong>{system?.memory.installed_gib ? `${system.memory.installed_gib.toFixed(1)} GiB` : (chinese ? "未检测" : "Not detected")}</strong>
              <p>{system?.memory.configured_speed_mt_s ? `${system.memory.configured_speed_mt_s} MT/s configured · ${system.memory.modules.length || "?"} module(s)` : (chinese ? "有效传输率未检测" : "Configured memory speed not detected")}</p>
            </div>
            <div className="hardware-card surface">
              <span>GPU / DRIVER</span>
              <strong>{system?.nvidia.gpus[0]?.name ?? (chinese ? "未检测到 NVIDIA GPU" : "No NVIDIA GPU detected")}</strong>
              <p>{system?.nvidia.gpus[0] ? `${formatMemory(system.nvidia.gpus[0].memory_mib)} · Driver ${system.nvidia.gpus[0].driver_version}` : (chinese ? "CPU 模式仍可用" : "CPU mode remains available")}</p>
            </div>
            <div className="hardware-card surface">
              <span>CUDA</span>
              <strong>{system?.nvidia.driver_cuda_version ? `Driver ${system.nvidia.driver_cuda_version}` : (chinese ? "不可用" : "Unavailable")}</strong>
              <p>{system?.nvidia.toolkit_cuda_version ? `Toolkit ${system.nvidia.toolkit_cuda_version}` : (chinese ? "未检测到 nvcc" : "nvcc not detected")}</p>
            </div>
            <div className="hardware-card surface">
              <span>PYTORCH</span>
              <strong>{system ? (system.pytorch.version ?? (chinese ? "未安装" : "Not installed")) : (chinese ? "检测中" : "Detecting")}</strong>
              <p>{system?.pytorch.installed ? `${system.pytorch.backend.toUpperCase()} build · CUDA available ${system.pytorch.cuda_available ? "yes" : "no"}` : (chinese ? "请在下方选择并安装" : "Choose and install a runtime below")}</p>
            </div>
            <div className="hardware-card surface">
              <span>cuDNN / ROCm</span>
              <strong>{system?.pytorch.cudnn_version ? `cuDNN ${system.pytorch.cudnn_version}` : system?.system_cudnn_version ? `System cuDNN ${system.system_cudnn_version}` : (chinese ? "cuDNN 不可用" : "cuDNN unavailable")}</strong>
              <p>{system?.rocm.platform_supported ? (system.rocm.detected ? `ROCm ${system.rocm.version ?? "detected"}` : (chinese ? "ROCm 未检测到" : "ROCm not detected")) : (chinese ? "ROCm 仅支持 Linux" : "ROCm is Linux-only")}</p>
            </div>
            <div className="hardware-card surface">
              <span>CUDA BF16</span>
              <strong>{system && !system.pytorch.installed ? (chinese ? "等待安装 PyTorch" : "Waiting for PyTorch") : system?.pytorch.backend === "cpu" ? (chinese ? "CPU 模式已忽略" : "Ignored in CPU mode") : system?.cuda_bf16.measured_tflops ? `${system.cuda_bf16.measured_tflops.toFixed(1)} TFLOP/s` : system?.cuda_bf16.supported === false ? (chinese ? "不支持原生 BF16" : "Native BF16 unsupported") : (chinese ? "未完成检测" : "Not measured")}</strong>
              <p>{chinese ? "RTX 4070 级实测下限" : "RTX 4070-class measured floor"} {system?.cuda_bf16.reference_floor_tflops ?? 60} TFLOP/s</p>
            </div>
          </div>

          <div className={`environment-confirm ${environmentConfirmed ? "confirmed" : ""}`}>
            <div>
              <span>ENVIRONMENT CONFIRMATION</span>
              <strong>{environmentConfirmed ? (chinese ? "当前环境已确认" : "Current environment confirmed") : (chinese ? "请确认当前运行环境" : "Confirm the current runtime environment")}</strong>
              <p>{chinese ? "确认你已按 GPU 配置选择合适的 PyTorch；CPU 模式也是有效选择。" : "Confirm that PyTorch matches the GPU configuration. CPU mode is also a valid choice."}</p>
            </div>
            <button className="primary-button" disabled={!system?.pytorch.installed || loading || environmentConfirmed} onClick={onEnvironmentConfirm}>
              {environmentConfirmed ? (chinese ? "已配置" : "Configured") : (chinese ? "使用当前环境" : "Use current environment")}
            </button>
          </div>

          <div className="install-panel surface">
            <div className="install-heading">
              <div><span>PYTORCH RUNTIME</span><h3>{chinese ? "选择 PyTorch 版本与计算后端" : "Choose a PyTorch version and compute backend"}</h3></div>
              <span className="minimum-badge">CUDA ≥ 13.0</span>
            </div>

            {(["cuda", "cpu", "rocm"] as const).map((backend) => (
              <div className="option-group" key={backend}>
                <span>{backend.toUpperCase()}</span>
                <div className="runtime-options">
                  {grouped[backend].map((option) => (
                    <button
                      key={option.id}
                      className={`runtime-option ${selectedOption === option.id ? "selected" : ""}`}
                      disabled={!option.compatible}
                      onClick={() => setSelectedOption(option.id)}
                      title={option.compatibility_reason}
                    >
                      <strong>{option.label}</strong>
                      <small>{option.compatible ? option.compatibility_reason : `${chinese ? "不可用" : "Unavailable"}: ${option.compatibility_reason}`}</small>
                    </button>
                  ))}
                </div>
              </div>
            ))}

            <div className="install-action">
              <div>
                <span>{chinese ? "将安装" : "TO INSTALL"}</span>
                <strong>{selected?.label ?? (chinese ? "请选择兼容版本" : "Select a compatible version")}</strong>
                <code>{selected?.index_url ?? "-"}</code>
              </div>
              <button className="primary-button" disabled={!selected?.compatible || ["waiting", "installing"].includes(status?.state ?? "")} onClick={() => void install()}>
                {["waiting", "installing"].includes(status?.state ?? "") ? (chinese ? "安装进行中…" : "Installation running…") : (chinese ? "确认并安装" : "Confirm and install")}
              </button>
            </div>

            {status && status.state !== "idle" && (
              <div className={`install-status ${status.state}`}>
                <div className="install-status-summary">
                  <span>{status.state.toUpperCase()}</span>
                  <p>{status.message}</p>
                  <small>{chinese ? "已用时" : "Elapsed"} {formatElapsed(status.elapsed_sec, language)}</small>
                </div>
                <div className={`install-progress-track ${["waiting", "installing"].includes(status.state) ? "active" : ""}`}>
                  <i />
                </div>
                {status.log_tail && status.log_tail.length > 0 && (
                  <pre className="install-log">{status.log_tail.join("\n")}</pre>
                )}
              </div>
            )}
          </div>
        </section>

        <section id="updates" className="settings-section">
          <div className="settings-section-title">
            <div><span>SOFTWARE UPDATE</span><h2>{chinese ? "检查更新" : "Check for updates"}</h2></div>
            <p>{chinese ? "比较本地 Git commit 与 GitHub main；不会自动修改或覆盖本地文件。" : "Compare the local Git commit with GitHub main. No local files are changed automatically."}</p>
          </div>
          <div className={`update-panel surface ${update?.status ?? "idle"}`}>
            <div className="update-heading">
              <div>
                <span>{update?.status === "update_available" || update?.status === "diverged" ? "UPDATE AVAILABLE" : "REPOSITORY STATUS"}</span>
                <strong>{!update ? (chinese ? "尚未检查" : "Not checked yet") : update.status === "up_to_date" ? (chinese ? "当前已是最新版本" : "WebLFP is up to date") : update.status === "local_ahead" ? (chinese ? "本地版本领先于 GitHub" : "Local checkout is ahead of GitHub") : update.status === "diverged" ? (chinese ? "本地与 GitHub 已分叉" : "Local and GitHub histories have diverged") : update.status === "update_available" ? (chinese ? "发现新的 GitHub commit" : "A newer GitHub commit is available") : (chinese ? "暂时无法检查更新" : "Update check unavailable")}</strong>
              </div>
              <button className="secondary-button compact" onClick={() => void checkUpdate()} disabled={checkingUpdate}>
                {checkingUpdate ? (chinese ? "检查中…" : "Checking…") : (chinese ? "重新检查" : "Check again")}
              </button>
            </div>
            <a className="repository-link" href="https://github.com/KohakuKirisame/WebLFP" target="_blank" rel="noreferrer">https://github.com/KohakuKirisame/WebLFP</a>
            {update && (
              <>
                <div className="commit-comparison">
                  <div><span>{chinese ? "本地 commit" : "Local commit"}</span><code>{update.local_commit?.slice(0, 12) ?? "-"}</code></div>
                  <div><span>GitHub {update.branch}</span><code>{update.remote_commit?.slice(0, 12) ?? "-"}</code></div>
                </div>
                {(update.status === "update_available" || update.status === "diverged") && (
                  <div className="update-guidance">
                    <p>{update.status === "diverged" ? (chinese ? "检测到本地修改与远端提交分叉。请先备份本地修改，再手动处理 Git 更新。" : "Local and remote commits have diverged. Back up local changes before resolving the Git update manually.") : (chinese ? "关闭 WebLFP 后，在项目目录运行 git pull --ff-only 获取更新。" : "Close WebLFP, then run git pull --ff-only in the project directory to update.")}</p>
                    {update.latest_commit_url && <a href={update.latest_commit_url} target="_blank" rel="noreferrer">{chinese ? "查看最新 commit" : "View latest commit"}</a>}
                  </div>
                )}
              </>
            )}
          </div>
        </section>

        <section id="language" className="settings-section">
          <div className="settings-section-title">
            <div><span>LANGUAGE</span><h2>{chinese ? "界面语言" : "Interface language"}</h2></div>
            <p>{chinese ? "语言偏好保存在当前浏览器中；首次打开默认使用英文。" : "The language preference is stored in this browser. English is the default on first open."}</p>
          </div>
          <div className="language-options">
            <button className={`language-option ${language === "en" ? "selected" : ""}`} onClick={() => onLanguageChange("en")}><strong>English</strong><small>English interface and guide</small></button>
            <button className={`language-option ${language === "zh" ? "selected" : ""}`} onClick={() => onLanguageChange("zh")}><strong>中文</strong><small>中文界面与完整说明</small></button>
          </div>
        </section>

        <section id="appearance" className="settings-section">
          <div className="settings-section-title">
            <div><span>APPEARANCE</span><h2>{chinese ? "界面外观" : "Appearance"}</h2></div>
            <p>{chinese ? "“跟随系统”会实时响应操作系统的深浅色设置。" : "Follow system responds to operating-system light and dark mode changes."}</p>
          </div>
          <div className="theme-options">
            {([
              ["auto", chinese ? "跟随系统" : "Follow system", chinese ? "自动响应系统设置" : "Use the operating-system setting"],
              ["dark", chinese ? "深色" : "Dark", chinese ? "适合低光环境" : "Designed for low-light environments"],
              ["light", chinese ? "浅色" : "Light", chinese ? "适合明亮环境" : "Designed for bright environments"],
            ] as const).map(([value, label, description]) => (
              <button className={`theme-option ${theme === value ? "selected" : ""}`} onClick={() => onThemeChange(value)} key={value}>
                <span className={`theme-preview ${value}`}><i /><i /><i /></span>
                <strong>{label}</strong>
                <small>{description}</small>
              </button>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
