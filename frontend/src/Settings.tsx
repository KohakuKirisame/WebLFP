import { useEffect, useMemo, useState } from "react";

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
    version: string;
    backend: "cpu" | "cuda" | "rocm";
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

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail ?? `Request failed: ${response.status}`);
  return payload as T;
}

function formatMemory(mebibytes: number): string {
  return `${(mebibytes / 1024).toFixed(1)} GiB`;
}

function formatElapsed(seconds = 0): string {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return minutes ? `${minutes} 分 ${remainder} 秒` : `${remainder} 秒`;
}

export default function Settings({
  theme,
  onThemeChange,
  onBack,
}: {
  theme: ThemeMode;
  onThemeChange: (theme: ThemeMode) => void;
  onBack: () => void;
}) {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [options, setOptions] = useState<PyTorchOption[]>([]);
  const [selectedOption, setSelectedOption] = useState("");
  const [status, setStatus] = useState<InstallStatus | null>(null);
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

  useEffect(() => {
    void refresh();
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
          message: "WebLFP 已停止，正在安装 PyTorch 并等待服务重启…",
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
      `安装 ${selected.label}？\n\nWebLFP 将停止，替换当前 PyTorch 后自动重启。独立进度窗口会显示安装日志，关闭该窗口不会中断后台安装。`,
    );
    if (!confirmed) return;
    setError("");
    setStatus({ state: "waiting", message: "正在安排安装任务…" });
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
        setStatus({ state: "waiting", message: "服务正在停止，安装即将开始…" });
      } else {
        setError(message);
        setStatus(null);
      }
    }
  }

  return (
    <div className="settings-shell">
      <aside className="settings-sidebar">
        <button className="back-button" onClick={onBack}>← 返回工作区</button>
        <span className="settings-label">SETTINGS</span>
        <a href="#compute">计算环境</a>
        <a href="#appearance">外观</a>
        <div className="settings-sidebar-note">
          安装仅使用 PyTorch 官方 wheel 索引，不接受自定义命令或下载地址。
        </div>
      </aside>

      <main className="settings-content">
        <header className="settings-heading">
          <div><span>APPLICATION SETTINGS</span><h1>设置</h1></div>
          <button className="secondary-button compact" onClick={() => void refresh(true)} disabled={loading}>
            {loading ? "检测中…" : "重新检测"}
          </button>
        </header>

        {error && <div className="error-banner">{error}</div>}

        <section id="compute" className="settings-section">
          <div className="settings-section-title">
            <div><span>COMPUTE ENVIRONMENT</span><h2>计算环境</h2></div>
            <p>CUDA wheel 最低为 13.0；ROCm 官方 wheel 仅支持 Linux。</p>
          </div>

          {system && (system.performance.warnings.length ? (
            <div className="performance-alert poor">
              <div><span>PERFORMANCE WARNING</span><strong>机器性能过差</strong></div>
              <ul>{system.performance.warnings.map((warning) => <li key={warning.code}>{warning.message}</li>)}</ul>
            </div>
          ) : (
            <div className="performance-alert good">
              <div><span>PERFORMANCE CHECK</span><strong>当前性能门槛已通过</strong></div>
              <p>{system.pytorch.backend === "cpu" ? "当前为 CPU 模式，已按规则忽略显存和 CUDA BF16 项。" : "内存、显存和当前计算后端均未触发性能告警。"}</p>
            </div>
          ))}

          <div className="hardware-grid">
            <div className="hardware-card surface">
              <span>SYSTEM MEMORY</span>
              <strong>{system?.memory.installed_gib ? `${system.memory.installed_gib.toFixed(1)} GiB` : "未检测"}</strong>
              <p>{system?.memory.configured_speed_mt_s ? `${system.memory.configured_speed_mt_s} MT/s configured · ${system.memory.modules.length || "?"} module(s)` : "有效传输率未检测"}</p>
            </div>
            <div className="hardware-card surface">
              <span>GPU / DRIVER</span>
              <strong>{system?.nvidia.gpus[0]?.name ?? "未检测到 NVIDIA GPU"}</strong>
              <p>{system?.nvidia.gpus[0] ? `${formatMemory(system.nvidia.gpus[0].memory_mib)} · Driver ${system.nvidia.gpus[0].driver_version}` : "CPU 模式仍可用"}</p>
            </div>
            <div className="hardware-card surface">
              <span>CUDA</span>
              <strong>{system?.nvidia.driver_cuda_version ? `Driver ${system.nvidia.driver_cuda_version}` : "不可用"}</strong>
              <p>{system?.nvidia.toolkit_cuda_version ? `Toolkit ${system.nvidia.toolkit_cuda_version}` : "未检测到 nvcc"}</p>
            </div>
            <div className="hardware-card surface">
              <span>PYTORCH</span>
              <strong>{system?.pytorch.version ?? "检测中"}</strong>
              <p>{system ? `${system.pytorch.backend.toUpperCase()} build · CUDA available ${system.pytorch.cuda_available ? "yes" : "no"}` : "-"}</p>
            </div>
            <div className="hardware-card surface">
              <span>cuDNN / ROCm</span>
              <strong>{system?.pytorch.cudnn_version ? `cuDNN ${system.pytorch.cudnn_version}` : system?.system_cudnn_version ? `System cuDNN ${system.system_cudnn_version}` : "cuDNN 不可用"}</strong>
              <p>{system?.rocm.platform_supported ? (system.rocm.detected ? `ROCm ${system.rocm.version ?? "detected"}` : "ROCm 未检测到") : "ROCm 仅支持 Linux"}</p>
            </div>
            <div className="hardware-card surface">
              <span>CUDA BF16</span>
              <strong>{system?.pytorch.backend === "cpu" ? "CPU 模式已忽略" : system?.cuda_bf16.measured_tflops ? `${system.cuda_bf16.measured_tflops.toFixed(1)} TFLOP/s` : system?.cuda_bf16.supported === false ? "不支持原生 BF16" : "未完成检测"}</strong>
              <p>RTX 4070 级实测下限 {system?.cuda_bf16.reference_floor_tflops ?? 60} TFLOP/s</p>
            </div>
          </div>

          <div className="install-panel surface">
            <div className="install-heading">
              <div><span>PYTORCH RUNTIME</span><h3>选择 PyTorch 版本与计算后端</h3></div>
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
                      <small>{option.compatible ? option.compatibility_reason : `不可用：${option.compatibility_reason}`}</small>
                    </button>
                  ))}
                </div>
              </div>
            ))}

            <div className="install-action">
              <div>
                <span>将安装</span>
                <strong>{selected?.label ?? "请选择兼容版本"}</strong>
                <code>{selected?.index_url ?? "-"}</code>
              </div>
              <button className="primary-button" disabled={!selected?.compatible || ["waiting", "installing"].includes(status?.state ?? "")} onClick={() => void install()}>
                {["waiting", "installing"].includes(status?.state ?? "") ? "安装进行中…" : "确认并安装"}
              </button>
            </div>

            {status && status.state !== "idle" && (
              <div className={`install-status ${status.state}`}>
                <div className="install-status-summary">
                  <span>{status.state.toUpperCase()}</span>
                  <p>{status.message}</p>
                  <small>已用时 {formatElapsed(status.elapsed_sec)}</small>
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

        <section id="appearance" className="settings-section">
          <div className="settings-section-title">
            <div><span>APPEARANCE</span><h2>界面外观</h2></div>
            <p>“跟随系统”会实时响应操作系统的深浅色设置。</p>
          </div>
          <div className="theme-options">
            {([
              ["auto", "跟随系统", "自动响应系统设置"],
              ["dark", "深色", "适合低光环境"],
              ["light", "浅色", "适合明亮环境"],
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
