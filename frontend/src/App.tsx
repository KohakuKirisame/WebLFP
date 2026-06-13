import { LineChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import { Scatter3DChart } from "echarts-gl/charts";
import { Grid3DComponent } from "echarts-gl/components";
import * as echarts from "echarts/core";
import type { EChartsCoreOption } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useMemo, useRef, useState } from "react";
import ChannelMultiSelect from "./ChannelMultiSelect";
import Guide from "./Guide";
import History, { type HistoricalInference } from "./History";
import type { Language } from "./i18n";
import Settings, { type ThemeMode } from "./Settings";

echarts.use([
  LineChart,
  ScatterChart,
  Scatter3DChart,
  DataZoomComponent,
  GridComponent,
  Grid3DComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

type Source = {
  path: string;
  format: string;
  sampling_rate_hz?: number;
  data_key?: string;
  channel_axis: "auto" | "first" | "last";
  stream_id?: string;
  electrical_series_path?: string;
  dtype?: string;
  num_channels?: number;
  time_axis?: 0 | 1;
  file_offset?: number;
};

type Metadata = {
  path: string;
  format: string;
  sampling_rate_hz: number;
  channel_ids: string[];
  num_channels: number;
  num_samples: number;
  duration_sec: number;
  dtype: string;
  num_segments: number;
  stream_id?: string;
};

type StreamOption = {
  stream_id: string;
  stream_name: string;
};

type StreamOptions = {
  format: string;
  streams: StreamOption[];
};

type ModelInfo = {
  id: string;
  display_name: string;
  checkpoint_available: boolean;
  checkpoint_sha256: string;
  epoch: number;
  target_sample_rate_hz: number;
  window_sec: number;
  hop_sec: number;
  embedding_dim: number;
  recommended_channels: number;
  max_channels: number;
  limitations: string[];
};

type Preview = {
  metadata: Metadata;
  channel_ids: string[];
  times_sec: number[];
  raw_traces: number[][];
  normalized_traces: number[][];
};

type Inference = {
  run_id: string;
  window_count: number;
  embedding_dim: number;
  device: string;
  source_sample_rate_hz: number;
  model_sample_rate_hz: number;
  selected_channel_ids: string[];
  window_start_sec: number[];
  umap_3d: number[][];
  umap_window_start_sec: number[];
  adjacent_cosine_similarity: number[];
  embedding_norm_min: number;
  embedding_norm_max: number;
  downstream?: SpikeTypeDecode | null;
  embeddings_download_url: string;
  run_download_url: string;
};

type InferenceJobStatus = {
  job_id: string;
  state: "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  message: string;
  result?: Inference;
  error?: string;
};

type SpikeTypeDecode = {
  decoder_id: string;
  display_name: string;
  device: string;
  labels: Array<{ id: string; name: string }>;
  window_sec: number;
  hop_sec: number;
  window_start_sec: number[];
  predicted_counts: number[][];
  rounded_counts: number[][];
  presence_probabilities: number[][];
  presence: boolean[][];
  mean_counts: Record<string, number>;
  total_predicted_counts: Record<string, number>;
  presence_rates: Record<string, number>;
  reference_metrics: Record<string, number>;
  limitations: string[];
};

type OperationProgress = {
  operation: "inspect" | "preview" | "infer";
  percent: number;
  label: string;
  detail: string;
};

const formats = [
  "auto",
  "npy",
  "npz",
  "mat",
  "binary",
  "spikeglx",
  "openephys",
  "intan",
  "plexon",
  "plexon2",
  "alphaomega",
  "nwb",
];
async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail ?? `Request failed: ${response.status}`);
  }
  return payload as T;
}

function Chart({ option, height = 320 }: { option: EChartsCoreOption; height?: number }) {
  const element = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!element.current) return;
    const chart = echarts.init(element.current, undefined, { renderer: "canvas" });
    chart.setOption(option);
    const resize = () => chart.resize();
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      chart.dispose();
    };
  }, [option]);

  return <div ref={element} style={{ height }} />;
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

function ProgressBar({
  progress,
  cancellable,
  onCancel,
  language,
}: {
  progress: OperationProgress;
  cancellable: boolean;
  onCancel: () => void;
  language: Language;
}) {
  return (
    <section className="operation-progress surface" aria-live="polite">
      <div className="operation-progress-heading">
        <div><span>{progress.label}</span><strong>{progress.detail}</strong></div>
        <div className="operation-progress-value">
          <span>{Math.round(progress.percent)}%</span>
          {cancellable && <button onClick={onCancel}>{language === "zh" ? "取消" : "Cancel"}</button>}
        </div>
      </div>
      <div className="progress-track"><i style={{ width: `${progress.percent}%` }} /></div>
    </section>
  );
}

function App() {
  const [page, setPage] = useState<"workspace" | "settings" | "guide" | "history">("workspace");
  const [language, setLanguage] = useState<Language>(() => (
    window.localStorage.getItem("weblfp-language") === "zh" ? "zh" : "en"
  ));
  const [environmentConfirmed, setEnvironmentConfirmed] = useState(
    () => window.localStorage.getItem("weblfp-environment-confirmed") === "1",
  );
  const [theme, setTheme] = useState<ThemeMode>(() => {
    const saved = window.localStorage.getItem("weblfp-theme");
    return saved === "dark" || saved === "light" || saved === "auto" ? saved : "auto";
  });
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [source, setSource] = useState<Source>({
    path: "",
    format: "auto",
    sampling_rate_hz: 1875,
    channel_axis: "auto",
    dtype: "float32",
    time_axis: 0,
    file_offset: 0,
  });
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [inference, setInference] = useState<Inference | null>(null);
  const [channels, setChannels] = useState<string[]>([]);
  const [startSec, setStartSec] = useState(0);
  const [endSec, setEndSec] = useState(0);
  const [device, setDevice] = useState("auto");
  const [previewMode, setPreviewMode] = useState<"raw" | "normalized">("normalized");
  const [streamOptions, setStreamOptions] = useState<StreamOption[]>([]);
  const [detectedFormat, setDetectedFormat] = useState("");
  const [busy, setBusy] = useState<"select" | "inspect" | "preview" | "infer" | null>(null);
  const [operationProgress, setOperationProgress] = useState<OperationProgress | null>(null);
  const [inferenceJobId, setInferenceJobId] = useState("");
  const [spikeTypeResult, setSpikeTypeResult] = useState<SpikeTypeDecode | null>(null);
  const [decoderBusy, setDecoderBusy] = useState(false);
  const [error, setError] = useState("");
  const chinese = language === "zh";

  useEffect(() => {
    requestJson<ModelInfo>("/api/model").then(setModel).catch((reason) => setError(reason.message));
  }, []);

  useEffect(() => {
    window.localStorage.setItem("weblfp-language", language);
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  }, [language]);

  function confirmEnvironment() {
    window.localStorage.setItem("weblfp-environment-confirmed", "1");
    setEnvironmentConfirmed(true);
  }

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const applyTheme = () => {
      const resolved = theme === "auto" ? (media.matches ? "dark" : "light") : theme;
      document.documentElement.dataset.theme = resolved;
      document.documentElement.dataset.themeMode = theme;
      window.localStorage.setItem("weblfp-theme", theme);
      document.querySelector('meta[name="theme-color"]')?.setAttribute(
        "content",
        resolved === "dark" ? "#0a1015" : "#eef3f5",
      );
    };
    applyTheme();
    media.addEventListener("change", applyTheme);
    return () => media.removeEventListener("change", applyTheme);
  }, [theme]);

  useEffect(() => {
    if (busy !== "inspect" && busy !== "preview") return;
    const cap = busy === "inspect" ? 88 : 92;
    const timer = window.setInterval(() => {
      setOperationProgress((current) => {
        if (!current || current.operation !== busy || current.percent >= cap) return current;
        return { ...current, percent: Math.min(cap, current.percent + Math.max(1, (cap - current.percent) * 0.08)) };
      });
    }, 350);
    return () => window.clearInterval(timer);
  }, [busy]);

  const selectedChannels = channels;

  const payloadSource = useMemo(() => {
    const payload: Record<string, unknown> = {
      path: source.path,
      format: source.format,
      channel_axis: source.channel_axis,
    };
    if (source.sampling_rate_hz) payload.sampling_rate_hz = source.sampling_rate_hz;
    if (source.data_key) payload.data_key = source.data_key;
    if (source.stream_id) payload.stream_id = source.stream_id;
    if (source.electrical_series_path) payload.electrical_series_path = source.electrical_series_path;
    if (source.dtype) payload.dtype = source.dtype;
    if (source.num_channels) payload.num_channels = source.num_channels;
    if (source.time_axis !== undefined) payload.time_axis = source.time_axis;
    if (source.file_offset !== undefined) payload.file_offset = source.file_offset;
    return payload;
  }, [source]);

  async function chooseRecordingFile() {
    setBusy("select");
    setError("");
    try {
      const value = await requestJson<{ path: string | null }>("/api/dialogs/recording-file", {
        method: "POST",
      });
      if (value.path) {
        setSource((current) => ({ ...current, path: value.path as string, stream_id: undefined }));
        setStreamOptions([]);
        setDetectedFormat("");
        setMetadata(null);
        setPreview(null);
        setInference(null);
        setChannels([]);
      }
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function inspect() {
    setBusy("inspect");
    setOperationProgress({ operation: "inspect", percent: 5, label: chinese ? "读取记录" : "Read recording", detail: chinese ? "正在检测格式和数据流。" : "Detecting format and data streams." });
    setError("");
    setInference(null);
    try {
      const available = await requestJson<StreamOptions>("/api/streams", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: payloadSource }),
      });
      setStreamOptions(available.streams);
      setDetectedFormat(available.format);
      setOperationProgress({ operation: "inspect", percent: 35, label: chinese ? "读取记录" : "Read recording", detail: chinese ? "数据流检测完成，正在读取元数据。" : "Streams detected. Reading metadata." });

      let selectedStreamId = source.stream_id;
      if (available.streams.length) {
        const selectedIsAvailable = available.streams.some((item) => item.stream_id === selectedStreamId);
        if (!selectedIsAvailable) {
          const lfpStream = available.streams.find(
            (item) => item.stream_name.trim().toLowerCase() === "lfp" || item.stream_id.trim().toLowerCase() === "lfp",
          );
          if (lfpStream) {
            selectedStreamId = lfpStream.stream_id;
          } else if (available.streams.length === 1) {
            selectedStreamId = available.streams[0].stream_id;
          } else {
            setSource((current) => ({ ...current, stream_id: undefined }));
            setError(chinese ? "检测到多个数据流。请在“数据流”中选择 LFP 对应的数据流，然后重新检查记录。" : "Multiple streams were detected. Select the LFP stream under Data stream, then inspect the recording again.");
            setOperationProgress(null);
            return;
          }
          setSource((current) => ({ ...current, stream_id: selectedStreamId }));
        }
      }

      const resolvedSource = selectedStreamId
        ? { ...payloadSource, stream_id: selectedStreamId }
        : payloadSource;
      const value = await requestJson<Metadata>("/api/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: resolvedSource }),
      });
      setMetadata(value);
      const recommended = value.channel_ids.slice(0, model?.recommended_channels ?? 4);
      setChannels(recommended);
      setStartSec(0);
      setEndSec(value.duration_sec);
      setOperationProgress({ operation: "inspect", percent: 100, label: chinese ? "读取记录" : "Read recording", detail: chinese ? "记录元数据读取完成。" : "Recording metadata is ready." });
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(null);
      window.setTimeout(() => setOperationProgress((current) => current?.operation === "inspect" ? null : current), 450);
    }
  }

  async function loadPreview() {
    const durationSec = endSec - startSec;
    if (!selectedChannels.length) {
      setError(chinese ? "请至少选择一个通道。" : "Select at least one channel.");
      return;
    }
    if (durationSec <= 0) {
      setError(chinese ? "结束时间必须大于开始时间。" : "End time must be greater than start time.");
      return;
    }
    setBusy("preview");
    setOperationProgress({ operation: "preview", percent: 6, label: chinese ? "读取与处理" : "Read and preprocess", detail: chinese ? "正在读取选定片段并执行预处理。" : "Reading and preprocessing the selected interval." });
    setError("");
    try {
      const value = await requestJson<Preview>("/api/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: payloadSource,
          start_sec: startSec,
          duration_sec: Math.max(0.2, durationSec),
          channel_ids: selectedChannels.length ? selectedChannels : null,
          max_points: 5000,
        }),
      });
      setPreview(value);
      setOperationProgress({ operation: "preview", percent: 100, label: chinese ? "读取与处理" : "Read and preprocess", detail: chinese ? "波形预处理完成。" : "Waveform preprocessing is complete." });
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(null);
      window.setTimeout(() => setOperationProgress((current) => current?.operation === "preview" ? null : current), 450);
    }
  }

  async function infer() {
    if (!selectedChannels.length) {
      setError(chinese ? "请至少选择一个通道。" : "Select at least one channel.");
      return;
    }
    setBusy("infer");
    setOperationProgress({ operation: "infer", percent: 1, label: chinese ? "生成 LFP feature" : "Generate LFP features", detail: chinese ? "正在创建推理任务。" : "Creating the inference job." });
    setError("");
    try {
      let job = await requestJson<InferenceJobStatus>("/api/inference-jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: payloadSource,
          start_sec: startSec,
          end_sec: endSec,
          channel_ids: selectedChannels.length ? selectedChannels : null,
          batch_size: 32,
          device,
        }),
      });
      setInferenceJobId(job.job_id);
      while (job.state === "queued" || job.state === "running") {
        setOperationProgress({
          operation: "infer",
          percent: job.progress,
          label: chinese ? "生成 LFP feature" : "Generate LFP features",
          detail: job.message,
        });
        await new Promise((resolve) => window.setTimeout(resolve, 350));
        job = await requestJson<InferenceJobStatus>(`/api/inference-jobs/${job.job_id}`);
      }
      if (job.state === "completed" && job.result) {
        setInference(job.result);
        setSpikeTypeResult(null);
        setOperationProgress({ operation: "infer", percent: 100, label: chinese ? "生成 LFP feature" : "Generate LFP features", detail: job.message });
      } else if (job.state === "cancelled") {
        setOperationProgress(null);
      } else {
        throw new Error(job.error ?? job.message);
      }
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(null);
      setInferenceJobId("");
      window.setTimeout(() => setOperationProgress((current) => current?.operation === "infer" ? null : current), 600);
    }
  }

  async function cancelInference() {
    if (!inferenceJobId) return;
    try {
      await requestJson<InferenceJobStatus>(`/api/inference-jobs/${inferenceJobId}`, { method: "DELETE" });
      setOperationProgress((current) => current ? { ...current, detail: chinese ? "正在取消推理任务…" : "Cancelling the inference job…" } : current);
    } catch (reason) {
      setError((reason as Error).message);
    }
  }

  function openHistoricalResult(value: HistoricalInference) {
    setInference(value);
    setSpikeTypeResult(value.downstream ?? null);
    setMetadata(null);
    setPreview(null);
    setError("");
    setPage("workspace");
  }

  async function runSpikeTypeDecode() {
    if (!inference) return;
    setDecoderBusy(true);
    setError("");
    try {
      const value = await requestJson<SpikeTypeDecode>(`/api/results/${inference.run_id}/type-decode`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ batch_size: 32, device }),
      });
      setSpikeTypeResult(value);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setDecoderBusy(false);
    }
  }

  const previewOption = useMemo<EChartsCoreOption | null>(() => {
    if (!preview) return null;
    const values = previewMode === "normalized" ? preview.normalized_traces : preview.raw_traces;
    return {
      animation: false,
      backgroundColor: "transparent",
      tooltip: { trigger: "axis", valueFormatter: (value: unknown) => Number(value).toFixed(3) },
      legend: { top: 0, textStyle: { color: "#82909d" } },
      grid: { left: 56, right: 18, top: 42, bottom: 76 },
      dataZoom: [
        {
          type: "inside",
          xAxisIndex: 0,
          filterMode: "none",
          zoomOnMouseWheel: true,
          moveOnMouseMove: true,
        },
        {
          type: "slider",
          xAxisIndex: 0,
          filterMode: "none",
          bottom: 10,
          height: 18,
          borderColor: "#25343e",
          backgroundColor: "rgba(11, 19, 25, 0.45)",
          fillerColor: "rgba(56, 198, 184, 0.18)",
          handleStyle: { color: "#38c6b8", borderColor: "#38c6b8" },
          textStyle: { color: "#71808d" },
        },
      ],
      xAxis: {
        type: "category",
        data: preview.times_sec.map((value) => value.toFixed(3)),
        name: chinese ? "时间 / s" : "Time / s",
        nameTextStyle: { color: "#71808d" },
        axisLabel: { color: "#71808d", interval: "auto" },
        axisLine: { lineStyle: { color: "#28333d" } },
      },
      yAxis: {
        type: "value",
        name: previewMode === "normalized" ? (chinese ? "归一化幅值" : "Normalized amplitude") : (chinese ? "原始幅值" : "Raw amplitude"),
        nameTextStyle: { color: "#71808d" },
        axisLabel: { color: "#71808d" },
        splitLine: { lineStyle: { color: "#1d2831" } },
      },
      series: values.map((trace, index) => ({
        type: "line",
        name: `Ch ${preview.channel_ids[index]}`,
        data: trace,
        showSymbol: false,
        lineStyle: { width: 1.2 },
        sampling: "lttb",
      })),
    };
  }, [chinese, preview, previewMode]);

  const umapOption = useMemo<EChartsCoreOption | null>(() => {
    if (!inference) return null;
    return {
      animation: false,
      tooltip: {
        formatter: (params: unknown) => {
          const item = params as { data?: number[]; dataIndex?: number };
          const index = item.dataIndex ?? 0;
          const time = inference.umap_window_start_sec[index];
          return `${chinese ? "可视化点" : "Visualization point"} ${index + 1}<br/>${time.toFixed(3)} s`;
        },
      },
      visualMap: {
        min: 0,
        max: Math.max(1, inference.umap_3d.length - 1),
        dimension: 3,
        show: false,
        inRange: { color: ["#23b5a9", "#6f7bf7", "#b56ee8"] },
      },
      xAxis3D: {
        type: "value",
        name: "UMAP 1",
        nameTextStyle: { color: "#82909d" },
        axisLabel: { color: "#71808d" },
        axisLine: { lineStyle: { color: "#35434e" } },
        splitLine: { lineStyle: { color: "#1d2831" } },
      },
      yAxis3D: {
        type: "value",
        name: "UMAP 2",
        nameTextStyle: { color: "#82909d" },
        axisLabel: { color: "#71808d" },
        axisLine: { lineStyle: { color: "#35434e" } },
        splitLine: { lineStyle: { color: "#1d2831" } },
      },
      zAxis3D: {
        type: "value",
        name: "UMAP 3",
        nameTextStyle: { color: "#82909d" },
        axisLabel: { color: "#71808d" },
        axisLine: { lineStyle: { color: "#35434e" } },
        splitLine: { lineStyle: { color: "#1d2831" } },
      },
      grid3D: {
        boxWidth: 120,
        boxHeight: 90,
        boxDepth: 120,
        environment: "transparent",
        viewControl: {
          projection: "perspective",
          autoRotate: false,
          distance: 190,
          rotateSensitivity: 1.2,
          zoomSensitivity: 1.2,
          panSensitivity: 1,
        },
        light: {
          main: { intensity: 1.1, shadow: false },
          ambient: { intensity: 0.7 },
        },
      },
      series: [{
        type: "scatter3D",
        symbolSize: 6,
        data: inference.umap_3d.map((point, index) => [point[0], point[1], point[2], index]),
        emphasis: { itemStyle: { color: "#ffffff" } },
      }],
    } as EChartsCoreOption;
  }, [chinese, inference]);

  const similarityOption = useMemo<EChartsCoreOption | null>(() => {
    if (!inference) return null;
    return {
      animation: false,
      tooltip: { trigger: "axis" },
      grid: { left: 52, right: 18, top: 18, bottom: 42 },
      xAxis: {
        type: "category",
        data: inference.window_start_sec.slice(1).map((value) => value.toFixed(3)),
        name: chinese ? "时间 / s" : "Time / s",
        axisLabel: { color: "#71808d" },
        axisLine: { lineStyle: { color: "#28333d" } },
      },
      yAxis: {
        type: "value",
        min: -1,
        max: 1,
        name: chinese ? "相邻余弦" : "Adjacent cosine",
        axisLabel: { color: "#71808d" },
        splitLine: { lineStyle: { color: "#1d2831" } },
      },
      series: [{
        type: "line",
        data: inference.adjacent_cosine_similarity,
        showSymbol: false,
        lineStyle: { color: "#38c6b8", width: 1.5 },
        areaStyle: { color: "rgba(56, 198, 184, 0.08)" },
      }],
    };
  }, [chinese, inference]);

  const presenceOption = useMemo<EChartsCoreOption | null>(() => {
    if (!spikeTypeResult) return null;
    return {
      animation: false,
      tooltip: { trigger: "axis" },
      legend: { top: 0, textStyle: { color: "#82909d" } },
      grid: { left: 52, right: 18, top: 42, bottom: 42 },
      xAxis: {
        type: "category",
        data: spikeTypeResult.window_start_sec.map((value) => value.toFixed(3)),
        name: chinese ? "时间 / s" : "Time / s",
        axisLabel: { color: "#71808d" },
        axisLine: { lineStyle: { color: "#28333d" } },
      },
      yAxis: {
        type: "value",
        min: 0,
        max: 1,
        name: chinese ? "Presence 概率" : "Presence probability",
        axisLabel: { color: "#71808d" },
        splitLine: { lineStyle: { color: "#1d2831" } },
      },
      series: spikeTypeResult.labels.map((label, labelIndex) => ({
        type: "line",
        name: label.id === "narrow" ? (chinese ? "窄波" : "Narrow") : label.id === "non_narrow" ? (chinese ? "非窄波" : "Non-narrow") : label.name,
        data: spikeTypeResult.presence_probabilities.map((row) => row[labelIndex]),
        showSymbol: false,
        lineStyle: { width: 1.5 },
      })),
    };
  }, [chinese, spikeTypeResult]);

  const countOption = useMemo<EChartsCoreOption | null>(() => {
    if (!spikeTypeResult) return null;
    return {
      animation: false,
      tooltip: { trigger: "axis" },
      legend: { top: 0, textStyle: { color: "#82909d" } },
      grid: { left: 52, right: 18, top: 42, bottom: 42 },
      xAxis: {
        type: "category",
        data: spikeTypeResult.window_start_sec.map((value) => value.toFixed(3)),
        name: chinese ? "时间 / s" : "Time / s",
        axisLabel: { color: "#71808d" },
        axisLine: { lineStyle: { color: "#28333d" } },
      },
      yAxis: {
        type: "value",
        min: 0,
        name: chinese ? "预测 count / window" : "Predicted count / window",
        axisLabel: { color: "#71808d" },
        splitLine: { lineStyle: { color: "#1d2831" } },
      },
      series: spikeTypeResult.labels.map((label, labelIndex) => ({
        type: "line",
        name: label.id === "narrow" ? (chinese ? "窄波" : "Narrow") : label.id === "non_narrow" ? (chinese ? "非窄波" : "Non-narrow") : label.name,
        data: spikeTypeResult.predicted_counts.map((row) => row[labelIndex]),
        showSymbol: false,
        lineStyle: { width: 1.5 },
      })),
    };
  }, [chinese, spikeTypeResult]);

  const step = inference ? 4 : preview ? 2 : metadata ? 2 : 1;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">L</span>
          <div>
            <strong>WebLFP</strong>
            <span>Unified LFP feature and spike activity decoding</span>
          </div>
        </div>
        <div className="topbar-actions">
          <div className="model-status">
            <span className={model?.checkpoint_available ? "status-dot online" : "status-dot"} />
            {model?.checkpoint_available ? (chinese ? "最佳权重已就绪" : "Verified model ready") : (chinese ? "权重不可用" : "Model unavailable")}
          </div>
          <button className={`settings-button ${page === "history" ? "active" : ""}`} onClick={() => setPage(page === "history" ? "workspace" : "history")}>
            {page === "history" ? (chinese ? "工作区" : "Workspace") : (chinese ? "历史" : "History")}
          </button>
          <button className={`settings-button ${page === "guide" ? "active" : ""}`} onClick={() => setPage(page === "guide" ? "workspace" : "guide")}>
            {page === "guide" ? (chinese ? "工作区" : "Workspace") : (chinese ? "说明" : "Guide")}
          </button>
          <button className={`settings-button ${page === "settings" ? "active" : ""}`} onClick={() => setPage(page === "settings" ? "workspace" : "settings")}>
            {page === "settings" ? (chinese ? "工作区" : "Workspace") : (chinese ? "设置" : "Settings")}
          </button>
        </div>
      </header>

      {page === "settings" ? (
        <Settings
          theme={theme}
          language={language}
          environmentConfirmed={environmentConfirmed}
          onThemeChange={setTheme}
          onLanguageChange={setLanguage}
          onEnvironmentConfirm={confirmEnvironment}
          onBack={() => setPage("workspace")}
        />
      ) : page === "history" ? (
        <History language={language} onBack={() => setPage("workspace")} onSelect={openHistoricalResult} />
      ) : page === "guide" ? (
        <Guide language={language} onBack={() => setPage("workspace")} />
      ) : (
        <div className="workspace">
        <aside className="step-rail">
          {(chinese ? ["数据", "处理", "模型", "结果"] : ["Data", "Process", "Model", "Result"]).map((name, index) => (
            <div className={`step ${step >= index + 1 ? "active" : ""}`} key={name}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <strong>{name}</strong>
            </div>
          ))}
          <div className="rail-note">
            <span>{chinese ? "运行模式" : "RUN MODE"}</span>
            <strong>LFP only</strong>
            <p>{chinese ? "Spike 仅用于预训练，不参与当前推理。" : "Spike was used during pretraining and is not required for this inference."}</p>
          </div>
        </aside>

        <main className="canvas">
          <section className="canvas-heading">
            <div>
              <span className="eyebrow">LOCAL INFERENCE WORKSPACE</span>
              <h1>{inference ? (chinese ? "LFP feature 结果" : "LFP feature result") : preview ? (chinese ? "信号预览" : "Signal preview") : (chinese ? "导入 LFP 记录" : "Import an LFP recording")}</h1>
            </div>
            {metadata && (
              <div className="recording-pill">
                <span>{metadata.format.toUpperCase()}</span>
                <strong>{metadata.num_channels} ch · {metadata.duration_sec.toFixed(2)} s</strong>
              </div>
            )}
          </section>

          {error && <div className="error-banner">{error}</div>}
          {operationProgress && (
            <ProgressBar
              progress={operationProgress}
              cancellable={operationProgress.operation === "infer" && inferenceJobId !== ""}
              onCancel={() => void cancelInference()}
              language={language}
            />
          )}

          {!metadata && !inference && (
            <section className="empty-state surface">
              <div className="signal-glyph">
                <i /><i /><i /><i /><i /><i /><i />
              </div>
              <h2>{chinese ? "从本地记录开始" : "Start with a local recording"}</h2>
              <p>{chinese ? "输入文件或记录目录路径，系统先读取元数据，不会立即载入完整记录。" : "Enter a file or recording-directory path. WebLFP reads metadata before loading signal data."}</p>
              <button className="primary-button" disabled={!source.path || busy !== null} onClick={inspect}>
                {busy === "inspect" ? (chinese ? "正在读取…" : "Reading…") : (chinese ? "检查记录" : "Inspect recording")}
              </button>
            </section>
          )}

          {metadata && !preview && (
            <section className="metadata-grid">
              {[
                [chinese ? "采样率" : "Sampling rate", `${metadata.sampling_rate_hz.toLocaleString()} Hz`],
                [chinese ? "通道" : "Channels", `${metadata.num_channels}`],
                [chinese ? "时长" : "Duration", `${metadata.duration_sec.toFixed(3)} s`],
                [chinese ? "数据类型" : "Data type", metadata.dtype],
              ].map(([label, value]) => (
                <div className="metric-card surface" key={label}>
                  <span>{label}</span>
                  <strong>{value}</strong>
                </div>
              ))}
              <div className="surface metadata-detail">
                <span>{chinese ? "数据路径" : "Recording path"}</span>
                <code>{metadata.path}</code>
                <span>{chinese ? "已选择通道" : "Selected channels"}</span>
                <code>{selectedChannels.join(", ") || (chinese ? "未选择" : "None")}</code>
              </div>
              <button className="primary-button wide" disabled={busy !== null} onClick={loadPreview}>
                {busy === "preview" ? (chinese ? "生成预览…" : "Preparing preview…") : (chinese ? "预览处理结果" : "Preview processed signal")}
              </button>
            </section>
          )}

          {preview && !inference && previewOption && (
            <>
              <section className="surface chart-card">
                <div className="section-title">
                  <div><span>PREPROCESSING PREVIEW</span><h2>{chinese ? "Robust z-score 后波形" : "Waveform after robust z-score"}</h2></div>
                  <div className="preview-toggle">
                    <button className={previewMode === "raw" ? "selected" : ""} onClick={() => setPreviewMode("raw")}>{chinese ? "原始" : "Raw"}</button>
                    <button className={previewMode === "normalized" ? "selected" : ""} onClick={() => setPreviewMode("normalized")}>{chinese ? "处理后" : "Processed"}</button>
                  </div>
                </div>
                <Chart option={previewOption} height={390} />
                <p className="chart-help">{chinese ? "拖动下方滑块或在图中滚动鼠标滚轮以缩放时间轴。" : "Drag the slider or use the mouse wheel over the chart to zoom the time axis."}</p>
              </section>
              <section className="process-strip surface">
                {(chinese ? ["读取", "通道选择", "重采样至 1875 Hz", "0.2 s 窗口", "Robust z-score"] : ["Read", "Select channels", "Resample to 1875 Hz", "0.2 s windows", "Robust z-score"]).map((item, index) => (
                  <div key={item}><span>{index + 1}</span><strong>{item}</strong></div>
                ))}
              </section>
              <button className="primary-button run-button" disabled={busy !== null} onClick={infer}>
                {busy === "infer" ? (chinese ? "正在生成 LFP feature…" : "Generating LFP features…") : (chinese ? "运行 LFP-only 推理" : "Run LFP-only inference")}
              </button>
            </>
          )}

          {inference && umapOption && similarityOption && (
            <>
              <section className="result-metrics">
                <div className="metric-card surface"><span>{chinese ? "窗口" : "Windows"}</span><strong>{inference.window_count}</strong></div>
                <div className="metric-card surface"><span>{chinese ? "LFP feature 维度" : "LFP feature dimension"}</span><strong>{inference.embedding_dim}</strong></div>
                <div className="metric-card surface"><span>{chinese ? "设备" : "Device"}</span><strong>{inference.device.toUpperCase()}</strong></div>
                <div className="metric-card surface"><span>{chinese ? "特征范数" : "Feature norm"}</span><strong>{inference.embedding_norm_min.toFixed(4)}–{inference.embedding_norm_max.toFixed(4)}</strong></div>
              </section>
              <section className="chart-grid result-chart-stack">
                <div className="surface chart-card">
                  <div className="section-title"><div><span>FEATURE GEOMETRY</span><h2>{chinese ? "UMAP 三维表征空间" : "Three-dimensional UMAP feature space"}</h2></div></div>
                  <Chart option={umapOption} height={480} />
                  <p className="chart-help">{chinese ? "按住鼠标拖动以旋转坐标，滚动滚轮缩放，右键拖动平移。最多均匀显示 5000 个窗口。" : "Drag to rotate, use the mouse wheel to zoom, and right-drag to pan. Up to 5,000 windows are sampled evenly for display."}</p>
                </div>
                <div className="surface chart-card">
                  <div className="section-title"><div><span>TEMPORAL CONSISTENCY</span><h2>{chinese ? "相邻窗口余弦相似度" : "Adjacent-window cosine similarity"}</h2></div></div>
                  <Chart option={similarityOption} height={340} />
                </div>
              </section>
              <section className="export-bar surface">
                <div><span>RUN ID</span><code>{inference.run_id}</code></div>
                <a href={inference.embeddings_download_url}>{chinese ? "下载" : "Download"} embeddings.npz</a>
                <a href={inference.run_download_url}>{chinese ? "下载" : "Download"} run.json</a>
              </section>
              <section className="downstream-panel surface">
                <div className="section-title">
                  <div><span>DOWNSTREAM TASK</span><h2>{chinese ? "窄波 / 非窄波活动解码" : "Narrow / non-narrow activity decoding"}</h2></div>
                  <span className="tag">SAME CHECKPOINT HEAD</span>
                </div>
                <p className="downstream-note">
                  {chinese
                    ? "当前 LFP 推理已使用统一权重生成 256 维 feature。这里直接把这些 feature 接入同一权重中的 SpikeCountPresenceHead，按原项目任务定义估计窄波与非窄波的 presence 概率和 spike count。"
                    : "The unified checkpoint generated these 256D LFP features. They are passed directly to the same checkpoint's SpikeCountPresenceHead to estimate narrow and non-narrow presence probability and spike count."}
                </p>
                <div className="downstream-actions">
                  <span className="downstream-note">{chinese ? "设备沿用当前推理设置" : "Uses the current inference device"}: {device.toUpperCase()}</span>
                  <button className="primary-button" disabled={decoderBusy} onClick={() => void runSpikeTypeDecode()}>
                    {decoderBusy ? (chinese ? "正在解码…" : "Decoding…") : (chinese ? "运行 presence 与 count" : "Run presence and count")}
                  </button>
                </div>
                <div className="compatibility-warning">
                  {chinese
                    ? "这是窗口级活动代理，不是单神经元真实细胞类型。结果来自当前 run 已保存的 256 维 LFP feature，不会重新读取原始记录，也不会使用未训练的线性适配层。"
                    : "This is a window-level activity proxy, not true single-neuron cell typing. It uses the current run's saved 256D LFP features without reopening the recording or using an untrained linear adapter."}
                </div>
                {spikeTypeResult && presenceOption && countOption && (
                  <>
                    <div className="downstream-summary">
                      {spikeTypeResult.labels.map((label) => (
                        <div key={label.id}>
                          <span>{label.id === "narrow" ? (chinese ? "窄波" : "Narrow") : label.id === "non_narrow" ? (chinese ? "非窄波" : "Non-narrow") : label.name}</span>
                          <strong>{((spikeTypeResult.presence_rates[label.id] ?? 0) * 100).toFixed(1)}%</strong>
                          <small>{chinese ? "平均 count" : "Mean count"} {(spikeTypeResult.mean_counts[label.id] ?? 0).toFixed(3)} / {chinese ? "窗口" : "window"}</small>
                        </div>
                      ))}
                      <div>
                        <span>{chinese ? "论文验证集参考" : "Paper validation reference"}</span>
                        <strong>{((spikeTypeResult.reference_metrics.presence_micro_accuracy ?? 0) * 100).toFixed(2)}%</strong>
                        <small>presence micro accuracy</small>
                      </div>
                      <div>
                        <span>{chinese ? "论文验证集参考" : "Paper validation reference"}</span>
                        <strong>{((spikeTypeResult.reference_metrics.total_count_exact_accuracy ?? 0) * 100).toFixed(2)}%</strong>
                        <small>total count exact accuracy</small>
                      </div>
                    </div>
                    <div className="downstream-chart-grid">
                      <div className="downstream-chart">
                        <div><span>{spikeTypeResult.display_name}</span><strong>{chinese ? "逐窗口 presence 概率" : "Per-window presence probability"}</strong></div>
                        <Chart option={presenceOption} height={300} />
                      </div>
                      <div className="downstream-chart">
                        <div><span>COUNT REGRESSION</span><strong>{chinese ? "逐窗口预测 count" : "Per-window predicted count"}</strong></div>
                        <Chart option={countOption} height={300} />
                      </div>
                    </div>
                    {(chinese ? spikeTypeResult.limitations : [
                      "The 256D CLS feature comes from feature_extractor.forward_features() and is passed directly to the bundled presence/count head.",
                      "Outputs estimate narrow and non-narrow activity per 0.2-second window; they are not true single-neuron cell-type labels.",
                      "Reference metrics come from KA chronic epilepsy mouse hippocampal data and may not generalize to other distributions.",
                    ]).map((limitation) => (
                      <p className="scientific-note" key={limitation}>{limitation}</p>
                    ))}
                  </>
                )}
              </section>
            </>
          )}
        </main>

        <aside className="parameter-panel">
          <div className="panel-heading"><span>PARAMETERS</span><h2>{chinese ? "输入与模型" : "Input and model"}</h2></div>
          <Field label={chinese ? "本地路径" : "Local path"} hint={chinese ? "文件或记录目录，数据不会上传外部服务。" : "File or recording directory. Data is not uploaded to an external service."}>
            <div className="path-picker">
              <input value={source.path} onChange={(event) => {
                setSource({ ...source, path: event.target.value, stream_id: undefined });
                setStreamOptions([]);
                setDetectedFormat("");
                setMetadata(null);
                setPreview(null);
                setInference(null);
                setChannels([]);
              }} placeholder={chinese ? "选择记录文件或输入路径" : "Select a recording or enter a path"} />
              <button type="button" onClick={() => void chooseRecordingFile()} disabled={busy !== null}>
                {busy === "select" ? (chinese ? "选择中…" : "Selecting…") : (chinese ? "选择文件" : "Select file")}
              </button>
            </div>
          </Field>
          <div className="field-row">
            <Field label={chinese ? "格式" : "Format"}>
              <select value={source.format} onChange={(event) => {
                setSource({ ...source, format: event.target.value, stream_id: undefined });
                setStreamOptions([]);
                setDetectedFormat("");
                setMetadata(null);
                setPreview(null);
                setInference(null);
                setChannels([]);
              }}>
                {formats.map((format) => <option key={format}>{format}</option>)}
              </select>
            </Field>
            <Field label={chinese ? "通道轴" : "Channel axis"}>
              <select value={source.channel_axis} onChange={(event) => setSource({ ...source, channel_axis: event.target.value as Source["channel_axis"] })}>
                <option value="auto">{chinese ? "自动" : "Auto"}</option><option value="first">{chinese ? "第一维" : "First dimension"}</option><option value="last">{chinese ? "最后一维" : "Last dimension"}</option>
              </select>
            </Field>
          </div>
          <Field label={chinese ? "源采样率 / Hz" : "Source sampling rate / Hz"} hint={chinese ? "NumPy、MAT 和 raw binary 必填。" : "Required for NumPy, MATLAB, and raw binary inputs."}>
            <input type="number" value={source.sampling_rate_hz ?? ""} onChange={(event) => setSource({ ...source, sampling_rate_hz: Number(event.target.value) })} />
          </Field>
          {(source.format === "npz" || source.format === "mat") && (
            <Field label={chinese ? "数组键" : "Array key"} hint={chinese ? "文件包含多个数组时必填。" : "Required when the file contains multiple arrays."}>
              <input value={source.data_key ?? ""} onChange={(event) => setSource({ ...source, data_key: event.target.value })} placeholder="lfp" />
            </Field>
          )}
          {(streamOptions.length > 0 || ["spikeglx", "openephys", "intan", "plexon", "plexon2", "alphaomega"].includes(source.format) || ["spikeglx", "openephys", "intan", "plexon", "plexon2"].includes(detectedFormat)) && (
            <Field label={chinese ? "数据流" : "Data stream"} hint={chinese ? "PLX 可能同时包含 LFP 和 SPK；LFP-only 推理请选择 LFP。流 ID 中的空格会被完整保留。" : "PLX may contain both LFP and SPK streams. Select LFP for LFP-only inference. Spaces in stream IDs are preserved."}>
              {streamOptions.length > 0 ? (
                <select value={source.stream_id ?? ""} onChange={(event) => setSource({ ...source, stream_id: event.target.value || undefined })}>
                  <option value="">{chinese ? "请选择数据流" : "Select a data stream"}</option>
                  {streamOptions.map((item) => (
                    <option key={item.stream_id} value={item.stream_id}>
                      {(item.stream_name.trim() || (chinese ? "未命名流" : "Unnamed stream"))} · ID {JSON.stringify(item.stream_id)}
                    </option>
                  ))}
                </select>
              ) : (
                <input value={source.stream_id ?? ""} onChange={(event) => setSource({ ...source, stream_id: event.target.value || undefined })} placeholder={chinese ? "检查记录后可选择" : "Available after recording inspection"} />
              )}
            </Field>
          )}
          {source.format === "nwb" && (
            <Field label={chinese ? "ElectricalSeries 路径" : "ElectricalSeries path"} hint={chinese ? "NWB 中存在多个序列时填写。" : "Required when an NWB file contains multiple series."}>
              <input value={source.electrical_series_path ?? ""} onChange={(event) => setSource({ ...source, electrical_series_path: event.target.value })} />
            </Field>
          )}
          {source.format === "binary" && (
            <div className="advanced-block">
              <span>RAW BINARY</span>
              <div className="field-row">
                <Field label="dtype"><input value={source.dtype ?? ""} onChange={(event) => setSource({ ...source, dtype: event.target.value })} /></Field>
                <Field label={chinese ? "通道数" : "Channel count"}><input type="number" value={source.num_channels ?? ""} onChange={(event) => setSource({ ...source, num_channels: Number(event.target.value) })} /></Field>
              </div>
              <div className="field-row">
                <Field label={chinese ? "时间轴" : "Time axis"}>
                  <select value={source.time_axis ?? 0} onChange={(event) => setSource({ ...source, time_axis: Number(event.target.value) as 0 | 1 })}>
                    <option value={0}>0 / {chinese ? "行优先" : "row-major"}</option><option value={1}>1 / {chinese ? "列优先" : "column-major"}</option>
                  </select>
                </Field>
                <Field label={chinese ? "文件偏移 / byte" : "File offset / byte"}><input type="number" value={source.file_offset ?? 0} onChange={(event) => setSource({ ...source, file_offset: Number(event.target.value) })} /></Field>
              </div>
            </div>
          )}
          <div className="time-range-field">
            <div className="field-row">
              <Field label={chinese ? "开始 / s" : "Start / s"}><input type="number" step="0.1" value={startSec} onChange={(event) => setStartSec(Number(event.target.value))} /></Field>
              <Field label={chinese ? "结束 / s" : "End / s"}><input type="number" step="0.1" value={endSec} onChange={(event) => setEndSec(Number(event.target.value))} /></Field>
            </div>
            <span className="field-hint">{chinese ? "检查记录后默认选择完整记录；可手动缩短处理范围。" : "The full recording is selected after inspection. You can shorten the processing range manually."}</span>
          </div>
          <Field label={chinese ? "通道 ID" : "Channel IDs"} hint={chinese ? "搜索并多选记录中实际存在的通道；检查记录后默认选择推荐通道。" : "Search and select channel IDs found in the recording. Recommended channels are selected after inspection."}>
            <ChannelMultiSelect
              options={metadata?.channel_ids ?? []}
              value={channels}
              maxSelected={model?.max_channels ?? 16}
              chinese={chinese}
              onChange={setChannels}
            />
          </Field>
          <Field label={chinese ? "推理设备" : "Inference device"}>
            <select value={device} onChange={(event) => setDevice(event.target.value)}>
              <option value="auto">{chinese ? "自动" : "Auto"}</option><option value="cpu">CPU</option><option value="cuda">CUDA</option>
            </select>
          </Field>

          <div className="model-card">
            <span className="model-kicker">LOCKED MODEL PROFILE</span>
            <h3>{model?.display_name ?? (chinese ? "读取模型…" : "Loading model…")}</h3>
            <div className="model-values">
              <div><span>Epoch</span><strong>{model?.epoch ?? "-"}</strong></div>
              <div><span>Window</span><strong>{model ? `${model.window_sec} s` : "-"}</strong></div>
              <div><span>Rate</span><strong>{model ? `${model.target_sample_rate_hz} Hz` : "-"}</strong></div>
              <div><span>Feature</span><strong>{model ? `${model.embedding_dim} D` : "-"}</strong></div>
            </div>
            <code>{model ? model.checkpoint_sha256.slice(0, 16) : "-"}…</code>
          </div>

          {metadata && <button className="secondary-button" onClick={inspect} disabled={busy !== null}>{chinese ? "重新读取元数据" : "Inspect metadata again"}</button>}
          <p className="scientific-note">{chinese ? "该 LFP feature 与参考分类头仅在报告所述数据与任务上验证，不用于重建 Spike 波形。" : "The LFP feature and reference head are validated only for the reported data and tasks. They do not reconstruct Spike waveforms."}</p>
        </aside>
        </div>
      )}
      {!environmentConfirmed && page !== "settings" && (
        <div className="setup-overlay" role="dialog" aria-modal="true" aria-labelledby="setup-title">
          <section className="setup-dialog surface">
            <span className="eyebrow">FIRST-RUN SETUP</span>
            <h2 id="setup-title">{chinese ? "请先配置运行环境" : "Configure the runtime environment first"}</h2>
            <p>{chinese ? "首次使用前，请在设置页检查硬件，并根据 GPU 选择 CUDA、ROCm 或 CPU PyTorch。确认当前环境后才能返回工作区。" : "Before first use, open Settings, inspect the hardware, and choose CUDA, ROCm, or CPU PyTorch for this machine. Confirm the current environment before returning to the workspace."}</p>
            <button className="primary-button" onClick={() => setPage("settings")}>{chinese ? "前往设置" : "Open settings"}</button>
          </section>
        </div>
      )}
    </div>
  );
}

export default App;
