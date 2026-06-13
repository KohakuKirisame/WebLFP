import { useEffect, useState } from "react";
import type { Language } from "./i18n";

export type HistoricalInference = {
  run_id: string;
  window_count: number;
  embedding_dim: number;
  device: string;
  source_sample_rate_hz: number;
  model_sample_rate_hz: number;
  selected_channel_ids: string[];
  window_start_sec: number[];
  pca_2d: number[][];
  adjacent_cosine_similarity: number[];
  embedding_norm_min: number;
  embedding_norm_max: number;
  downstream?: {
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
  } | null;
  embeddings_download_url: string;
  run_download_url: string;
};

type RunSummary = {
  run_id: string;
  created_at: string;
  source_name: string;
  source_path: string;
  start_sec: number;
  end_sec: number;
  window_count: number;
  embedding_dim: number;
  device: string;
};

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail ?? `Request failed: ${response.status}`);
  return payload as T;
}

export default function History({
  language,
  onBack,
  onSelect,
}: {
  language: Language;
  onBack: () => void;
  onSelect: (result: HistoricalInference) => void;
}) {
  const chinese = language === "zh";
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [opening, setOpening] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    getJson<RunSummary[]>("/api/results")
      .then(setRuns)
      .catch((reason) => setError((reason as Error).message))
      .finally(() => setLoading(false));
  }, []);

  async function openRun(runId: string) {
    setOpening(runId);
    setError("");
    try {
      onSelect(await getJson<HistoricalInference>(`/api/results/${runId}`));
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setOpening("");
    }
  }

  return (
    <main className="history-shell">
      <header className="history-heading">
        <div>
          <span>LOCAL RUN HISTORY</span>
          <h1>{chinese ? "历史记录" : "Run history"}</h1>
          <p>{chinese ? "已完成的 runs 保存在本机，可重新查看 LFP feature 图形并再次导出结果。" : "Completed runs remain on this machine. Reopen feature plots and export results again."}</p>
        </div>
        <button className="back-button" onClick={onBack}>{chinese ? "返回工作区" : "Back to workspace"}</button>
      </header>

      {error && <div className="error-banner">{error}</div>}
      {loading ? (
        <section className="history-empty surface">{chinese ? "正在读取历史记录…" : "Loading run history…"}</section>
      ) : runs.length === 0 ? (
        <section className="history-empty surface">
          <strong>{chinese ? "暂无历史 run" : "No completed runs"}</strong>
          <p>{chinese ? "成功完成一次 LFP-only 推理后，结果会出现在这里。" : "Results appear here after a successful LFP-only inference run."}</p>
        </section>
      ) : (
        <section className="history-list">
          {runs.map((run) => (
            <article className="history-card surface" key={run.run_id}>
              <div className="history-main">
                <span>{new Date(run.created_at).toLocaleString(chinese ? "zh-CN" : "en-US")}</span>
                <h2>{run.source_name}</h2>
                <code title={run.source_path}>{run.source_path}</code>
              </div>
              <div className="history-metrics">
                <div><span>{chinese ? "范围" : "Range"}</span><strong>{run.start_sec.toFixed(2)}–{run.end_sec.toFixed(2)} s</strong></div>
                <div><span>{chinese ? "窗口" : "Windows"}</span><strong>{run.window_count}</strong></div>
                <div><span>Feature</span><strong>{run.embedding_dim} D</strong></div>
                <div><span>{chinese ? "设备" : "Device"}</span><strong>{run.device.toUpperCase()}</strong></div>
              </div>
              <button className="primary-button" disabled={opening !== ""} onClick={() => void openRun(run.run_id)}>
                {opening === run.run_id ? (chinese ? "正在打开…" : "Opening…") : (chinese ? "查看结果" : "View result")}
              </button>
            </article>
          ))}
        </section>
      )}
    </main>
  );
}
