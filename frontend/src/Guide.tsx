import type { Language } from "./i18n";

type GuideSection = {
  title: string;
  paragraphs?: string[];
  bullets?: string[];
  steps?: string[];
  warning?: string;
};

const englishSections: GuideSection[] = [
  {
    title: "What WebLFP does",
    paragraphs: [
      "WebLFP reads local LFP recordings, applies the preprocessing and windowing required by the pretrained model, and generates a 256-dimensional LFP feature for each time window. The unified spike_count/clip checkpoint also contains the narrow/non-narrow presence and count head.",
    ],
    bullets: [
      "Inference requires LFP only; Spike input is not required.",
      "WebLFP does not reconstruct Spike waveforms.",
      "Outputs are not an automatic diagnosis or a direct neuron-type conclusion.",
      "Current evidence mainly comes from the mouse hippocampal recordings and tasks described in the report.",
    ],
  },
  {
    title: "Before you begin",
    paragraphs: ["Confirm the following before importing a recording:"],
    steps: [
      "The recording is on this computer or on accessible storage.",
      "You know the recording format, or at least the acquisition system that produced it.",
      "For arrays or raw binary files, you know the sampling rate. Raw binary also requires dtype, channel count, and data layout.",
    ],
    warning: "If the format is uncertain, start with Auto and add information according to the error message.",
  },
  {
    title: "Open the application",
    paragraphs: [
      "Start WebLFP and open the address shown by the launcher. The header reports model-weight status. If the weight is unavailable, verify that Git LFS downloaded the bundled model file.",
      "Workspace imports recordings, previews signals, runs inference, and exports results. History reopens or permanently deletes locally saved runs. Settings configures the compute environment and appearance. Guide shows this complete user guide.",
    ],
  },
  {
    title: "Import an LFP recording",
    paragraphs: [
      "Enter a local file or recording-directory location in Local path, or use Select file. Selecting a file only fills the local path; it does not upload data.",
      "Supported inputs include NumPy, MATLAB, raw binary, common electrophysiology acquisition formats, and NWB. Recording directories may need to be entered manually. Raw binary requires dtype, channel count, sampling rate, and layout because WebLFP cannot infer them safely.",
    ],
  },
  {
    title: "Inspect metadata",
    paragraphs: ["Select Inspect recording. WebLFP reads metadata first rather than loading the complete recording. Confirm:"],
    bullets: [
      "The sampling rate is correct.",
      "The channel count matches the experiment.",
      "The duration is reasonable.",
      "The dtype and reader format match the recording.",
    ],
    warning: "If any value is clearly wrong, correct the format or input parameters before continuing.",
  },
  {
    title: "Choose time and channels",
    paragraphs: [
      "After inspection, the full recording is selected by default with no fixed processing-duration limit. You can shorten the start and end time in seconds, search and select channel IDs from the recording metadata, and choose the inference device: Auto, CPU, or CUDA.",
      "More channels increase memory and GPU-memory use. Begin with a small representative set.",
    ],
  },
  {
    title: "Preview preprocessing",
    paragraphs: ["Select Preview processed signal and inspect both raw and processed traces. The selected raw segment is retained in memory so inference can reuse it without reading the recording again. Check:"],
    bullets: [
      "The waveform resembles the expected LFP signal.",
      "There are no all-zero, constant, or obviously bad channels.",
      "The selected time range covers the experimental stage of interest.",
      "The processed waveform is not abnormally amplified.",
    ],
    warning: "Do not run inference when the preview is unreasonable. Correct channels, time range, or reader parameters first.",
  },
  {
    title: "Run LFP-only inference",
    paragraphs: ["After checking the preview, select Run LFP-only inference. If the source, stream, time range, and channels are unchanged, WebLFP reuses the segment already held in memory. It will:"],
    steps: [
      "Read the selected time range and channels.",
      "Resample to the model rate when required.",
      "Split the signal into fixed-length windows.",
      "Apply robust z-score independently to each channel.",
      "Generate a 256-dimensional feature with the LFP branch.",
    ],
    warning: "This process does not read Spike data or perform Spike sorting.",
  },
  {
    title: "Understand the result",
    paragraphs: [
      "The result shows the valid window count, feature dimension, actual device, feature-norm check, interactive three-dimensional UMAP feature space, and adjacent-window cosine similarity.",
      "These plots describe how representations change over time. They are not physiological labels or diagnostic conclusions by themselves.",
    ],
  },
  {
    title: "Decode narrow / non-narrow activity",
    paragraphs: [
      "After feature generation, select Run presence and count. No separate classifier file is required: WebLFP uses the head in the same SHA-256-verified checkpoint and reads the current run's saved 256D features without reopening the recording.",
      "For each 0.2-second window, it reports narrow and non-narrow presence probabilities, binary presence at a 0.5 threshold, and continuous predicted counts. Integer evaluation rounds counts and clips them to zero or above.",
    ],
    warning: "This is a window-level activity proxy, not true single-neuron cell typing. It cannot replace histology, transcriptomics, patch clamp, or other independent validation. Reference metrics come from KA chronic epilepsy mouse hippocampal data and may not generalize.",
  },
  {
    title: "Export results",
    paragraphs: [
      "Download embeddings.npz for the per-window 256D features and run.json for reader parameters, channels, time range, model information, and preprocessing configuration.",
      "Keep run.json with formal analyses so the settings can be reproduced and reviewed.",
    ],
  },
  {
    title: "Settings and compute environment",
    paragraphs: [
      "Settings reports PyTorch, CPU/CUDA/ROCm backend, GPU and VRAM, CUDA and cuDNN, system memory and memory speed, plus a short CUDA BF16 benchmark.",
      "Project initialization does not install PyTorch. Choose and install a build that matches the machine: CUDA 13.0 or newer for a compatible NVIDIA driver, ROCm on supported Linux AMD systems, or CPU when no compatible GPU is available. Confirm the current environment after installation before returning to the workspace.",
      "A poor-performance warning means short previews may still work, but large recordings or long intervals are unsuitable. Reduce duration or channels, or use a stronger workstation. A PyTorch replacement stops the local service and restarts it after installation; the progress window shows logs.",
    ],
  },
];

const chineseSections: GuideSection[] = [
  {
    title: "WebLFP 做什么",
    paragraphs: ["WebLFP 读取本地 LFP 记录，按预训练模型要求完成预处理和切窗，为每个时间窗生成 256 维 LFP feature。统一的 spike_count/clip 权重还包含窄波/非窄波 presence 与 count 分类头。"],
    bullets: ["推理时只需要 LFP，不需要 Spike。", "WebLFP 不重建 Spike 波形。", "输出不是自动诊断结论，也不能直接外推为神经元类型结论。", "当前证据主要来自报告所述小鼠海马记录和相关任务。"],
  },
  {
    title: "使用前准备",
    paragraphs: ["请先确认三件事："],
    steps: ["记录文件在当前电脑或可访问的存储设备上。", "你知道记录格式，或至少知道它来自哪种采集系统。", "对于数组或原始二进制文件，你知道采样率；原始二进制还需要数据类型、通道数和数据排列方式。"],
    warning: "如果不确定格式，可以先选择“自动”，再根据错误提示补充信息。",
  },
  {
    title: "打开应用",
    paragraphs: ["启动 WebLFP 后，用浏览器打开启动窗口提示的地址。页面顶部会显示模型权重状态；如果权重不可用，请确认 Git LFS 已下载内置模型文件。", "工作区用于导入记录、预览、推理和导出；历史页可重新查看或永久删除本机 runs；设置用于检查计算环境、安装 PyTorch 和调整外观；说明页展示这份完整指南。"],
  },
  {
    title: "导入 LFP 记录",
    paragraphs: ["在工作区右侧的“本地路径”中输入文件或记录目录位置，也可以点击“选择文件”。选择文件只会填写本机路径，不会上传数据。", "支持 NumPy、MATLAB、原始二进制、常见电生理采集格式和 NWB。目录型记录可能需要手动输入目录位置。原始二进制必须填写数据类型、通道数、采样率和布局，因为软件不能可靠猜测。"],
  },
  {
    title: "检查元数据",
    paragraphs: ["点击“检查记录”后，WebLFP 先读取元数据，而不会立即载入完整记录。请确认："],
    bullets: ["采样率是否正确。", "通道数量是否符合预期。", "记录时长是否合理。", "数据类型和读取格式是否符合记录。"],
    warning: "如果信息明显不对，请先修正格式或参数，再继续。",
  },
  {
    title: "选择时间和通道",
    paragraphs: ["检查记录后默认选择完整记录，不设置固定的单次处理时长上限。可以手动缩短开始和结束时间（秒），在搜索框中从元数据返回的通道 ID 里多选通道，并选择自动、CPU 或 CUDA 推理设备。", "长记录和较多通道会增加内存、显存占用与处理时间；需要快速检查细节时可先缩短处理范围。"],
  },
  {
    title: "预览处理结果",
    paragraphs: ["点击“预览处理结果”，并检查原始波形和处理后波形。选定的原始片段会保留在内存中，后续推理无需再次读取记录："],
    bullets: ["波形是否像预期的 LFP。", "是否存在全零、全常数或明显坏道。", "时间范围是否覆盖关心的实验阶段。", "处理后波形是否没有明显异常放大。"],
    warning: "预览不合理时不要直接推理，应先调整通道、时间范围或读取参数。",
  },
  {
    title: "运行 LFP-only 推理",
    paragraphs: ["预览确认后，点击“运行 LFP-only 推理”。如果数据源、数据流、时间范围和通道没有变化，WebLFP 会复用内存中的片段，然后："],
    steps: ["读取选定时间和通道。", "必要时重采样到模型采样率。", "切分固定长度时间窗。", "逐通道执行 robust z-score。", "使用 LFP 分支生成 256 维 feature。"],
    warning: "整个过程不读取 Spike，也不执行 Spike sorting。",
  },
  {
    title: "查看结果",
    paragraphs: ["结果页显示有效窗口数、feature 维度、实际设备、特征范数检查、可交互的 UMAP 三维表征空间和相邻窗口余弦相似度。UMAP 图可用鼠标拖动旋转、滚轮缩放和右键拖动平移。", "这些图用于观察表征随时间变化，不能单独当作生理标签或诊断结论。"],
  },
  {
    title: "窄波 / 非窄波活动解码",
    paragraphs: ["生成 feature 后，点击“运行 presence 与 count”。无需额外分类文件：应用使用同一份经过 SHA-256 校验的权重，并直接读取当前 run 保存的 256 维 feature，不会重新读取记录。", "每个 0.2 秒窗口会输出窄波和非窄波 presence 概率、以 0.5 为阈值的是否出现，以及连续 count 预测。整数评估时会四舍五入并限制为不小于 0。"],
    warning: "该输出是窗口级活动代理，不是真实单神经元细胞类型，不能替代组织学、转录组、膜片钳等独立验证。参考指标来自 KA 慢性癫痫小鼠海马数据，不能直接外推。",
  },
  {
    title: "导出结果",
    paragraphs: ["下载 embeddings.npz 可获得每个窗口的 256 维 feature；下载 run.json 可保存读取参数、通道、时间范围、模型和预处理配置。", "正式分析时应保留 run.json，便于复现实验条件和核对结果。"],
  },
  {
    title: "设置与计算环境",
    paragraphs: ["设置页显示 PyTorch、CPU/CUDA/ROCm 后端、GPU 和显存、CUDA 和 cuDNN、系统内存和内存速率，以及 CUDA BF16 简短测试。", "项目初始化不会安装 PyTorch。请按机器选择并安装：NVIDIA 且驱动兼容时选择 CUDA 13.0 或更高版本；支持的 Linux AMD 系统选择 ROCm；没有兼容 GPU 时选择 CPU。安装后需确认当前环境才能返回工作区。", "性能过差提醒表示小片段可能仍可运行，但不适合较大记录或长时间范围。可缩短时间、减少通道或换用更强工作站。更换 PyTorch 时本机服务会停止并在安装后重启，进度窗口会显示日志。"],
  },
];

const englishFaq = [
  ["Why is there no upload progress after selecting a file?", "WebLFP is local. File selection only tells the local service which recording to read; it does not upload the recording."],
  ["Why must I enter the sampling rate?", "Arrays and raw binary files often do not store it. A wrong rate produces incorrect windows and resampling."],
  ["Why not start with a very long interval?", "Long intervals create many windows and use more memory, GPU memory, and time. Validate a short segment first."],
  ["Can the result replace Spike?", "Only partially in validated downstream tasks. It is not a Spike waveform or a complete reconstruction of neural activity."],
  ["What should I check after an error?", "Identify whether it occurred during reading, preview, inference, model loading, or device setup. Check access, format, sampling rate, channel count, dtype, time range, and try CPU for a short validation segment when CUDA is unavailable."],
];

const chineseFaq = [
  ["为什么选择文件后没有上传进度？", "WebLFP 是本地应用。选择文件只是告诉本机服务读取哪个记录，不会上传到外部服务器。"],
  ["为什么必须填写采样率？", "数组和原始二进制通常不保存采样率。错误采样率会导致切窗和重采样错误。"],
  ["为什么不建议一开始分析很长数据？", "长时间范围会产生大量窗口，占用更多内存、显存和时间。应先用短片段确认读取与预处理。"],
  ["结果能代替 Spike 吗？", "只能在经过验证的下游任务中部分替代。它不是 Spike 波形，也不是完整神经活动重建。"],
  ["出现错误怎么办？", "先判断错误发生在读取、预览、推理、模型还是设备阶段，再检查文件访问、格式、采样率、通道数、dtype 和时间范围；CUDA 不可用时可先用 CPU 验证短片段。"],
];

export default function Guide({ language, onBack }: { language: Language; onBack: () => void }) {
  const chinese = language === "zh";
  const sections = chinese ? chineseSections : englishSections;
  const faq = chinese ? chineseFaq : englishFaq;

  return (
    <main className="guide-shell">
      <section className="guide-hero surface">
        <div>
          <span className="eyebrow">COMPLETE USER GUIDE</span>
          <h1>{chinese ? "面向生物学研究人员的 WebLFP 完整使用说明" : "Complete WebLFP guide for biology researchers"}</h1>
          <p>{chinese ? "本页完整覆盖导入、检查、预览、推理、活动解码、导出和环境配置。数据留在本机，浏览器只用于操作本机服务。" : "This page covers import, inspection, preview, inference, activity decoding, export, and environment setup. Data stays on the local machine; the browser controls only the local service."}</p>
        </div>
        <button className="secondary-button compact" onClick={onBack}>{chinese ? "返回工作区" : "Back to workspace"}</button>
      </section>

      <section className="guide-grid">
        {sections.map((section, index) => (
          <article className="guide-card surface" key={section.title}>
            <span>{String(index + 1).padStart(2, "0")}</span>
            <h2>{section.title}</h2>
            {section.paragraphs?.map((paragraph) => <p key={paragraph}>{paragraph}</p>)}
            {section.bullets && <ul>{section.bullets.map((item) => <li key={item}>{item}</li>)}</ul>}
            {section.steps && <ol>{section.steps.map((item) => <li key={item}>{item}</li>)}</ol>}
            {section.warning && <p className="guide-warning">{section.warning}</p>}
          </article>
        ))}
      </section>

      <section className="guide-faq surface">
        <span className="eyebrow">13 · FAQ</span>
        <h2>{chinese ? "常见问题" : "Frequently asked questions"}</h2>
        {faq.map(([question, answer]) => (
          <div key={question}><h3>{question}</h3><p>{answer}</p></div>
        ))}
      </section>
    </main>
  );
}
