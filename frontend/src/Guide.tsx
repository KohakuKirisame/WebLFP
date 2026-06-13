export default function Guide({ onBack }: { onBack: () => void }) {
  return (
    <main className="guide-shell">
      <section className="guide-hero surface">
        <div>
          <span className="eyebrow">USER GUIDE</span>
          <h1>给生物学研究人员的 WebLFP 使用说明</h1>
          <p>
            WebLFP 在本机读取 LFP 记录，生成 CLIP 对齐的隐空间表征。浏览器只负责交互，
            数据不会因为选择文件而上传到外部服务。
          </p>
        </div>
        <button className="secondary-button compact" onClick={onBack}>返回工作区</button>
      </section>

      <section className="guide-grid">
        <article className="guide-card surface">
          <span>01</span>
          <h2>这个工具能做什么</h2>
          <p>
            只使用 LFP 记录，按模型要求完成预处理、切窗和推理，为每个时间窗生成 128 维表征。
            该表征可在已验证的下游任务中部分替代 Spike 表征。
          </p>
          <p className="guide-warning">
            它不重建 Spike 波形，也不是自动诊断或神经元类型判定工具。
          </p>
        </article>

        <article className="guide-card surface">
          <span>02</span>
          <h2>导入记录</h2>
          <p>
            在工作区右侧填写记录位置，或点击“选择文件”。选择文件只是把本机文件位置填入表单，
            不会上传数据。目录型记录可以手动填写目录位置。
          </p>
          <p>
            数组文件和原始二进制文件通常需要你填写采样率；原始二进制还需要数据类型、通道数和布局。
          </p>
        </article>

        <article className="guide-card surface">
          <span>03</span>
          <h2>先检查元数据</h2>
          <p>
            点击“检查记录”后，请确认采样率、通道数、记录时长和数据类型是否符合实验记录。
            如果这些信息明显不对，应先修正格式或参数。
          </p>
        </article>

        <article className="guide-card surface">
          <span>04</span>
          <h2>预览再推理</h2>
          <p>
            建议先选择较短时间范围和少量代表性通道。运行推理前必须查看原始波形和处理后波形，
            确认没有明显坏道、全零通道或异常放大。
          </p>
        </article>

        <article className="guide-card surface">
          <span>05</span>
          <h2>运行 LFP-only 推理</h2>
          <p>
            预览确认后，点击“运行 LFP-only 推理”。WebLFP 会完成通道和时间选择、必要的重采样、
            窗口切分、robust z-score 归一化和 LFP 分支表征生成。
          </p>
          <p>推理过程不读取 Spike，也不执行 Spike 排序。</p>
        </article>

        <article className="guide-card surface">
          <span>06</span>
          <h2>理解结果</h2>
          <p>
            结果页展示窗口数量、表征维度、运行设备、L2 范数、PCA 二维轨迹和相邻窗口余弦相似度。
            这些图用于观察表征随时间的变化，不应单独解释为生理标签。
          </p>
        </article>

        <article className="guide-card surface">
          <span>07</span>
          <h2>窄波 / 非窄波活动解码</h2>
          <p>
            隐空间生成后，可以运行原项目的参考下游任务。系统按 0.2 秒窗口分别给出窄波和
            非窄波的 presence 概率，以及该窗口内的预测 count。权重已随应用放置，无需选择额外文件。
          </p>
          <p className="guide-warning">
            结果表示窗口内两类波形活动的模型估计，不能直接等同于单神经元真实细胞类型。
          </p>
        </article>

        <article className="guide-card surface">
          <span>08</span>
          <h2>导出和复现</h2>
          <p>
            导出 embeddings 文件可用于后续分析；导出 run 配置文件可记录读取参数、时间范围、
            通道、模型和处理流程。建议每次正式分析都保存 run 配置文件。
          </p>
        </article>

        <article className="guide-card surface">
          <span>09</span>
          <h2>设置与性能提醒</h2>
          <p>
            设置页会检查 PyTorch、CPU/CUDA/ROCm 后端、GPU、显存、内存、cuDNN 和 BF16 简短性能测试。
            如果出现“机器性能过差”，可以先缩短时间范围、减少通道数，或换用性能更好的工作站。
          </p>
        </article>
      </section>

      <section className="guide-faq surface">
        <span className="eyebrow">COMMON QUESTIONS</span>
        <h2>常见问题</h2>
        <div>
          <h3>为什么必须填写采样率？</h3>
          <p>采样率决定时间窗和重采样是否正确。部分文件不会保存采样率，所以需要人工填写真实值。</p>
        </div>
        <div>
          <h3>能直接分析很长一段记录吗？</h3>
          <p>可以，但不建议一开始就这样做。长记录会产生大量窗口，占用更多内存和显存。</p>
        </div>
        <div>
          <h3>结果能完全代替 Spike 吗？</h3>
          <p>不能。它只能在经过验证的下游任务中部分替代 Spike 表征，不是 Spike 波形重建。</p>
        </div>
        <div>
          <h3>报错时先看什么？</h3>
          <p>先判断错误发生在读取、预览、推理、模型还是设备阶段，再检查格式、采样率、通道和时间范围。</p>
        </div>
      </section>
    </main>
  );
}
