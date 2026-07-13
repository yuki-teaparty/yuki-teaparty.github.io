# 《家用(现代)扩散模型》系列结构速查

> 这是写给以后 session 的工作笔记，**不是 post**，不要加进 series.yaml。
> 最后更新：2026-07-11（写完 modern_diffusion_4 之后）。

## 1. 发布机制

- `content/series.yaml` 是唯一的结构来源：只有列在里面的 `content/posts/*.md` 才会发布，专题顺序、文章顺序都由它定。没列的一律当草稿跳过（包括 `claude_drafts/` 整个目录）。
- front-matter 只需要内容元信息：`title` / `date`（格式 `"YYYY-MM-DD HH:MM"`）/ `slug` / `summary`。**不需要** `draft` / `order` / `series`（历史遗留字段，已废除）。旧系列迁移文还有 `original_url` / `source: 知乎专栏`。
- 发布流程：写完 → 定 date → 把 slug 加进 series.yaml 对应专题列表末尾 → commit（惯例消息：`Publish 家用现代扩散模型速成 (N)：XXX`）。发布与否由用户决定，Claude 不要擅自改 series.yaml。
- 构建：`node tools/build.mjs`（或 build.ps1/build.sh），产物在 `blog/`。不要为了"验证"随手跑 build，会改动生成文件。
- 图片放 `assets/img/posts/<slug或系列名>/`，正文用绝对路径 `![](/assets/img/posts/.../xxx.png)`。

## 2. 文章骨架（modern 系列标准型）

1. `## 前言` —— 与上一篇的接口（"上一章我们……"），本篇在系列叙事里的位置，先把本篇的"一句话主线"剧透掉。系列第 1 篇有个人近况自嘲开场，后续篇不用。
2. `## Recap: …`（可选）—— 用现代符号快速重述旧知识，配一句"读了这个就不用读之前的《家用扩散模型》了"或"详细的介绍见《家用扩散模型》"。
3. 主体若干节，一节一个方法/概念。节内节奏固定：
   - **动机**：上一个方法的痛点（"XXX 能 work，但……"）；
   - **设定/定义**：display 公式给出对象；
   - **关键推导**：显式、可跟，每步交代"为什么能这么做"；
   - **落点**："本质是 XXX"/"这不就是 XXX 吗"式的揭底；
   - **结果数字**：FID / NFE / 参数量，一两句带过。
4. `## 小结与下集预告` —— 把本篇塞回系列主线（常用对照表），结尾"请看下集！"。
5. `## Reference` —— 见 §4。

## 3. 声音 / 风格

- 第一人称"**鸽子**"，自嘲、吐槽；句尾故意不闭合的"（"或空括号"（）"是招牌（每篇几处即可，别滥用）。
- `>` blockquote 三种用途：①严格性 caveat（"严格来讲……"）；②历史八卦 / Trivia（可用 "Trivia：" 开头）；③实现细节旁注。
- 中英混排：动词直接用英文（propose / ablate / penalize / introduce / argue / work），术语不硬译（weighting scheme、mode collapse、训推一致除外——"训推一致"是自家黑话，反复出现）。
- 反复"念经"的口头禅（跨篇 callback 是本系列的粘合剂）：
  - "换元到 λ"（post 2）
  - "ℓ2 回归学的是条件期望"（post 1，后面反复引用）
  - "训推一致"（post 2/3，讲 forcing 时还会再用）
  - "diffusion 在连续语境下才优雅"（全系列中心思想）
  - "把想要的梯度包成 stopgrad loss"（post 3 DMD、post 4 sCM）

## 4. 引用格式

- 正文内联：`XXX[[N]][rN]`。
- `## Reference` 是编号列表：全作者名单（and 连接最后一位）+ 题目（sentence case）+ venue（`In ICLR, 2026.` 或 `arXiv preprint, 2025.`）+ `[arXiv:xxxx.xxxxx][rN]`。
- 文末集中放链接定义 `[rN]: https://arxiv.org/abs/...`。
- 公司/无正式论文的引 GitHub（如 FLUX）。

## 5. 数学口味（重要，用户明确表达过）

- **连续时间 diffusion（清华 TSAIL 系：DPM-Solver、sCM、rCM、Causal-rCM）为正统**；离散化（DDPM 求和、dCM/LCM 的 step schedule）一律当反面教材/"时代眼泪"写。
- v 的定义用一般 schedule 的 $\boldsymbol v=\dot\alpha_t\boldsymbol x_0+\dot\sigma_t\boldsymbol\epsilon$；Salimans–Ho 的 $\alpha\boldsymbol\epsilon-\sigma\boldsymbol x_0$ 只是角度参数化特例（TrigFlow 同理，是 $\alpha=\cos t,\sigma=\sin t$ 的特例）。
- 向量一律 `\boldsymbol`；display 公式多、推导显式；喜欢"把 loss 的梯度算出来看它本质是什么"。
- 喜欢逐项拆 weighting/系数并解释"这一项管什么"（post 3 的 DMD weight、post 4 的 sCM tangent）。

## 6. 各篇索引

| # | slug | 一句话 |
|---|------|--------|
| 1 | modern_diffusion_1 | 2022 diffusion 速成（3 pred × 3 loss、DDIM=DPM-Solver-1）+ Rectified Flow |
| 2 | modern_diffusion_2 | weighting scheme：π(t) 即 weighting、logit-normal=λ 上的高斯、timestep shift=平移 λ、Wan2.2 "MoE" |
| 3 | modern_diffusion_3 | reverse divergence：VSD→DMD/DMD2，DMD 在 Causal-rCM 里的 sg-loss 写法 |
| 4 | modern_diffusion_4 | forward divergence：dCM→sCM→CTM→MeanFlow→iMF，object×机器双人舞，JVP 引擎 |
| 5 (计划) | — | rCM（sCM 打底 + score 蒸馏正则）+ Causal-rCM（自回归视频、JVP kernel、连续比离散快 10×）；素材见 claude_drafts/diffusion-3p3-rcm.md |

系列叙事主线：连续时间才优雅 → 蒸馏两条路（forward/reverse）→ 两条路在 rCM 会师 → Causal-rCM 收官。

## 7. 素材关系（claude_drafts/）

- `diffusion-3-scm-meanflow.md`、`diffusion-3p1-dmd2.md`（已删）、`diffusion-3p2-self-forcing.md`、`diffusion-3p3-rcm.md`：**旧版规划**的草稿（当时编号 2/3.1/3.2/3.3，front-matter 还是旧字段），推导可拆素材，但编号、链接、符号都过时了，别直接照抄。
- `beyond_imf.md`：iMF 之后的范式扫描（SoFlow / TiM / W-Flow / TwinFlow），已核实过靠谱度，可作未来某篇或 post 4 之后的外延引用（post 4 只点名了 SoFlow/TiM）。
- 旧系列《家用扩散模型》（diffusion-1-sde 等 6 篇）是 2023 年知乎迁移文，新系列引用它时用书名号提及即可，不加链接也行（post 3/4 的先例）。
