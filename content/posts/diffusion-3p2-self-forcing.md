---
title: "家用现代扩散模型速成 (4)：自回归视频蒸馏——Self-Forcing 与 Causal-Forcing"
date: "2026-06-28 01:30"
slug: diffusion-3p2-self-forcing
order: 3.3
summary: "把分布匹配蒸馏推进到一帧一帧往下吐的视频：Self-Forcing 治曝光偏差，Self-Forcing++ 拉到分钟级，Causal-Forcing(++) 修双向→因果的架构错配。"
draft: true
---
免责声明：鸽子只是一个平凡的数学爱好者，如果 blog 里出现了错误还请大佬们指正……

## 从图像到流式视频

[上一篇 (3.1)](/blog/posts/diffusion-3p1-dmd2.html) 我们把分布匹配蒸馏（DMD/DMD2）讲完了，那还是「一张图一步出」的场景。这一篇把场景换成**自回归（autoregressive）流式视频**：模型一帧（或一小段 chunk）接一帧地往下吐，每生成新帧时，要把**前面已经生成的帧**当条件——工程上靠**因果注意力 + KV cache** 实现，这样才能做到实时、无限长、可交互（想想游戏 / 世界模型）。

这一换，立刻冒出一个老朋友：**train/test gap（曝光偏差 exposure bias）**。

## 三种 forcing：问题出在「拿什么当条件」

自回归生成训练有一个绕不开的选择：训练时，每帧的「前文条件」用什么？

-   **Teacher Forcing（TF）**：用**真值前文**（ground-truth 历史帧）当条件训练。简单、并行、好训。但推理时哪有真值？只能拿**自己生成的、带瑕疵的**历史帧当条件。训练分布 ≠ 推理分布，瑕疵会被一帧帧放大——**误差累积**，视频越长越崩。这就是经典曝光偏差。
-   **Diffusion Forcing（DF）**：给每帧独立的噪声水平来训 AR 扩散，灵活了一些，但条件分布和推理时仍对不齐，长程还是会漂。
-   **Self Forcing（SF）**：干脆训练时就**用模型自己生成的前文**当条件——把推理时的自回归 rollout 搬到训练里来。训练分布和推理分布天然一致，曝光偏差从根上消失。

> 一句话：TF 训得爽、推得崩；SF 训得贵、推得稳。下面三篇半的工作，基本都在 SF 这条「自己喂自己」的路上往前怼。

## Self Forcing：训练时就自己 rollout

Self Forcing \[1\]（Adobe + UT Austin）把上面的 SF 思想落地：

-   **训练时自回归 rollout + KV cache**：一帧帧真的生成下去，每帧条件是前面**自生成**的帧，和推理完全一致；
-   **holistic 的视频级损失**：不再是每帧各算各的 frame-wise loss，而是对**整段生成视频**算一个分布匹配损失（用的正是 [3.1](/blog/posts/diffusion-3p1-dmd2.html) 的 DMD 那一套——所以 Self Forcing 本质就是「DMD + 自回归自 rollout」）；
-   **few-step + 随机梯度截断（stochastic gradient truncation）**：整段 rollout 全程反传，显存和算力都顶不住，所以学生用 few-step 扩散，并且只在随机选的若干步上回传梯度，省下大头开销。

结果：在单张 RTX 4090 上做到**实时、流式**生成，质量追平甚至略超非因果的 Wan2.1，而延迟低了约两个数量级。曝光偏差这块硬骨头，被「训练即推理」啃下来了。

## Self Forcing++：把短视频老师榨成分钟级长视频

Self Forcing 治好了曝光偏差，但还有个天花板：**老师只会做短视频**。学生想往老师的训练时长之外**外推**（extrapolate）生成更长的视频时，连续 latent 空间里的误差会**复利式累积**——画面过曝、漂移、糊掉。

Self Forcing++ \[2\] 的招数：

-   **自 rollout 出长视频**，再从中**采样片段（segment）**；
-   对这些片段做**短时老师的 DMD**——老师只在自己擅长的短窗口里给监督，但片段可以取自长视频的任意位置，于是监督信号被「平移」到了远超老师时长的地方；
-   **backward noise initialization** + **rolling KV cache**：让长程 rollout 的噪声初始化和缓存滚动起来，既高效又不必重算重叠帧。

靠这套，视频长度被拉到约 **4 分 15 秒**（接近 base model 位置编码支持的极限，约老师时长的 20×），还能保持时间一致性、不过曝、不累积误差。可以理解成：Self Forcing 解决了「单步往前不崩」，Self Forcing++ 解决了「一直往前也不崩」。

## Causal Forcing：双向老师 → 因果学生的架构错配

前面两篇默认了一件事：学生是因果（causal）的，但**老师往往是双向（bidirectional，full-attention）的**——绝大多数强力视频扩散模型（Wan、Cosmos…）都是双向的，因为双向质量更好。把一个双向老师蒸成因果学生，中间藏着一个被忽视的坑。

Causal Forcing \[3\]（thu-ml，清华这一套）把它点破了。关键是 ODE 蒸馏（确定性蒸馏，学的是一个 **flow map**）需要**帧级单射性（frame-level injectivity）**：每个加噪帧要能唯一映回一个干净帧，flow map 才良定义。

-   双向老师在去噪当前帧时，**偷看了未来帧**当上下文；
-   一旦强制学生因果（只能看过去），同一个加噪的当前帧，对应的干净帧就**不唯一**了（取决于看不到的未来）——**单射性被破坏**；
-   此时确定性蒸馏的最优解退化成**条件期望**（对所有可能未来求平均），结果就是糊、是 average，而**不是**老师真正的 flow map。

Causal Forcing 的修法干净利落：**先用一个自回归（因果）老师来做 ODE 初始化**，把「双向 → 因果」这道架构鸿沟先填平（此时单射性恢复），**然后再走和 Self Forcing 一样的蒸馏流程**。一句话——「Autoregressive Diffusion Distillation Done Right」。代价只是多准备一个 AR 老师做初始化，换来的是相对 Self Forcing 在 Dynamic Degree（+19.3%）、VisionReward（+8.7%）、Instruction Following（+16.7%）等指标上的显著提升。

> 注意这一篇的副标题就在阴阳：之前的做法不是不能跑，而是**没做对**——单射性被破坏却没人管，质量自然上不去。

## Causal Forcing++：把一致性塞进因果自回归

Causal Forcing 把质量做对了，Causal Forcing++ \[4\] 接着把它做**快、做激进**：从 4-step chunk-wise 推到**逐帧（frame-wise）自回归、只用 1–2 步采样**——这是为低延迟、可交互（实时世界模型 / 游戏）准备的最狠设定。

核心是 **Causal Consistency Distillation（causal CD，因果一致性蒸馏）**：

-   它学的是和因果 ODE 蒸馏**同一个** AR 条件 flow map；
-   但监督不再来自昂贵的「跑出整条 ODE 轨迹做初始化」，而是来自**相邻时刻之间单步在线老师 ODE** 提供的信号——一个原则性的廉价替身，省掉了 trajectory 生成的大开销。

效果：2-step 逐帧 AR 比 4-step chunk-wise **延迟降一半，质量还更好**。

但对我们这条故事线，Causal Forcing++ 最重要的一点是名字里那个 **consistency**——注意，前面 Self/Causal Forcing 一直是 reverse-divergence / 分布匹配（[3.1](/blog/posts/diffusion-3p1-dmd2.html) 那条反向路线）的延伸，而 **causal CD 第一次把「一致性」（也就是 [3](/blog/posts/diffusion-3-scm-meanflow.html) 那条前向路线）正式引进了因果自回归**。前向和反向，在因果视频这个战场上开始合流了。

## 小结：自回归把两条线逼到一起

这一篇沿着 self-forcing 把分布匹配蒸馏推进到了流式视频：

-   **Self Forcing**：训练即推理，治曝光偏差；
-   **Self Forcing++**：长视频外推不崩，治误差累积；
-   **Causal Forcing**：修双向→因果的单射性破坏，把质量做对；
-   **Causal Forcing++**：逐帧 1–2 步、因果一致性蒸馏，做到又快又好。

到这里，两条主线已经在因果视频里碰头：self-forcing 是反向（DMD），causal CD 是前向（一致性）。那么有没有一个框架，把**连续时间一致性（前向）**和**分布匹配（反向）**正经地缝成一个目标，而且天生就为因果自回归视频准备？有——[下一篇 (3.3)](/blog/posts/diffusion-3p3-rcm.html) 的 **rCM 与 Causal-rCM**，本系列的收官。

## 参考资料

\[1\] [Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/abs/2506.08009)

\[2\] [Self-Forcing++: Towards Minute-Scale High-Quality Video Generation](https://arxiv.org/abs/2510.02283)

\[3\] [Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation](https://arxiv.org/abs/2602.02214)

\[4\] [Causal Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation for Real-Time Interactive Video Generation](https://arxiv.org/abs/2605.15141)
