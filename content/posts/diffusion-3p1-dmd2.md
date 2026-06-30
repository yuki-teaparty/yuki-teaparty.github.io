---
title: "家用现代扩散模型速成 (3)：从 VSD 到 DMD2"
date: "2026-06-28 01:00"
slug: diffusion-3p1-dmd2
order: 3.2
summary: "分布匹配蒸馏这条反向散度的路线：DMD 怎么把 VSD 用在一步生成上，DMD2 又怎么把回归项扔掉、靠 TTUR 和 GAN 反超老师。"
draft: true
---
免责声明：鸽子只是一个平凡的数学爱好者，如果 blog 里出现了错误还请大佬们指正……

## 接着上一篇

[上一篇 (3)](/blog/posts/diffusion-3-scm-meanflow.html) 讲的 sCM / MeanFlow / iMF 是**前向**那条路：逼学生贴住老师的 ODE 轨迹（一致性 / teacher-forcing），是回归式的、稳、不塌缩。

这一篇走**另一条路**：不管轨迹长什么样，我直接让学生**生成出来的分布**去对齐老师的分布。这就是 reverse-KL / 分布匹配那一系，也是后面 self-forcing（[3.2](/blog/posts/diffusion-3p2-self-forcing.html)）的思想源头。代表作就是 DMD 和 DMD2。

好消息是，它的数学我们[在 (2.1) VSD 那篇](/blog/posts/diffusion-2p1-vsd.html)已经推完了。所以这篇可以轻装上阵。

## 三十秒回顾 VSD

[(2.1)](/blog/posts/diffusion-2p1-vsd.html) 里我们想优化一个生成器 $\mathbf{x} = g(\mathbf{z}; \theta)$，让它的分布 $q_\theta$ 贴近预训练扩散模型代表的真实分布 $p$。直接做 DSM 要在巨大的老师上求 Jacobian，劝退。VSD 的办法是再养一个「fake」扩散模型 $\boldsymbol{\epsilon}_\phi$ 去估计生成分布 $q_\theta$ 的 score，于是 reverse-KL 的梯度变成两个 score 之差：

$$
\nabla_\theta \mathrm{KL}(q_\theta \,\|\, p) = \mathbb{E}\left[(\nabla_{\mathbf{x}}\log q_\theta(\mathbf{x}) - \nabla_{\mathbf{x}}\log p(\mathbf{x}))\,\frac{\partial \mathbf{x}}{\partial \theta}\right]
$$

翻译成 $\boldsymbol{\epsilon}$ 的语言（差个负号和 $\sigma_t$）就是

$$
\nabla_\theta \mathcal{L}_{\text{VSD}} = \mathbb{E}_{t,\mathbf{z},\boldsymbol{\epsilon}}\left[\omega(t)\big(\boldsymbol{\epsilon}_\psi(\mathbf{x}_t, t) - \boldsymbol{\epsilon}_\phi(\mathbf{x}_t, t)\big)\frac{\partial \mathbf{x}}{\partial \theta}\right]
$$

实现上**交替优化**：优化 $\phi$ 时用 DSM 让 fake 模型追上当前生成分布；优化 $\theta$ 时用上面这个梯度把生成分布往真实分布推。我们当时还吐槽过：这套交替优化「看着像 GAN」。记住这句话，等会儿 DMD2 要打脸。

> reverse-KL 是 **mode-seeking** 的——它倾向于把质量集中到老师分布的几个高概率峰上，而不是铺满所有 mode。这是分布匹配这条线一切优点（锐利、质量高）和缺点（多样性差、容易塌缩）的总根源。

## DMD：把 VSD 装到一步生成器上

DMD（Distribution Matching Distillation \[1\]）说白了就是：**把 VSD 的生成器 $g$ 换成「一步去噪」的学生扩散模型**，目标从 3D NeRF 变成「把多步老师蒸成一步图像生成器」。

它的分布匹配梯度和 VSD 一模一样，只是把两个 score 的来源讲清楚了：

-   **real score** $s_{\text{real}}$：冻住的老师，直接用；
-   **fake score** $s_{\text{fake}}$：一个在线训练的扩散模型，专门拟合**当前学生生成出来的**分布 $q_\theta$；

$$
\nabla_\theta \mathcal{L}_{\text{DM}} \propto \mathbb{E}_{t}\left[\big(s_{\text{fake}}(\mathbf{x}_t, t) - s_{\text{real}}(\mathbf{x}_t, t)\big)\frac{\partial \mathbf{x}}{\partial \theta}\right]
$$

两个 score 一减，就是「学生分布要往哪个方向挪才更像真实分布」的指南针。

但只有这一项，DMD 训起来会塌（mode collapse）、还会有各种 artifact。所以 DMD 加了第二项——**回归损失（regression / distillation loss）**：

-   预先拿老师**完整跑多步采样**，构造一批「噪声 $\mathbf{z}$ → 老师成品 $\mathbf{y}$」的配对数据；
-   让一步学生在同样的 $\mathbf{z}$ 上输出去对齐 $\mathbf{y}$（用 LPIPS 之类的感知距离）。

这一项相当于给分布匹配上了一道「别跑偏」的护栏，把模式锚住。两项一起，DMD 第一次让一步生成的质量逼近多步老师。

### DMD 的两个痛点

1.  **回归数据贵**。那批配对数据要老师老老实实跑完整的多步采样才能拿到，又慢又占盘。
2.  **还是会掉多样性**。reverse-KL 的 mode-seeking 本性 + 回归项把目标钉死在老师的成品上，生成多样性常常不如老师。

## DMD2：把回归项扔了

DMD2（Improved DMD \[2\]）的核心动作非常激进：**把那个又贵又限制多样性的回归损失直接删掉**，只留分布匹配。但前面说了，光留分布匹配会塌——所以 DMD2 用两个新东西把坑填上。

### 1. TTUR：让 fake score 追得上

删掉回归项后，训练能不能稳，全看 fake score $s_{\text{fake}}$ 有没有**实时、准确地**拟合当前学生分布。如果 fake 模型落后了，那个「两 score 之差」的指南针就是错的，训练直接崩。

DMD2 的办法是 **Two Time-scale Update Rule（TTUR，两时间尺度更新）**：让 fake score 网络的更新频率**高于**生成器（比如生成器更新 1 次，fake 模型更新 5 次）。fake 模型跑得快、生成器走得慢，前者就能稳稳咬住后者移动中的分布。

> 是不是很眼熟？TTUR 本来就是 GAN 训练里平衡判别器/生成器的经典 trick。VSD 那篇说「这玩意看着像 GAN」，DMD2 直接把 GAN 的工具箱搬过来了。

### 2. GAN loss：反超老师

蒸馏的天花板按理说是老师。但 DMD2 想**超过**老师（尤其是高频细节）。办法是再加一个**GAN 损失**：拿一个判别器去区分「学生生成」和「真实数据」，把对抗损失加到生成器上。真实数据里的细节是老师也未必有的，GAN 项让学生能从真实数据里偷师，补回锐度。

### 3. 多步生成

DMD 是严格一步。DMD2 允许**几步**（比如 4 步）生成，并且在训练时**模拟这几步的采样过程**（注意要让训练时的中间状态分布和推理时一致，否则又是 train/test gap——这个问题到 [3.2](/blog/posts/diffusion-3p2-self-forcing.html) 会变成主角）。多给几步，质量进一步上去。

三招齐下，DMD2 在多个 benchmark 上成了 few-step 蒸馏的 SOTA，质量甚至能压老师一头。

## 小结：反向这条线的得与失

DMD / DMD2 这一系的画像很清楚：

-   **优点**：reverse-KL 让它锐利、质量高，配上 GAN 还能反超老师；不需要像一致性那样贴着轨迹走。
-   **代价**：
    -   mode-seeking 的本性 → **多样性 / mode collapse** 始终是悬在头上的剑；
    -   DMD2 的质量很吃 **GAN 调参**（判别器结构、loss 权重、TTUR 比例……人均炼丹）；
    -   要在线维护一个 fake score 网络，工程上比纯回归重。

对照[上一篇 (3)](/blog/posts/diffusion-3-scm-meanflow.html) 的前向一致性（稳、不塌缩，但质量上限受老师轨迹约束），你会发现这是一对漂亮的互补：

| | 前向（一致性 / teacher-forcing） | 反向（分布匹配 / self-forcing） |
|---|---|---|
| 散度味道 | forward，回归轨迹 | reverse-KL，mode-seeking |
| 代表 | sCM / MeanFlow / iMF | DMD / DMD2 |
| 强项 | 稳、多样性好、不塌缩 | 锐利、质量高、能反超老师 |
| 软肋 | 质量上限受老师约束 | 易塌缩、吃 GAN 调参 |

既然一个管多样性、一个管质量，能不能**把两条线缝在一起**？这正是 [(3.3) 的 rCM](/blog/posts/diffusion-3p3-rcm.html) 要干的事——用连续时间一致性打底（forward），再加一个 score 蒸馏正则（reverse），不靠 GAN 就拿到 DMD2 的质量、还把多样性捡回来。

不过在那之前，[下一篇 (3.2)](/blog/posts/diffusion-3p2-self-forcing.html) 我们先把分布匹配这条线推进到**自回归视频**：当生成器要一帧一帧往下吐、还要用自己吐出来的帧当条件时，DMD 就长出了 self-forcing 和 causal-forcing 这一串新名字。

## 参考资料

\[1\] [One-step Diffusion with Distribution Matching Distillation (DMD)](https://arxiv.org/abs/2311.18828)

\[2\] [Improved Distribution Matching Distillation for Fast Image Synthesis (DMD2)](https://arxiv.org/abs/2405.14867)

\[3\] [ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation (VSD)](https://arxiv.org/abs/2305.16213)

\[4\] [家用扩散模型 (2.1)：Variational Score Distillation](/blog/posts/diffusion-2p1-vsd.html)
