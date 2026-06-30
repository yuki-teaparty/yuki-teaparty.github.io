---
title: "家用现代扩散模型速成 (5)：rCM 与 Causal-rCM"
date: "2026-06-28 02:00"
slug: diffusion-3p3-rcm
order: 3.4
series: "家用现代扩散模型速成"
summary: "本系列收官：rCM 用连续时间一致性打底、score 蒸馏作正则，把前向与反向缝成一个目标；Causal-rCM 再把它推到因果自回归视频，连续时间比离散快 10×。"
draft: true
---
免责声明：鸽子只是一个平凡的数学爱好者，如果 blog 里出现了错误还请大佬们指正……

## 收个尾

第三章一路走来，我们其实在反复念叨同一组对照。把前几篇的结论摊开：

| | 前向（一致性 / teacher-forcing） | 反向（分布匹配 / self-forcing） |
|---|---|---|
| 散度味道 | forward，回归老师轨迹 | reverse-KL，mode-seeking |
| 代表 | [sCM / MeanFlow / iMF (3)](/blog/posts/diffusion-3-scm-meanflow.html) | [DMD / DMD2 (3.1)](/blog/posts/diffusion-3p1-dmd2.html) |
| 强项 | 稳、多样性好、不塌缩 | 锐利、质量高、能反超老师 |
| 软肋 | 细节/质量上限受老师轨迹约束 | 易塌缩、吃 GAN 调参 |

[(3.1)](/blog/posts/diffusion-3p1-dmd2.html) 结尾我埋了个伏笔：一个管多样性、一个管质量，为什么不把它们缝在一起？[(3.2)](/blog/posts/diffusion-3p2-self-forcing.html) 里我们又看到，causal CD 已经悄悄把一致性引进了因果自回归视频。这一篇就把这两件事正式做完——**rCM** 把前向和反向缝成一个目标，**Causal-rCM** 再把它搬到流式视频上。

这两篇都来自清华 TSAIL + NVIDIA 那一拨人（Kaiwen Zheng、Jun Zhu 等），是「连续 diffusion」这套审美的集大成者。鸽子私心很喜欢——因为它从头到尾都站在**连续时间**这边，对离散一致性是明牌的嫌弃。

## rCM：前向打底，反向作正则

rCM \[1\] 的全名是 **Score-Regularized Continuous-Time Consistency**，拆开就是它的配方：

-   **主体是连续时间一致性（sCM）**：还是 [(3)](/blog/posts/diffusion-3-scm-meanflow.html) 那套 TrigFlow + tangent + JVP，提供 forward 的自洽，负责**稳和多样性**（不塌缩）；
-   **外加一个 score 蒸馏正则项**：把 [(3.1)](/blog/posts/diffusion-3p1-dmd2.html) DMD/VSD 的「两 score 之差」当成一个 reverse-divergence 的正则，负责把**细节锐度**补上来。

示意地写，loss 大概长这样：

$$
\mathcal{L}_{\text{rCM}} = \underbrace{\mathcal{L}_{\text{sCM}}}_{\text{前向自洽 (JVP)}} \;+\; \lambda \underbrace{\mathcal{L}_{\text{score}}}_{\text{反向 score 蒸馏 }\propto\, s_{\text{fake}} - s_{\text{real}}}
$$

为什么要这么缝？因为两边的软肋恰好是对方的强项：

-   纯 sCM（前向）在大模型 / 高分辨率上，细节容易发虚——它只保证贴着老师的轨迹走，没有额外的压力去逼出锐利的高频；
-   纯 DMD（反向）锐是锐，但 mode-seeking 容易塌、还得靠 GAN 调参续命。

把 score 蒸馏当**正则**轻轻挂在一致性主体上，等于让前向负责「别塌、别丢 mode」，反向负责「细节给我顶上来」。

rCM 的成绩单很硬核：

-   **首次把连续时间一致性蒸馏 scale 到通用应用级**的图像 / 视频扩散——一路捅到 **14B 参数、5 秒视频**（Cosmos-Predict2、Wan2.1 这种量级）；
-   **1–4 步**采样，相对老师**加速 15–50×**；
-   质量整体**追平 SOTA 的 DMD2**，但**多样性更好、缓解 mode collapse**，而且**不需要 GAN、不需要满世界搜超参**。

> 一句话：rCM 拿到了 DMD2 的质量，却没付 DMD2 的代价（GAN 炼丹 + 多样性塌缩）。这正是「前向 + 反向」互补该有的样子。

## Causal-rCM：把统一配方搬上流式视频

rCM 是在（基本）双向、非自回归的设定下缝合前向反向。**Causal-rCM** \[2\] 则把它推到 [(3.2)](/blog/posts/diffusion-3p2-self-forcing.html) 那个因果自回归视频 / 交互世界模型的战场，并且把两条线和那里的两种 forcing 一一对上：

-   **Teacher-Forcing（TF）= 前向一致性（CM）**：带因果 mask 的一致性蒸馏，提供 forward divergence 的监督；
-   **Self-Forcing（SF）= 反向分布匹配（DMD）**：训练时自 rollout，提供 reverse divergence 的监督。

Causal-rCM 的论点是：**teacher-forcing 的 CM 恰好是 self-forcing 的 DMD 最好的互补**——一个 offline、forward、保多样性，一个 on-policy、reverse、提质量，合起来就是因果视频蒸馏的「全套配方」。这也正是 [(3.2)](/blog/posts/diffusion-3p2-self-forcing.html) 里 self-forcing 和 causal CD 隐隐要会师的那个终点。

### 为什么非连续时间不可：10× 的实锤

这里就是鸽子最想给离散 diffusion 上眼药的地方了。要在因果自回归 Transformer 上做**连续时间**一致性，你得算那个 tangent JVP——而注意力是带**因果 / 自定义 mask** 的，标准 attention kernel 的 forward-mode 求导又慢又费显存。Causal-rCM 专门写了一个 **custom-mask FlashAttention-2 的 JVP kernel**，把这条路打通。

打通之后的结论非常解气：**连续时间一致性模型相比离散时间一致性模型（dCM），收敛快约 10×**。

回想[一开始 (3)](/blog/posts/diffusion-3-scm-meanflow.html) 我们吐槽离散 CM 的那串毛病——要选 $N$、要调 step schedule、跳步累计误差、收敛慢——在这里全都变成了实打实的训练成本差距。连续时间不是「数学上更优雅」的玄学，它是 10× 的吞吐。离散一致性，可以退场了（

靠这套算法 + 基础设施的统一配方，Causal-rCM 用合成数据训练，就拿到了**流式视频生成的 SOTA**。

## 全系列总结

第三章这四篇，其实是同一个故事的展开：

1.  **[(3)](/blog/posts/diffusion-3-scm-meanflow.html)** 把一致性从离散搬到连续——sCM / MeanFlow / iMF，共用 JVP，是**前向**那条线；
2.  **[(3.1)](/blog/posts/diffusion-3p1-dmd2.html)** 把 VSD 推到 DMD2——分布匹配，是**反向**那条线；
3.  **[(3.2)](/blog/posts/diffusion-3p2-self-forcing.html)** 把反向线推进到因果自回归视频——self-forcing / causal-forcing，并让前向（causal CD）开始合流；
4.  **(3.3)** rCM 把前向 + 反向缝成一个目标，Causal-rCM 再把它落到流式视频，并用 10× 给连续时间盖章。

三句话收束这一整套「清华味」的连续 diffusion：

-   **连续时间**让一致性摆脱离散化，JVP 是它的引擎，10× 是它的回报；
-   **前向 + 反向散度的统一**让一个模型同时拿到多样性（CM）和质量（DMD），还甩开了 GAN；
-   **因果自回归**让这一切能一帧帧实时吐出来，通向可交互的世界模型。

连续 diffusion 的优雅 + forward/reverse 的统一 + 因果自回归 = **Causal-rCM**。家用扩散模型这个系列，到此先告一段落——感谢一路看到这里的各位，茶喝完了，鸽子先溜了（

## 参考资料

\[1\] [Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency (rCM)](https://arxiv.org/abs/2510.08431)

\[2\] [Causal-rCM: Unified Teacher-Forcing and Self-Forcing for Autoregressive Diffusion Distillation](https://arxiv.org/abs/2606.25473)

\[3\] [Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models (sCM)](https://arxiv.org/abs/2410.11081)

\[4\] [Improved Distribution Matching Distillation for Fast Image Synthesis (DMD2)](https://arxiv.org/abs/2405.14867)
