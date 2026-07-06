---
title: "家用现代扩散模型速成 (4)：iMF 之后——one-step 生成还能怎么改"
date: "2026-07-01 00:00"
slug: modern_diffusion_4
order: 3.15
series: "家用现代扩散模型速成"
summary: "MeanFlow / iMF 之后，one-step 生成在“怎么参数化、怎么训练这个 map”上的几条新范式。"
draft: true
---

## 定位

上一篇把 consistency distillation 一路写到了 MeanFlow / Improved MeanFlow (iMF)。这一篇专门收 **iMF 之后**、在**范式层面**（而不是 loss 层面）改动 one-step / few-step 生成的代表作。

> 先划清边界：像 Representation Fréchet Loss (RFL)[[8]][r8] 那种，本质是一个 **post-training objective**——在表示空间里优化 Fréchet distance，把一个多步生成器改造成一步——它不改变 map 本身的参数化，属于 loss 侧。这一篇不写它，只在最后拿它做对照。
>
> 这一篇关心的是另一件事：**这个从 $\boldsymbol{\epsilon}$ 到 $\boldsymbol{x}_0$ 的一步映射，到底该建成什么**——速度的时间平均（MeanFlow）？ODE 的解？任意区间的状态转移？还是干脆直接规定它的训练动力学？

回忆一下 MeanFlow[[1]][r1] 的出发点：Flow Matching 学的是瞬时速度 $\boldsymbol{v}$，一步走不了；MeanFlow 改学**平均速度** $\boldsymbol{u}(\boldsymbol{x}_t,s,t):=\frac1{t-s}\int_s^t \boldsymbol{v}\,d\tau$，配一个 average/instantaneous 之间的 identity 做训练目标，于是一次 NFE 就能从 $t$ 跳到 $s$。iMF 在此基础上修了“训练目标依赖网络自身”和 CFG 尺度的问题（见上一篇）。

下面几条线，都是在追问“除了平均速度，还能建什么”。

## 视角一：建模“解”而不是“速度”——SoFlow

**SoFlow: Solution Flow Models for One-Step Generative Modeling**[[2]][r2]（Tianze Luo, Haotian Yuan, Zhuang Liu，**ICLR 2026**）。

- 核心：不去拟合速度场，而是直接分析 velocity ODE 的**解函数 (solution)** 与速度之间的关系，把“解”本身作为网络对象。训练用两个 loss：一个 Flow Matching loss，一个 **solution consistency loss**。
- 最实在的卖点：solution consistency loss **不需要 JVP**（Jacobian-vector product）。MeanFlow/iMF 的 average-velocity identity 要对网络求 JVP，这是训练慢、显存高、不稳的主要来源之一——SoFlow 正是冲这个痛点去的。
- 结果：同一个 DiT 架构、同样 epoch 数下，1-NFE FID **优于 MeanFlow**。
- 定位：离 iMF 叙事最近的“下一篇”，也是最直接的 MeanFlow 平替。

> 靠谱度：Zhuang Liu（ConvNeXt 作者）在 Princeton 的组，已中 ICLR 2026，官方代码 `zlab-princeton/SoFlow`。放心引。

**解函数与 solution consistency loss.** SoFlow 不学速度 $\boldsymbol v$，而是直接学 velocity ODE 的**解函数** $f_\theta(\boldsymbol x_t,t,s)$——把 $t$ 时刻的状态 $\boldsymbol x_t$ 沿 ODE 送到 $s$ 时刻的解，边界条件 $f_\theta(\boldsymbol x_t,t,t)=\boldsymbol x_t$。实现上用一个 boundary-respecting 的 Euler 参数化：

$$
f_\theta(\boldsymbol x_t,t,s)=\boldsymbol x_t+(s-t)\,F_\theta(\boldsymbol x_t,t,s)
$$

一个合法的解函数要满足 PDE 约束 $\partial_1 f_\theta\cdot\boldsymbol v(\boldsymbol x_t,t)+\partial_2 f_\theta=0$。SoFlow 把它变成一个 consistency loss：

$$
\mathcal L_{\text{SCM}}=\mathbb E\Big[\,w\,\big\|\,f_\theta(\boldsymbol x_t,t,s)-f_{\theta^-}\!\big(\boldsymbol x_t+(\alpha_t'\boldsymbol x_0+\beta_t'\boldsymbol x_1)(l-t),\,l,\,s\big)\big\|^2\Big]
$$

其中 $l\in(s,t)$ 取在 $t$ 附近，$\theta^-$ 是 stop-grad，$\alpha_t'\boldsymbol x_0+\beta_t'\boldsymbol x_1$ 是那个 tractable 的 conditional velocity（顶替 $\boldsymbol v(\boldsymbol x_t,t)$）。直觉上就是：不管从 $t$ 出发、还是从沿轨迹稍微前进一点的 $l$ 出发，映到同一个终点 $s$ 的结果都得一致。另外再挂一个普通 FM loss（好在训练时用 CFG），总 loss 是两者的凸组合。

**为什么绕开了 JVP.** PDE 约束里那个 $\partial_1 f_\theta\cdot\boldsymbol v$ 本身就是一个对网络的 JVP——MeanFlow/iMF 正是靠显式算它来上 average-velocity identity 的，而 JVP（double-backward）又慢又吃显存、还和 memory-efficient attention 不兼容。SoFlow 的招是用一阶有限差分/泰勒把它换掉：

$$
\frac{f_\theta(\boldsymbol x_t,t,s)-f_\theta(\boldsymbol x_t+\boldsymbol v\,(l-t),\,l,\,s)}{t-l}=\partial_1 f_\theta\cdot\boldsymbol v+\partial_2 f_\theta+o(1)
$$

于是 JVP 项只出现在**两次普通前向**（在 $(t,s)$ 和 $(l,s)$ 各跑一遍网络）的差里，把这个差压到 0，就等价于满足了 MeanFlow 要用 JVP 才能满足的同一个条件——但全程只需标准 forward+backward，一次 `jvp` 都不用调。省下的很实在：peak 显存降约 31%、训练快约 23%（能重新用上 memory-efficient attention），同一个 DiT、同样 epoch 下 1-NFE FID 全面优于 MeanFlow（B/2 4.85 vs 6.17，XL/2 2.96 vs 3.43）。

顺一句：$f_\theta$ 和 MeanFlow 的平均速度 $\boldsymbol u$ 只差一步换算（Euler 下 $f=\boldsymbol x_t+(s-t)\boldsymbol u$，即 $F_\theta=-\boldsymbol u$），差别全在训练目标——SoFlow 走轨迹自洽（有限差分），MeanFlow 走平均-瞬时速度恒等式（JVP）。

## 视角二：学任意区间的状态转移——Transition Models (TiM)

**Transition Models: Rethinking the Generative Learning Objective**[[3]][r3]（Zidong Wang 等，上海 AI Lab + CUHK MMLab）。

- 核心：推导一个**精确的连续时间状态转移方程**，解析地定义任意有限区间 $t\to s$ 的跳转——既不是只建模无穷小动力学（PF-ODE 的瞬时速度），也不是只建模端点（x-pred）。MeanFlow 的“平均速度”可以看成它的一个特例视角。
- 意义：同一个模型从单步一路到多步**无缝滑动**，而且**随采样步数单调变好**。这一点很关键——以往的 few-step 蒸馏模型普遍有“质量天花板”，加步数不涨甚至掉，TiM 打破了这个。
- 规模：865M 参数在各 step 数下压过 SD3.5（8B）/ FLUX.1（12B）。
- 定位：这一批里范式高度最高的一篇，把“该学什么”从速度/端点抬到了“任意区间 transition”。

> 靠谱度：上海 AI Lab + CUHK（Wanli Ouyang、Lei Bai、Xiangyu Yue），HF 有讨论（~29 赞），官方代码+预训练权重 `WZDTHU/TiM`。2025-09 arXiv，中稿状态待确认，但代码权重齐、实验硬。

**状态转移方程.** 记 $\boldsymbol x_t=\alpha_t\boldsymbol x+\sigma_t\boldsymbol\epsilon$。TiM 的网络 $f_\theta(\boldsymbol x_t,t,r)$ 比 diffusion 多喂一个**目标时刻 $r$**（不是只有当前 $t$）。把 diffusion 的 $\boldsymbol x$-/$\boldsymbol\epsilon$-prediction 代进 $\boldsymbol x_r=\alpha_r\hat{\boldsymbol x}+\sigma_r\hat{\boldsymbol\epsilon}$，得到任意区间 $t\to r$ 的**精确**跳转（Eq. 6）：

$$
\boldsymbol x_r=A_{t,r}\,\boldsymbol x_t+B_{t,r}\,f_\theta(\boldsymbol x_t,t,r),\qquad B_{t,r}=\frac{\sigma_r\alpha_t-\alpha_r\sigma_t}{\hat\sigma_t\alpha_t-\hat\alpha_t\sigma_t}
$$

（$\hat\alpha_t,\hat\sigma_t$ 是 diffusion 目标 $\hat\alpha_t\boldsymbol x+\hat\sigma_t\boldsymbol\epsilon$ 的系数。）关键一步：把 Eq. 6 不当数值近似，而当作**对每个 $t$ 都成立、且给出同一个 $\boldsymbol x_r$** 的恒等式，对 $t$ 求导，得到 **State Transition Identity**（Eq. 8）：

$$
\frac{d}{dt}\Big[B_{t,r}\big(\hat\alpha_t\boldsymbol x+\hat\sigma_t\boldsymbol\epsilon-f_\theta(\boldsymbol x_t,t,r)\big)\Big]=0
$$

拆开就是 $(\text{残差})\cdot\frac{dB}{dt}+B\cdot\frac{d(\text{残差})}{dt}=0$——前项论文叫 "PF-ODE supervision"，后项叫 "time-slope matching"。从这个 identity 解出训练 target $\hat f$（Eq. 9），loss 就是 $\mathbb E\,[w(t,r)\,d(f_\theta-\hat f)]$。里头那个 $df_{\theta^-}/dt$ 又是个时间导数——TiM 和 SoFlow 一样躲开 JVP，用有限差分（它叫 DDE）：$df/dt\approx[f(\boldsymbol x_{t+\varepsilon},t+\varepsilon,r)-f(\boldsymbol x_{t-\varepsilon},t-\varepsilon,r)]/2\varepsilon$，这也是它能 865M 从头训起来的关键。

**和 MeanFlow identity 的关系.** 前面说"MeanFlow 是它的一个特例"——论文（App. B.2）明确证了：取 OT-FM 参数化 $\{\alpha_t=1-t,\sigma_t=t,\hat\alpha_t=-1,\hat\sigma_t=1\}$，就有 $B_{t,r}=r-t$、$dB/dt=-1$，TiM 目标退化成 MeanFlow 的 average-velocity 目标。更妙的是这个家族两头都收得住：$t\to r$ 时 $B\to0$、$\hat f\to\hat\alpha_t\boldsymbol x+\hat\sigma_t\boldsymbol\epsilon$，退回普通 diffusion；$r=0$ 时退回 continuous-time consistency model。所以 diffusion（瞬时速度）、consistency（端点）、MeanFlow（平均速度）在 TiM 里是同一条 identity 的三个切面。

**为什么能单调变好.** TiM 的批评是：MeanFlow"整条轨迹取平均"把 local dynamics 抹掉了，所以加步数不涨（质量天花板）。TiM 的 identity 同时上了两个约束——**Implicit Trajectory Consistency**（直接 $t\to r$ 必须等于任意分解 $(t\to s)\circ(s\to r)$，这是普通 consistency model 没有的）和 **Time-Slope Matching**（不只压残差 $h(t)\to0$，还压 $dh/dt\to0$，多了一阶监督）——于是加采样步是在**细化同一条解流形**而非改路径，质量随 NFE 单调上升。数字上 865M 在各 step 数下压过 SD3.5(8B)/FLUX.1(12B)（GenEval 0.67@1-NFE → 0.83@128-NFE），而 FLUX.1-schnell 这类反而会从 0.68 掉到 0.58。

## 视角三：直接规定训练动力学——W-Flow（OT / gradient-flow 视角）

**One-Step Generative Modeling via Wasserstein Gradient Flows**[[4]][r4]（Jiaqi Han, …, Stefano Ermon, Emmanuel J. Candès，Stanford + ByteDance Seed，2026-05）。

- 核心：不手工设计要拟合的速度场或自洽条件，而是**直接规定 one-step generator 训练期的分布演化**——让生成分布沿某个能量泛函的 **Wasserstein gradient flow** 做最速下降。能量泛函取 **Sinkhorn divergence**，得到一个基于最优传输（OT）的更新，组合了 generated→real 与 generated→generated（自输运）两个 transport plan。
- 与 CM / MeanFlow 的区别：不蒸馏、不在预定义轨迹上强制 self-consistency；训练信号有 OT 理论支撑，而不是继承为多步迭代设计的扩散轨迹。
- 数字：**1-NFE ImageNet-256 FID 1.29 (XL) / 1.35 (L)**，是目前看到最强的一档。
- 定位：想要“最新 SOTA 数字 + 干净的理论动机”就读这篇。

> 靠谱度：Stefano Ermon + Emmanuel Candès（Stanford）+ ByteDance Seed，作者阵容顶级。太新（2026-05），暂未见中稿；社区赞数没抓到（HF/alphaXiv 抓取失败）。

**更新规则.** W-Flow 不手设速度场、也不设自洽条件，而是直接规定 one-step generator 输出分布 $q_t$ 的**训练期演化**：让它沿某个能量泛函 $\mathcal F$ 的 Wasserstein gradient flow 做最速下降（连续形式是连续性方程 $\partial_t q_t=\nabla\!\cdot\big(q_t\,\nabla\tfrac{\delta\mathcal F}{\delta q}(q_t)\big)$）。离散成显式 Euler，就得到作用在样本上的 velocity $V^{(k)}=-\nabla\tfrac{\delta\mathcal F}{\delta q}(q^{(k)})$，再把"当前样本 + 沿 $V$ 挪一步"作为 stop-grad target 去回归（Eq. 14）：

$$
\mathcal L_{\text{W-Flow}}=\frac1N\sum_i\big\|\,\boldsymbol x_i-\operatorname{sg}\!\big(\boldsymbol x_i+\eta\,V^\varepsilon(\boldsymbol x_i)\big)\big\|^2,\qquad \boldsymbol x_i=f_\theta(\boldsymbol z_i)
$$

**Sinkhorn 能量泛函.** $\mathcal F$ 取 debiased **Sinkhorn divergence**（熵正则 OT）：$S_\varepsilon(q,p)=\mathrm{OT}_\varepsilon(q,p)-\tfrac12\mathrm{OT}_\varepsilon(q,q)-\tfrac12\mathrm{OT}_\varepsilon(p,p)$。它诱导的 velocity 恰好是两个 transport plan 之差（Eq. 10）：

$$
V^\varepsilon_{q,p}(\boldsymbol x)=T^\varepsilon_{q,p}(\boldsymbol x)-T^\varepsilon_{q,q}(\boldsymbol x)
$$

$T^\varepsilon$ 是最优熵耦合的 barycentric projection。第一项 $T^\varepsilon_{q,p}$ 是 generated→real 的传输（把生成分布往真实数据 $p$ 拉），第二项 $T^\varepsilon_{q,q}$ 是 generated→generated 的自传输（debias 项，实现时用**另一个独立 batch** 的生成样本估）。

**和 IMM / DMD 这类分布匹配法的关系.** 三者都在"让生成分布逼近真实分布"，但**匹配的量**不同。DMD/DMD2 这支（variational score distillation）拿 teacher 的 **score / 反向 KL** 当信号，且这个信号继承自一条为多步迭代预定义的 **diffusion 轨迹**；IMM 那支本质是 **moment matching**（MMD 味）。W-Flow 不蒸馏、不借 diffusion 轨迹，直接用**全局最优传输（Sinkhorn）的 WGF** 当训练动力学，不需要估 score。妙的是它的 ablation 正好把这三条摆一起比：换 KL 要估 intractable 的 $\nabla\log q_t$、带 bias；换 MMD（≈IMM 的信号）在 $q_t$ 离 $p$ 远时 kernel 饱和、梯度消失；Sinkhorn 明显更稳（FID 7.29 vs KL 10.17 / MMD 10.40）。最终 1-NFE ImageNet-256 拿到 FID **1.29 (XL) / 1.35 (L)**，是这一批里最强的一档。

## 大模型上的落地——TwinFlow

**TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows**[[5]][r5]（Zhenglin Cheng, Peng Sun, Jianguo Li, Tao Lin，Inclusion AI / Westlake / ZJU，**ICLR 2026**）。

- 核心：面向**大模型**的 few-step / one-step 训练，用 **self-adversarial flow**。严格说它带对抗项，谱系上更接近 GAN 蒸馏，而不是纯 flow 范式——但它是少数把 one-step 真正推到大规模并**落地成产品**（`Z-Image-Turbo`）的工作，值得放进来对照。

> 靠谱度：蚂蚁 Inclusion AI + Westlake（Tao Lin），已中 ICLR 2026，且有真实产品落地。可信。

## MeanFlow 本身的“理解与改进”

这一类不算新范式，是把 MeanFlow 训练搞稳/搞快的工程与分析，如果要给上一篇的 iMF 补“后续”可以引：

- **AlphaFlow: Understanding and Improving MeanFlow Models**[[6]][r6]（Snap Research + Qing Qu@UMich）：对 MeanFlow 的系统性理解与改进，官方代码 `snap-research/alphaflow`。
- 另有一批 MeanFlow 训练侧改进（reflow 先拉直轨迹再训 MeanFlow、加速/稳定训练课程等），此处不逐一核实，用到再补。

## 外延（存疑，次要）

- **Discrete MeanFlow: One-Step Generation via Conditional Transition Kernels**[[7]][r7]（University of Kentucky）：把 MeanFlow 推广到**离散数据**，用 conditional transition kernels 定义离散状态转移。⚠️ 正规大学但小组、niche、社区关注低、结果不 competitive——列在这里做完整性，不当代表作。

## 对照：loss 侧的 RFL

**Representation Fréchet Loss for Visual Generation**[[8]][r8]：在表示空间里优化 Fréchet distance（把 FD 估计的 population size 与梯度的 batch size 解耦，即 FD-loss），能把多步生成器**免蒸馏、免对抗、免 per-sample target** 地改造成强 one-step 生成器（Inception 空间下 ImageNet-256 到 0.72 FID）。它和上面这些是正交的——改的是**训练目标**，不是 map 的参数化。两条线可以叠。

## 一句话总结

iMF 之后，one-step 的范式改动大致三条主线：

1. **建“解”而非“速度”**（SoFlow）——顺带干掉 JVP；
2. **建任意区间的 transition**（TiM）——单步到多步单调变好，打破质量天花板；
3. **直接规定训练动力学**（W-Flow）——OT / Wasserstein gradient flow，拿到当前最强 1-NFE FID。

要各取一篇：叙事最连贯读 SoFlow，范式最高读 TiM，SOTA 数字读 W-Flow。

## Reference

1. Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. In NeurIPS, 2025. [arXiv:2505.13447][r1]
2. Tianze Luo, Haotian Yuan, and Zhuang Liu. SoFlow: Solution flow models for one-step generative modeling. In ICLR, 2026. [arXiv:2512.15657][r2]
3. Zidong Wang, Yiyuan Zhang, Xiaoyu Yue, Xiangyu Yue, Yangguang Li, Wanli Ouyang, and Lei Bai. Transition models: Rethinking the generative learning objective. arXiv preprint, 2025. [arXiv:2509.04394][r3]
4. Jiaqi Han, Puheng Li, Qiushan Guo, Renyuan Xu, Stefano Ermon, and Emmanuel J. Candès. One-step generative modeling via Wasserstein gradient flows. arXiv preprint, 2026. [arXiv:2605.11755][r4]
5. Zhenglin Cheng, Peng Sun, Jianguo Li, and Tao Lin. TwinFlow: Realizing one-step generation on large models with self-adversarial flows. In ICLR, 2026. [arXiv:2512.05150][r5]
6. Huijie Zhang, Aliaksandr Siarohin, Willi Menapace, Michael Vasilkovsky, Sergey Tulyakov, Qing Qu, and Ivan Skorokhodov. AlphaFlow: Understanding and improving MeanFlow models. arXiv preprint, 2025. [arXiv:2510.20771][r6]
7. Fairoz Nower Khan, Nabuat Zaman Nahim, Md Sajid Ahmed, Ruiquan Huang, and Peizhong Ju. Discrete MeanFlow: One-step generation via conditional transition kernels. arXiv preprint, 2026. [arXiv:2605.12805][r7]
8. Jiawei Yang, et al. Representation Fréchet Loss for visual generation. arXiv preprint, 2026. [arXiv:2604.28190][r8]

[r1]: https://arxiv.org/abs/2505.13447
[r2]: https://arxiv.org/abs/2512.15657
[r3]: https://arxiv.org/abs/2509.04394
[r4]: https://arxiv.org/abs/2605.11755
[r5]: https://arxiv.org/abs/2512.05150
[r6]: https://arxiv.org/abs/2510.20771
[r7]: https://arxiv.org/abs/2605.12805
[r8]: https://arxiv.org/abs/2604.28190
