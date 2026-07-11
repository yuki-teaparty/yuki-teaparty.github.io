---
title: "家用现代扩散模型速成 (3)：Distribution Matching Distillation"
date: "2026-07-10 01:00"
slug: modern_diffusion_3
summary: "D(student||teacher)的蒸馏。下一章讲D(teacher||student)的蒸馏。"
---

## 前言：蒸馏的两条路

回忆一下我们的目标：我们要把一个连续的teacher distill成一个few-step的student。

Causal-rCM[[1]][r1]把distillation分成两类：
- Forward divergence，可以理解为D(teacher||student) 。
- Reverse divergence，可以理解为D(student||teacher) 。

![](/assets/img/posts/modern-diffusion-3/distillation.png)

注意有些散度是对称的（比如Jensen–Shannon散度），但有相当多的散度并不对称，因此两者并不一致，以最著名的KL为例：

$$
\mathrm{KL}(p\|q)=\int p(\boldsymbol{x})\log\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}\,d\boldsymbol{x}
$$

可见他penalize $q\to 0, p>0$。代回teacher和student：

- Forward divergence penalize "teacher有但student没有"，也即要求student盖住teacher的每一个mode（“mode covering”）。它的期望在**teacher** $p$ 上取——要估计它，你手里得先有 $p$ 的样本（真实数据，或teacher沿完整ODE跑出来的配对）。于是这条线天生**offline**：把teacher的轨迹当ground truth，让student去回归、去对齐（KD/ODE-pair回归，或consistency沿PF-ODE强制自洽）。student被钉在teacher trajectory上，mode一个不落，但代价是天花板就是teacher。
- Reverse divergence penalize "student有但teacher没有"，也即要求student在和teacher mode overlap的部分尽可能像，但漏掉teacher mode没有代价（“mode seeking”）。它的期望在**student** $q$ 上取——你只需要student当下吐出来的图，再问teacher一句"这张图的score是多少"，**全程不碰teacher的任何一条轨迹**，只把student的边际分布往teacher上怼，因此天生是**on-policy**的。因为不钉死teacher trajectory，给student留下了超越teacher的余地（当然如果只有reverse divergence一个loss，也最多只能追平teacher），但代价是著名的mode collapse。

> GAN就是一种典型的Reverse divergence（当然，GAN并不一定是KL），只不过在GAN的recipe里teacher和student全程都在互相提高（train的不好会变成互相折磨）

本文我们先介绍Reverse divergence的部分。Forward divergence的部分会在下一期介绍。

> 正好，我们今天只会用到KL散度（）

## Recap: Variational Score Distillation (VSD)

> 我们之前在《家用扩散模型》中介绍过一次SDS/VSD（VSD出自ProlificDreamer[[2]][r2]，SDS出自DreamFusion[[3]][r3]），不过为了方便读者，我们概括一下之前的内容。

目标：训一个**一步**生成器 $G_\theta$（$\boldsymbol{x}=G_\theta(\boldsymbol{z})$），让它push出的分布 $q_\theta$ 贴住teacher的数据分布 $p$，即最小化 $\mathrm{KL}(q_\theta\|p)$。

难点在于我们能轻松采样 $\boldsymbol{x}=G_\theta(\boldsymbol{z})$，却无法解析的写出它的密度 $q_\theta(\boldsymbol{x})$ ——偏偏KL的定义里白纸黑字写着 $\log q_\theta$ 。

VSD的关键一步，是发现我们真正要的是KL的梯度、不是KL的值，而这个梯度用不着 $q_\theta(\boldsymbol{x})$ 。把 $\boldsymbol{x}=G_\theta(\boldsymbol{z})$ 代进 $\mathrm{KL}=\mathbb{E}_{\boldsymbol{z}}[\log q_\theta(\boldsymbol{x})-\log p(\boldsymbol{x})]$，对 $\theta$ 求导（注意 $\log q_\theta$ 既通过下标 $\theta$、又通过 $\boldsymbol{x}=G_\theta(\boldsymbol z)$ 依赖 $\theta$）：

$$
\nabla_\theta\mathrm{KL}(q_\theta\|p)=\mathbb{E}_{\boldsymbol{z}}\Big[\underbrace{\nabla_\theta\log q_\theta(\boldsymbol{x})}_{\text{对下标 }\theta\text{ 求导}}+\big(\nabla_{\boldsymbol{x}}\log q_\theta(\boldsymbol{x})-\nabla_{\boldsymbol{x}}\log p(\boldsymbol{x})\big)\tfrac{\partial\boldsymbol{x}}{\partial\theta}\Big]
$$

注意到 $\mathbb{E}_{\boldsymbol{z}}[\nabla_\theta\log q_\theta]=\int q_\theta\nabla_\theta\log q_\theta=\nabla_\theta\!\int q_\theta=\nabla_\theta 1=0$ ，碰巧把唯一要 $q_\theta(\boldsymbol{x})$ 的地方蒸发掉了。

> 这个trick叫Log-derivative trick，最早在1992年的REINFORCE中提出。RL里也会见到这个trick，不过这就是别的专栏的事情了。

剩下的 $\nabla_{\boldsymbol{x}}\log q_\theta$ 是对 $\boldsymbol{x}$（而非 $\theta$）求导，是 $q_\theta$ 的(Stein) score、不是密度本身——而score只要能从 $q_\theta$ 采样就能估出来（见下）。于是只剩**两个score之差**：

$$
\nabla_\theta\mathrm{KL}(q_\theta\|p)=\mathbb{E}_{\boldsymbol{z}}\Big[\big(\underbrace{\nabla_{\boldsymbol{x}}\log q_\theta(\boldsymbol{x})}_{\text{fake score}}-\underbrace{\nabla_{\boldsymbol{x}}\log p(\boldsymbol{x})}_{\text{real score}}\big)\tfrac{\partial\boldsymbol{x}}{\partial\theta}\Big]
$$

而"某个分布在含噪点上的score"正是diffusion model干的活。real score就是teacher（记 $\boldsymbol{\epsilon}_\psi$），白嫖；fake score没现成的，VSD就**introduce一个diffuser** $\boldsymbol{\epsilon}_\phi$，在generator自己的样本上学出来。注意这一步把干净的 $\boldsymbol{x}$ 换成了含噪的 $\boldsymbol{x}_t$，score要在所有噪声档 $t$ 上取——干净 $q_\theta$ 的score本身ill-defined，必须加噪才有意义，这也是下式 $\mathbb{E}_t$ 和weighting $\omega(t)$（还是上一章那个weighting scheme）的来历（$c$ 为text prompt之类的condition）：

$$
\nabla_\theta\mathcal{L}_{\text{VSD}}=\mathbb{E}_{t,\boldsymbol{z},\boldsymbol{\epsilon}}\Big[\omega(t)\,\big(\boldsymbol{\epsilon}_\psi(\boldsymbol{x}_t,t,c)-\boldsymbol{\epsilon}_\phi(\boldsymbol{x}_t,t,c)\big)\,\frac{\partial G_\theta(\boldsymbol{z})}{\partial\theta}\Big]
$$

特别的，generator给出的 $q_\theta$ 本来是一整个分布；如果把它塌成一个确定性单点（$q_\theta\to\delta$），fake score就退化成加进去的噪声 $\boldsymbol{\epsilon}$，VSD也就退化成SDS。

VSD的recipe是交替优化： $\phi$ 走DSM loss去逼近 $q_\theta$ 的score，$\theta$ 走VSD loss。

> 事实上，VSD某种意义上可以被当作一种GAN，优化 $\theta$ 的是G step，而优化 $\phi$ 的是D step。详见《家用扩散模型（2.1）》

## Distribution Matching Distillation (DMD)

> 我们常说的DMD其实是两篇论文：DMD[[4]][r4]（CVPR '24）走了段弯路，大家真正使用的DMD是DMD2[[5]][r5]（NeurIPS '24 Oral）。我们在这里统一叫DMD了。

设定上，DMD loss长这样：

$$
\nabla_\theta\mathcal{L}_{\text{DMD}}=\mathbb{E}_{z,t}\Big[w_t \big(s_{\text{fake}}-s_{\text{real}}\big)\,\frac{\partial G_\theta}{\partial\theta}\Big]
$$

然后考虑到这两个score的定义是这样的（其中 $\mu$ 是模型）：

$$
s_{\text{real}}(\boldsymbol{x}_t,t)=-\frac{\boldsymbol{x}_t-\alpha_t\,\mu_{\text{real}}(\boldsymbol{x}_t,t)}{\sigma_t^2},\qquad
s_{\text{fake}}(\boldsymbol{x}_t,t)=-\frac{\boldsymbol{x}_t-\alpha_t\,\mu_{\text{fake}}(\boldsymbol{x}_t,t)}{\sigma_t^2}
$$

这式子怎么这么眼熟——等一下，这个所谓DMD loss不就是VSD loss的x-pred版吗，只差了一个weighting scheme $w_t$ 而已。

DMD的训练过程和VSD（以及GAN）一样，也是对抗式的：G step 按DMD loss更新 generator $G_\theta$，D step 用DSM loss在 generator 当下的样本上更新 fake diffuser $\mu_{\text{fake}}$ 。

但和VSD不同的是，DMD系列做出了如下的重大改进：

(1) VSD手选了一个weight，而DMD用的是一个逐样本、逐噪声档自适应的weight：

$$
w_t=\frac{\sigma_t^2}{\alpha_t}\cdot\frac{CS}{\big\|\mu_{\text{real}}(\boldsymbol{x}_t,t)-\boldsymbol{x}_0\big\|_1}
$$

其中$C,S$ 为通道数与空间尺寸。它干两件事：
- 前一半 $\frac{\sigma_t^2}{\alpha_t}$ 管**噪声档**——score差 $s_{\text{fake}}-s_{\text{real}}=\frac{\alpha_t}{\sigma_t^2}(\mu_{\text{fake}}-\mu_{\text{real}})$ 在低噪声端（$\sigma_t\to0$）会炸，这个因子正好把 $\frac{\alpha_t}{\sigma_t^2}$ 约掉、只留下干净的denoiser差；
- 后一半 $\frac{CS}{\|\mu_{\text{real}}-\boldsymbol{x}_0\|_1}$ 管**逐样本**——按teacher在这张图上本来的修正幅度归一化，让各样本、各prompt的有效step size对齐（写成 $w_t$ 只是简写，它其实还依赖样本 $\boldsymbol{x}_0$）。

(2) VSD是一个一步的generator——在2023年做1-NFE还是太超前了（真的在2023年用过VSD的朋友告诉鸽子说这玩意非常难train），所以DMD把generator从1-NFE放宽到了4-NFE。为了保证**训推一致**，DMD2做了backward simulation（DMD2 Sec 4.5）——训练时先把整条多步采样链跑一遍（毕竟只有4步），用generator自己的中间态（每一步都是自己上一步的denoise）当输入，而不是真实数据的加噪，让它提前见到推理时会碰到的分布。

> 类似的**训推一致**思想在未来讲forcing的时候会反复出现。另外由于DMD在1 step的时候和VSD几乎完全一致，所以DMD确实也可以做1-NFE的，只不过连DMD2作者都承认只靠DMD2在SDXL上1-NFE很难成功（Sec 4.4）。

(3) Two Time-scale Update Rule (TTUR) —— 每个G step update对应5个D step update。

> 这也是GAN时代的古老trick。TTUR 出自 GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (NeurIPS '17)，原意是给 G 和 D 设不同学习率、让 D 跑在更快的时间尺度上；而"D 多更新几步"的离散版更早在 WGAN 里就有（巧的是WGAN也取n_critic=5）。

> Trivia：TTUR今天（26年7月）有23k引用。但这么牛逼的论文鸽子怎么从来没读过呢——然后鸽子发现本文的major contribution竟然是FID（那确实该有23k引用）

(4) 之前提过，光靠distillation，student的天花板就是teacher；但reverse divergence是on-policy的，所以在distill的同时可以让真实数据也参加loss——也即，可以introduce一个真正的GAN discriminator，用GAN loss来帮助 $G_\theta$ 。

> 结果是ImageNet-64上1-NFE FID 1.28、反超teacher。

## DMD在Causal-rCM中作为Loss的真正实现

在Causal-rCM中，DMD loss被写成了

$$
\mathcal{L}_{\text{DMD}}=\mathbb{E}\big\|\,\boldsymbol{x}_0^\theta-\operatorname{sg}[\boldsymbol{x}_0^\theta-\boldsymbol{g}]\,\big\|_2^2
$$

其中

$$
\boldsymbol{g}:=\frac{f_{\text{fake}}(\boldsymbol{x}_t,t)-f_{\text{teacher}}(\boldsymbol{x}_t,t)}{\operatorname{mean}(\operatorname{abs}(\boldsymbol{x}_0^\theta-f_{\text{teacher}}(\boldsymbol{x}_t,t)))}
$$

这里采用了Causal-rCM的符号。式中 $f$ 就是denoiser（x-prediction），也就是前面DMD节的 $\mu$ 。

本loss看起来非常复杂；但事实上，它**等效于DMD loss**，因为它们有一样的梯度。我们之前define的DMD是个"梯度"，得手动塞进 backward；而这里的 $\mathcal{L}_{\text{DMD}}$ 是个正经标量loss，能直接进优化器、跟别的loss相加。Causal-rCM最后要把consistency主干loss和这个DMD正则一起 `.backward()`，所以必须把DMD也写成loss形式。

> 顺带，SDS/VSD/包括DMD2自己其实implementation上也是同款stopgrad套路。

具体来说

- 前向：sg前向是恒等，令 $\boldsymbol{y}=\boldsymbol{x}_0^\theta-\boldsymbol{g}$，于是 $\boldsymbol{x}_0^\theta-\boldsymbol{y}=\boldsymbol{g}$（loss值恰是 $\|\boldsymbol{g}\|^2$，但这个数没意义）；
- 反向：把 $\boldsymbol{y}$ 当常数，$\nabla_\theta\|\boldsymbol{x}_0^\theta-\boldsymbol{y}\|_2^2=2(\boldsymbol{x}_0^\theta-\boldsymbol{y})^\top\frac{d\boldsymbol{x}_0^\theta}{d\theta}=2\,\boldsymbol{g}^\top\frac{d\boldsymbol{x}_0^\theta}{d\theta}$。

此时应该已经看出DMD的形状了。至于底下这个神秘分母—— $f_{\text{teacher}}(\boldsymbol{x}_t,t)$ 是teacher对含噪图的去噪预测，$\boldsymbol{x}_0^\theta-f_{\text{teacher}}$ 就是"teacher觉得student这张图还差多少"；$\operatorname{mean}\operatorname{abs}$ 取它逐元素绝对值的平均，即这张图上teacher修正量的平均幅度 $\frac{1}{CS}\|\mu_{\text{real}}-\boldsymbol{x}_0\|_1$ ，也就是DMD里weight逐样本项的倒数——稍微对照一下就会发现其他系数确实都约掉了。


## Reference

1. Kaiwen Zheng, Guande He, Min Zhao, Jintao Zhang, Huayu Chen, Jianfei Chen, Chen-Hsuan Lin, Ming-Yu Liu, Jun Zhu, and Qianli Ma. Causal-rCM: A unified teacher-forcing and self-forcing open recipe for autoregressive diffusion distillation in streaming video generation and interactive world models. arXiv preprint, 2026. [arXiv:2606.25473][r1]
2. Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. ProlificDreamer: High-fidelity and diverse text-to-3D generation with variational score distillation. In NeurIPS, 2023. [arXiv:2305.16213][r2]
3. Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. DreamFusion: Text-to-3D using 2D diffusion. In ICLR, 2023. [arXiv:2209.14988][r3]
4. Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, 2024. [arXiv:2311.18828][r4]
5. Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Frédo Durand, and William T. Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024. [arXiv:2405.14867][r5]

[r1]: https://arxiv.org/abs/2606.25473
[r2]: https://arxiv.org/abs/2305.16213
[r3]: https://arxiv.org/abs/2209.14988
[r4]: https://arxiv.org/abs/2311.18828
[r5]: https://arxiv.org/abs/2405.14867
