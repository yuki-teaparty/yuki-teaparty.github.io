---
title: "家用现代扩散模型速成 (2)：从 sCM 到 MeanFlow"
date: "2026-06-28 00:30"
slug: diffusion-3-scm-meanflow
order: 3.1
series: "家用现代扩散模型速成"
summary: "一致性模型如何摆脱离散化：从 sCM 的 TrigFlow 与 JVP，到 MeanFlow 的平均速度，再到把目标解耦的 iMF。"
draft: true
---



## Consistency Model的limitaion

[上一章](/blog/posts/diffusion-2p2-lcm.html)我们停在了CM/LCM——23 年SOTA的 few-step 模型。但如果你还记得，CM有一个不太优雅的地方：它是**离散**的。我们要预先选一串时刻 $t_1 < t_2 < \dots < t_N$，然后让相邻两步（或者跳 $k$ 步）做 loss。
这个 $k$ 是个很不优雅的超参：选小了，相邻两步太近，CD loss 收敛奇慢；选大了，又有累计的离散化误差。
LCM 论文里的$k=20$ 是ablate出来的（见\[6\]的sec 5.2）。

说白了，CM是在用一串折线去逼近一条本该光滑的 ODE 轨迹。而我们家用扩散模型系列从第一章开始的中心思想就是：**diffusion 在连续语境下才优雅**。既然如此，一致性模型当然也应该有个连续时间的版本——把 $\Delta t \to 0$ 取极限，离散化误差直接消失。

这一章（2 / 3 / 4 / 5）整体在讲**蒸馏与一步生成**，而且会反复出现两条对照的主线：

-   **前向 vs 反向散度**：一致性 / teacher-forcing（本篇）逼着学生贴住老师的 ODE 轨迹（forward 味道）；而下一篇的分布匹配蒸馏（DMD）是 reverse-KL / self-forcing（mode-seeking）。
-   **连续 vs 离散时间**：连续时间的 sCM / MeanFlow / rCM 优雅、收敛快；离散的 LCM / dCM 是反面教材。

本篇先把**连续时间一致性**这条线讲清楚：sCM、MeanFlow（MF）和它的改进版 iMF。剧透一下，它们共享同一台机器——**JVP（Jacobian-vector product，雅可比–向量积）**。

## 从离散 CD 到连续自洽

先快速回忆[上一章](/blog/posts/diffusion-2p2-lcm.html)的一致性函数 $f_\theta(\mathbf{x}_t, t)$：它的活儿是把轨迹上**任意**时刻的 $\mathbf{x}_t$ 一步打回原点 $\mathbf{x}_\epsilon$。理想情况下，$f$ 沿着同一条 PF-ODE 轨迹应该是**常数**，这就是 "consistency" 这个名字的由来。

离散 CD 的 loss 是相邻两步对齐：

$$
\mathcal{L}_{\text{CD}} = \mathbb{E}\left[\lambda(t_n)\, d\!\left(f_\theta(\mathbf{x}_{t_{n+1}}, t_{n+1}),\ f_{\theta^-}(\hat{\mathbf{x}}_{t_n}^\phi, t_n)\right)\right]
$$

其中 $\hat{\mathbf{x}}_{t_n}^\phi$ 是用老师的 score 做一步 ODE solver 得到的。现在我们让 $t_{n+1} = t_n + \Delta t$，把 $\Delta t \to 0$。不严谨地说，相邻两项的差除以 $\Delta t$ 就变成了导数，离散的「相邻自洽」退化成一个**微分形式的自洽条件**：

$$
\frac{\mathrm{d}}{\mathrm{d}t} f_\theta(\mathbf{x}_t, t) = 0 \quad \text{（沿 PF-ODE 轨迹）}
$$

而这个全导数，按链式法则展开，正是 $f$ 沿轨迹的**方向导数（tangent，切向）**：

$$
\frac{\mathrm{d} f_\theta}{\mathrm{d}t} = \frac{\partial f_\theta}{\partial t} + \frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} \cdot \nabla_{\mathbf{x}} f_\theta
$$

这里 $\mathrm{d}\mathbf{x}_t/\mathrm{d}t$ 就是 PF-ODE 的速度场（由老师提供）。关键的观察是：**这一项不需要做有限差分，它是一个 JVP**——给定方向 $(\,1,\ \mathrm{d}\mathbf{x}_t/\mathrm{d}t\,)$，用一次 forward-mode 自动微分就能精确算出来，根本不用选什么 $N$ 或者 $k$。

> 这就是连续时间最爽的地方：离散 CD 里令人头疼的 step schedule、跳步累计误差，全都没了。代价是你得算一个 JVP——好在现代框架（`torch.func.jvp` / `jax.jvp`）forward-mode 一把梭。

听起来很美好。但其实早在 Consistency Models 原始论文里就给出了这个连续时间版本（continuous-time CM），结果是——**根本训不稳**。于是大家又退回去用离散版了。直到 sCM 出现。

## sCM：把连续时间一致性训稳、训大

sCM（**S**implifying, **S**tabilizing and **S**caling，OpenAI 的 Lu & Song \[1\]）干的事情就一句话：把上面那个一直训不稳的连续时间一致性，**真正训稳、并 scale 上去**。它做了三件事。

### 1. TrigFlow：用三角函数统一一切

EDM、Flow Matching、CM 各有各的系数约定，符号一团乱。sCM 提出一个极其干净的参数化 **TrigFlow**：把数据 std 归一到 $\sigma_d$，前向过程写成

$$
\mathbf{x}_t = \cos(t)\,\mathbf{x}_0 + \sin(t)\,\mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \sigma_d^2 \mathbf{I}),\ t \in [0, \tfrac{\pi}{2}]
$$

你看，$t=0$ 就是纯数据，$t=\pi/2$ 就是纯噪声，整条加噪轨迹是单位圆上的一段弧。对应的 PF-ODE 速度场也变成漂亮的三角函数组合，一致性函数可以写成

$$
f_\theta(\mathbf{x}_t, t) = \cos(t)\,\mathbf{x}_t - \sin(t)\,\sigma_d\, F_\theta\!\left(\tfrac{\mathbf{x}_t}{\sigma_d}, t\right)
$$

（$t=0$ 时 $f_\theta = \mathbf{x}_0$，边界条件天然满足）。TrigFlow 的意义是：所有系数都不再是手调魔数，而是 $\sin/\cos$，求 $\mathrm{d}f/\mathrm{d}t$ 那个 tangent 的时候表达式也干净，方便定位不稳定的来源。

### 2. 找到并驯服不稳定项

sCM 仔细分析了那个 tangent $\mathrm{d}f_\theta/\mathrm{d}t$，发现训练的方差主要来自其中一项（粗略地说，就是 $\partial_t f$ 这一块在某些 $t$ 会炸）。对症下药：

-   **Tangent normalization**：把 tangent 向量归一化，控制它的尺度，不让梯度方差爆掉；
-   **Tangent warmup**：训练初期先把这个不稳定项的权重压低，慢慢放开，类似 warmup 的思路；
-   一些 time-embedding / 位置编码上的细节修正（连续 $t$ 的条件注入要小心）。

### 3. Adaptive weighting

不同 $t$ 的 loss 量级差很多，手调 $\lambda(t)$ 不现实。sCM 借用了 EDM2 那套**自适应权重**（用一个不确定性式的可学习 $\lambda(t)$ 把各时刻 loss 自动归到同一量级），训练就不用再为加权操心了。

三招齐下，连续时间一致性第一次被**稳定地** scale 到了 1.5B 级别，两步采样就能逼近多步 diffusion 的质量。更重要的是它从原理上甩开了离散化：**没有 $N$，没有 schedule，没有跳步误差**——非常对鸽子的胃口。

> 严格说 sCM 里 $f_\theta$ 和 $F_\theta$ 之间还差了 EDM 那套 $c_\text{skip}/c_\text{out}/c_\text{in}$ 的 preconditioning，这里为了直观就糊过去了，细节请看 \[1\] 和 [EDM 那篇](/blog/posts/diffusion-1p7-samplers.html)。

## MeanFlow：从「一致性」到「平均速度」

sCM 的 $f$ 学的是「一步到原点」的映射。MeanFlow（MF \[2\]）换了个同样优雅、但视角不同的对象：**平均速度场**。

回忆 Flow Matching：我们有一个瞬时速度场 $v(\mathbf{z}_t, t)$，采样就是沿着它积分 $\mathrm{d}\mathbf{z}_t/\mathrm{d}t = v$。多步采样慢，是因为要把这条积分曲线一小段一小段地求出来。MeanFlow 的想法是：**直接学这段积分的平均值**。定义从 $r$ 到 $t$ 的平均速度

$$
u(\mathbf{z}_t, r, t) := \frac{1}{t-r}\int_r^t v(\mathbf{z}_\tau, \tau)\,\mathrm{d}\tau
$$

有了它，一步就能跨过整段区间：

$$
\mathbf{z}_r = \mathbf{z}_t - (t-r)\, u(\mathbf{z}_t, r, t)
$$

取 $r=0, t=1$，喂一个噪声进去，**1 次函数求值（1-NFE）**直接出图。问题只剩：$u$ 怎么训？

### MeanFlow Identity

把定义式两边乘 $(t-r)$ 记 $U(r,t) := (t-r)u = \int_r^t v\,\mathrm{d}\tau$，然后对 $t$ 求导。左边用乘积法则、右边是微积分基本定理：

$$
\underbrace{u + (t-r)\frac{\mathrm{d}u}{\mathrm{d}t}}_{\partial_t U} = \underbrace{v(\mathbf{z}_t, t)}_{\partial_t U}
$$

整理得到 **MeanFlow Identity**：

$$
u(\mathbf{z}_t, r, t) = v(\mathbf{z}_t, t) - (t-r)\,\frac{\mathrm{d}u}{\mathrm{d}t}
$$

其中全导数 $\dfrac{\mathrm{d}u}{\mathrm{d}t} = \partial_t u + v\,\partial_{\mathbf{z}} u$（$\mathbf{z}$ 随瞬时速度 $v$ 演化，$r$ 固定）。又是熟悉的配方——这个全导数恰好是 $u_\theta$ 沿方向 $(\partial_z\text{:}\,v,\ \partial_r\text{:}\,0,\ \partial_t\text{:}\,1)$ 的 **JVP**。

于是训练 loss 就是让网络 $u_\theta$ 去回归这个 identity 的右边（右边整体 stop-gradient）：

$$
\mathcal{L}_{\text{MF}} = \mathbb{E}\left\| u_\theta(\mathbf{z}_t, r, t) - \text{sg}\!\Big( v - (t-r)(\partial_t u_\theta + v\,\partial_{\mathbf{z}} u_\theta) \Big) \right\|^2
$$

这里 $v$ 用 Flow Matching 的条件速度（比如直线路径下 $v = \mathbf{z} - \mathbf{x}_0$）。MeanFlow 漂亮在：**完全 from scratch、不需要预训练老师、不需要蒸馏**，一个 identity 加一个 JVP 就把一步生成训出来了。

> 注意 $r=t$ 时 $(t-r)=0$，identity 退化成 $u = v$，也就是平均速度等于瞬时速度，自洽。实际训练会混合采样 $r=t$（学瞬时）和 $r\neq t$（学平均）两种样本。

## iMF：把目标从「自己」手里抢回来

MeanFlow 很美，但实战里有两个膈应人的地方（iMF \[3\] 指出的）：

1.  **训练目标依赖网络自身**。看上面那个 loss：回归目标里含 $\partial_t u_\theta + v\,\partial_{\mathbf{z}} u_\theta$，也就是说**目标是用网络自己（还没训好）的预测算出来的**。这是一种 bootstrap——网络一抖，目标跟着抖，训练自然不稳。
2.  **CFG 处理僵硬**。guidance scale $\omega$ 在训练时被写死，推理想换个 $\omega$ 不方便。

iMF 的核心修法，是把 MeanFlow Identity **反过来用**。原来是「拿 $v$ 当目标、让 $u_\theta$ 去回归（但目标里掺了 $u_\theta$ 自己）」；iMF 改成：

-   仍然让网络预测平均速度 $u_\theta$，但用 identity 反推出**对瞬时速度的预测** $\hat v := u_\theta + (t-r)(\partial_t u_\theta + v\,\partial_{\mathbf{z}} u_\theta)$；
-   然后把回归目标设成**干净的、与网络无关的** $v$（Flow Matching 的真值速度，比如 $\mathbf{z} - \mathbf{x}_0$）：

$$
\mathcal{L}_{\text{iMF}} = \mathbb{E}\left\| \hat v(\mathbf{z}_t, r, t) - \text{sg}(v_{\text{target}}) \right\|^2
$$

这样**回归目标和模型自己的预测解耦了**，bootstrap 没了，整个问题退化成一个标准、稳定的回归。第二步，iMF 把 **CFG scale $\omega$ 当成一个条件变量**喂进网络（像喂 $t$ 一样），推理时就能自由调 guidance。

效果很能打：完全 from scratch、不蒸馏、不用任何预训练模型，**ImageNet 256×256 上 1-NFE 拿到 1.72 FID**，相对原版 MeanFlow 提升约 50%，把一步生成和多步 diffusion 的差距进一步抹平。

> iMF 的标题叫《On the Challenges of Fastforward Generative Models》——"fastforward" 这个词挺传神：一步生成就是把多步积分快进掉。

## 小结：前向这条线

把这一篇串起来看，sCM、MeanFlow、iMF 其实是同一件事的三个切面：

-   **共同的机器**是 JVP——无论是 sCM 的 tangent $\mathrm{d}f/\mathrm{d}t$，还是 MeanFlow/iMF 的 $\mathrm{d}u/\mathrm{d}t$，本质都是「网络沿 ODE 方向的方向导数」，forward-mode 一次算出，连续时间因此不需要离散 schedule。
-   **共同的味道**是 forward / teacher-forcing：它们都在逼学生**贴住老师的 ODE 轨迹（或数据的瞬时速度）**，是一种回归式、前向散度的目标。好处是稳、覆盖全、不塌缩；代价是质量上限受老师轨迹约束。

而[下一篇 (3.1)](/blog/posts/diffusion-3p1-dmd2.html)我们要讲的 VSD→DMD2 走的是**另一条路**：不贴轨迹，直接让学生的**分布**去匹配老师的分布（reverse-KL / self-forcing）。两条路各有得失，到 [(3.3) 的 rCM](/blog/posts/diffusion-3p3-rcm.html)会合二为一。

## 参考资料

\[1\] [Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models (sCM)](https://arxiv.org/abs/2410.11081)

\[2\] [Mean Flows for One-step Generative Modeling (MeanFlow)](https://arxiv.org/abs/2505.13447)

\[3\] [Improved Mean Flows: On the Challenges of Fastforward Generative Models (iMF)](https://arxiv.org/abs/2512.02012)

\[4\] [Consistency Models](https://arxiv.org/abs/2303.01469)

\[5\] [Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364)

\[6\] [Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378)