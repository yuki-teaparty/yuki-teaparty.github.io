---
title: "家用现代扩散模型速成 (2)：Weighting Scheme"
date: "2026-06-30 22:00"
slug: modern_diffusion_2
order: 3.05
series: "家用现代扩散模型速成"
summary: "关于diffusion中t的采样，以及更多"
draft: false
---

## 什么是Weighting scheme

我们考虑作为一个条件期望的diffusion loss。以RF的v-loss为例：

$$
\mathcal{L}= \mathbb{E}_{t\sim\pi(t)} [\ell(t)] :=\mathbb{E}_{t\sim\pi(t)}\;\mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{\epsilon}}\big[\,\|\boldsymbol{v}_\theta(\boldsymbol{x}_t,t)-\boldsymbol{v}\|^2\,\big] 
$$

其中 $\pi(t)$ 是 $t$ 的分布。显然， $\pi(t)$ 是一个超参数。

古代diffusion比如DDPM里，t是离散且预定义的有限长数组，因此条件期望退化为 $\mathcal{L}=\frac1N\sum_{i=1}^N \ell(t_i)$ ，t的选取突出一个百花齐放（感兴趣的可以倒回去看上一章里EDM[[4]][r4]截图那个表）。

但对连续的现代diffusion（上一章我们展示了diffusion和flow matching本质是一回事，以后我们就都叫diffusion了），条件期望是一个积分：

$$
\mathbb{E}_{t\sim\pi}[\ell(t)]=\int_0^1 \pi(t)\,\ell(t)\,dt
$$

注意到根据条件期望的定义，我们其实有

$$
\int_0^1 \pi(t)\,\ell(t)\,dt=\mathbb{E}_{t\sim U[0,1]}\big[\pi(t)\,\ell(t)\big]
$$

所以，选取 $\pi(t)$ 相当于固定 $\pi(t)\sim U[0,1]$ 时，选取diffusion loss的weighting[[3]][r3]。
这也就是为什么 $\pi(t)$ 在huggingface的diffusers库里叫weighting scheme。

> weighting_scheme的选项包括 `logit_normal`、`mode`、`sigma_sqrt`、`cosmap`、和`none`。`logit_normal`/`mode` 控制的是采样密度 $\pi(t)$，而 `sigma_sqrt`/`cosmap` 控制的是 weight，可见两者确实是完全不分的。

### Uniform（也就是"none"）

最直接的选择当然是 $t\sim U[0,1]$ ——RF原文就是这么干的。

然而，当我们在 $t$ 上均匀采样的时候，我们究竟在做什么？回忆上一章里提过反向过程的ODE：

$$
\frac{\boldsymbol{x}_{t}}{\sigma_t} - \frac{\boldsymbol{x}_{s}}{\sigma_{s}} = \int_{\lambda_{s}}^{\lambda_{t}}e^{\lambda}\hat{\boldsymbol{x}}_{\theta}(\hat{\boldsymbol{x}}_{\lambda},\lambda)d\lambda, \ t\in [0, s]
$$

在这里真正参与积分的并不是 $t$ ，而是 $\lambda$ ！对RF来说

$$
\lambda_t=\log\frac{1-t}{t}=-\operatorname{logit}(t)
$$

把 $t\sim U[0,1]$ 换元到 $\lambda$，密度是

$$
p_\lambda(\lambda)=\Big|\frac{dt}{d\lambda}\Big|=\frac{e^{\lambda}}{(1+e^{\lambda})^2}
$$

这是一个logistic分布，它的中心在 $\lambda=0$（也就是 $t=0.5$），往两端 $\lambda\to\pm\infty$ 迅速衰减。

也就是说，均匀采样 $t$ 隐含地把大量样本压在了中间噪声档（$t\approx0.5$），几乎不管两个端点——直觉上来说， $t\to0$（几乎干净）和 $t\to1$（几乎纯噪声）这两头的回归目标太trivial，网络学不到什么，真正难、真正决定生成质量的是中间那段。

### Logit-normal

SD3[[1]][r1]带头走了另一条路：logit-normal。设定上，logit-normal分布把一个高斯用sigmoid压回 $[0,1]$：

$$
u\sim\mathcal N(m,\tau^2),\qquad t=\sigma(u)=\frac1{1+e^{-u}}
$$

$t$ 上的密度是

$$
\pi_{\text{ln}}(t;m,\tau)=\frac{1}{\tau\sqrt{2\pi}}\cdot\frac{1}{t(1-t)}\exp\!\Big(-\frac{(\operatorname{logit}(t)-m)^2}{2\tau^2}\Big)
$$

这玩意在 $t$ 上肮脏极了——但如果换到 $\lambda$ 上，注意到 $u=\operatorname{logit}(t)=-\lambda$，于是

$$
u\sim\mathcal N(m,\tau^2)\quad\Longleftrightarrow\quad \lambda\sim\mathcal N(-m,\,\tau^2)
$$

所以——logit-normal的本质是要求 $\lambda$ 服从高斯分布。SD3 ablate了许多参数，最后选定 $m=0, \tau=1$ ——也就是标准gaussian。

### 那大家都用什么分布呢？

- **logit-normal派**：SD3/3.5、Wan[[5]][r5]（2.1和2.2都是 $m=0,\tau=1$）、Qwen-Image、MovieGen，以及JiT（取 $m=-0.8,\tau=0.8$）都在训练时按logit-normal采 $t$。
- **uniform派**：FLUX[[6]][r6]就是 $t\sim U[0,1]$，把调节权全交给下一节要讲的timestep shift；跟随FLUX的Lumina-Image 2.0也是如此。
- 甚至还有**mode派**：Cosmos 3[[10]][r10] ablate了一把，然后宣布 "we use logit-normal … for image, audio, and action … and mode sampling for video batches”。

### timestep shift

说来惭愧——在真正上手做diffusion之前，鸽子曾经一度产生过“一定存在一个完美的 $\pi(t)$ 吧，就和大多数人从来不改AdamW的超参数一样”的想法——然而，这其实只是错觉。

正如之前多次反复强调的，选 $t$ 的本质是选SNR，因此 $\pi(t)$ 的选择和任务本身息息相关。

例如，在高分辨率下，相邻像素（或者latent）的冗余度极高[[2]][r2]。在同一个 $t$、同样的 $\sigma_t$ 下，往低分辨率图上加的噪声足以毁掉大半信息，但往高分辨率图上加同一量级的噪声，低频结构几乎纹丝不动——可能downsample回去甚至看不出加过噪。换句话说，分辨率越高，同一个 $t$ 对应的"有效信噪比"越高，网络在高噪声端根本没被训练够。

于是，大家发明了timestep shift：把整条schedule往"更多噪声"的方向推。具体来说，对一个shift factor $s$，timestep shift设定上把 $t$ 重映射成

$$
t'=\frac{s\,t}{1+(s-1)\,t}
$$

把这个curve画出来的话，可以发现它保证 $0\mapsto0$、$1\mapsto1$，中间往 $1$（高噪声）方向抬升。

在 $t$-空间里看，这是个莫名其妙的Möbius变换；但正如我们在本文里念了很多次的经，换元到 $\lambda$ 上，RF下

$$
\frac{\sigma'}{\alpha'}=\frac{t'}{1-t'}=\frac{s\,t}{1-t}=s\cdot\frac{\sigma}{\alpha}\quad\Longrightarrow\quad \lambda'=\lambda-\log s
$$

所以，timestep shift的本质是平移 $\lambda$ ！这就是为什么它叫"shift"。$s>1$ 让 $\lambda$ 整体变小，即整体更noisy。

关于 $s$ 的具体取值，各家不一：

- Wan[[5]][r5]里shift是个和分辨率以及任务都有关的常数。Wan2.1里t2v默认 $s=5$，i2v在480p降到 $s=3$（720p仍是5），而vace、flf2v（注：首尾帧生成视频）$s=16$；Wan2.2里（注：Wan 2.2有high-noise和low-noise两个expert，但两个expert共享shift）t2v用 $s=12$、i2v/ti2v用 $s=5$。可见“分辨率越高 $s$ 越大”只是最粗的规律，i2v/VACE这些带更多条件信息的任务，合适的噪声档位本来就不一样。Cosmos 2/2.5 作为换皮Wan也follow了Wan recipe。
- SD3[[1]][r1]把shift factor跟分辨率挂钩：$s=\sqrt{m/n}$，其中 $n,m$ 分别是参考、目标分辨率的像素数（$H\times W$）。
- FLUX[[6]][r6]让 $\mu=\log s$ 对token数 $L$ 取 $0.5$ ($L=256$)、$1.15$ ($L=4096$) 线性插值，也就是 $s=\exp\!\big(0.5+\tfrac{0.65}{3840}(L-256)\big)$。
- Cosmos 3[[10]][r10]的 $s$ 按分辨率分档：pre-training 256p/480p/720p 取 $s=1/3/5$，mid-training则抬到 $3/5/10$。

## 推理时的Timestep Sampling

训练时我们摆脱了离散求和，但推理时却又不得不把连续ODE离散回有限步——每走一步都要调用一次网络 $\boldsymbol{v}_\theta$（拿它估计当前该往哪个方向去噪），所以一次生成要跑多少步，就等于要前向多少次网络。
> 这个次数有个专门的名字叫 **NFE**（Number of Function Evaluations，函数求值次数）。蒸馏的终极目标就是把NFE压到个位数甚至1。

给定NFE预算 $N$，我们要在 $[0,1]$ 上挑 $N{+}1$ 个节点 $1=t_0>t_1>\dots>t_N\approx0$ 来做累加。

这里的关键是训推一致——推理通常会均匀的在 $[0,1]$ 上采网格，然后用和训练一致的 $s$ 做shift $t'=\frac{st}{1+(s-1)t}$ 。

这里其实藏了个容易混淆的点：**训练的采样密度 $\pi(t)$** 和 **推理的离散网格**根本不是一回事，"训推一致"也并不要求它俩同分布。

先说"训推一致"到底指什么——它指的是 $t\leftrightarrow$ 噪声档（SNR）的**映射**要对齐：训练时你用 shift-$s$ 的 schedule 教网络在某些噪声档去噪，推理时就得喂它同样 shift 过的噪声档。对齐的是 $s$（也就是 $\lambda$ 的平移量），而不是"每个档位放多少个点"。这一步 Wan 是一致的。

至于 $\pi(t)$ 和推理网格为什么是两码事：$\pi(t)$ 是个**重要性采样密度**——每个 SGD step 抽一个 $t$ 估梯度，往中间噪声档堆样本是为了在最难、信息量最大的地方压低梯度方差；而推理网格 $\{t_i\}$ 是对一条**固定 ODE 的数值求积**，节点该密在轨迹弯得最厉害的地方以压离散化误差。"哪里 loss 最有用"和"哪里 ODE 最需要细步"相关但不等同，谁也没规定要照抄对方的密度。

那"logit-normal 上的均匀 sample"是什么？如果指 i.i.d. 抽样，那本身就自相矛盾（抽出来的点服从 logit-normal，怎么会 uniform）。真正想问的其实是**分位点网格（inverse-CDF grid）**：拿 uniform 网格 $u_i=i/N$，喂进 logit-normal 的逆 CDF $t_i=\sigma\!\big(m+\tau\,\Phi^{-1}(u_i)\big)$，得到一串"密在中间、稀在两端"的节点；$m=0,\tau=1$ 时就是把 uniform 网格先过标准正态分位、再过 sigmoid——换到 $\lambda$ 上，这就是一串**高斯分位点网格**。

那"uniform 网格 + shift"和它差多少？回忆本文开头：uniform 的 $t$ 网格换到 $\lambda$ 上是 **logistic** 形状（$s$ 只是把它整体平移 $-\log s$），logit-normal 网格换到 $\lambda$ 上是 **gaussian** 形状——同一个套路，都在中间噪声档堆节点、两端稀疏，只是 logistic 尾巴更肥（方差 $\pi^2/3\approx3.3$，比 $\tau=1$ 的高斯宽），所以 uniform+shift 会比 logit-normal 分位网格稍微多铺一点到两端。

一句话：**该一致的（shift $s$，即 SNR 映射）已经一致了；推理节点密度是另一个自由的 solver 超参**，uniform+shift 只是一个够用、且已经很接近 logit-normal 分位网格的默认选择——真想照训练的 logit-normal 铺网格也行（过一下逆 CDF 即可），只是差别不大，没人为这点收益专门去改。

## 番外：Wan 2.2的MoE

先看 Wan 2.2 的 MoE 是什么：它把一个 dense DiT 拆成两个各约 14B 的专家——**high-noise expert** 管去噪早期（$t$ 大、噪声重、SNR 最低那段，负责整体 layout），**low-noise expert** 管后期（$t$ 小，负责细节）。切换点定在一个 SNR 阈值 $t_{\text{moe}}$（官方说取"$\text{SNR}_{\min}$ 的一半"对应的那一步），$t<t_{\text{moe}}$ 就换到 low-noise 专家。因为任一步只激活一个专家，总参数 27B 但每步只跑 14B，推理开销和显存跟单个 dense 模型基本一样——MoE 在这里买的是**容量**，不是算力。

那和 timestep shift 有关吗？**说没直接关系也对**——Wan 的文档里没把这两件事挂钩，$t_{\text{moe}}$ 纯按 SNR 定，跟那个手设的 shift $s$ 没有公式上的联系。**但说毫无关系也不对**：它俩其实是同一个观察的两种应对。这一整节反复念的经是"噪声轴不均匀，不同 $\lambda$/SNR 档位是性质不同的子任务"——timestep shift 是拿一个标量把整条 schedule 沿 $\lambda$ 平移，让一个共享网络的力气挪到合适的档位；MoE 则更激进，干脆承认"一个网络在整条噪声轴上当通才太难"，直接在 SNR 轴上切一刀、给 high/low 两段各配一套权重。一个是"移"，一个是"切"，出发点都是噪声轴的异质性。

（再多想一层：Wan 2.2 T2V 把 shift 从 2.1 的 5 一路抬到 12，是在往高噪声端多铺采样；MoE 又给高噪声段单独配了个专家——两件事在"高噪声那段才是定结构、值得下本钱的地方"这点上是同一种直觉，尽管官方没把话挑明。）

（小订正：高低两个 denoiser 是 Wan **2.2** 的事——2.1 是单个 dense denoiser，没有高低之分。）那这俩专家的 shift $s$ 一致吗？**一致**——$s$ 是**整条 schedule 的属性**，不是某个专家的。官方一个 A14B 模型只有一个 `sample_shift`（t2v $=12$、i2v/ti2v $=5$）：推理时先拿它把 schedule 整体 shift、`set_timesteps` 一次建完，再在去噪循环里纯按当前 timestep 和 boundary（t2v 切换点 $\approx0.875$、i2v $\approx0.9$）决定这一步喂哪个专家——两个专家吃的是同一条 shift 过的 schedule 的不同区段，自然共享同一个 $s$。

而且这是个**主动的设计选择**：同一份 config 里 `sample_guide_scale`（CFG 尺度）就是**按专家分开的二元组**（比如 t2v 的 low/high 取 $(3.0,4.0)$），可见他们完全能把 shift 也拆成两个、但没这么做。（ComfyUI 把两个专家建成两个模型节点，于是给你两个 shift 输入框，官方模板里都填一样的 $5$；硬给高/低段设不同 shift 也能跑，是社区常见魔改，不是官方默认。）


## Reference

1. Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. [arXiv:2403.03206][r1]
2. Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In ICML, 2023. [arXiv:2301.11093][r2]
3. Diederik P. Kingma and Ruiqi Gao. Understanding diffusion objectives as the ELBO with simple data augmentation. In NeurIPS, 2023. [arXiv:2303.00848][r3]
4. Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022. [arXiv:2206.00364][r4]
5. Wan Team, Alibaba. Wan: Open and advanced large-scale video generative models. arXiv preprint, 2025. [arXiv:2503.20314][r5]
6. Black Forest Labs. FLUX.1. 2024. [github.com/black-forest-labs/flux][r6]
7. Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022. [arXiv:2202.00512][r7]
8. Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In WACV, 2024. [arXiv:2305.08891][r8]
9. Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. In CVPR, 2026. [arXiv:2511.13720][r9]
10. NVIDIA. Cosmos 3: Omnimodal world models for Physical AI. arXiv preprint, 2026. [arXiv:2606.02800][r10]

[r1]: https://arxiv.org/abs/2403.03206
[r2]: https://arxiv.org/abs/2301.11093
[r3]: https://arxiv.org/abs/2303.00848
[r4]: https://arxiv.org/abs/2206.00364
[r5]: https://arxiv.org/abs/2503.20314
[r6]: https://github.com/black-forest-labs/flux
[r7]: https://arxiv.org/abs/2202.00512
[r8]: https://arxiv.org/abs/2305.08891
[r9]: https://arxiv.org/abs/2511.13720
[r10]: https://arxiv.org/abs/2606.02800
