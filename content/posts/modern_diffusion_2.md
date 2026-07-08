---
title: "家用现代扩散模型速成 (2)：Weighting Scheme"
date: "2026-06-30 22:00"
slug: modern_diffusion_2
summary: "关于diffusion中t的采样，以及更多"
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

所以，选取 $\pi(t)$ 相当于固定 $t\sim U[0,1]$ 时，选取diffusion loss的weighting[[3]][r3]。
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

也就是说，均匀采样 $t$ 隐含地把大量样本压在了中间噪声档（$t\approx0.5$），几乎不管两个端点——直觉上来说， $t\to0$（几乎干净）和 $t\to1$（几乎纯噪声）这两头的回归目标太trivial，网络学不到什么，所以网络需要bias on中间的部分。

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

- **logit-normal派**：SD3/3.5、Wan[[5]][r5]（2.1和2.2都是 $m=0,\tau=1$）、Qwen-Image[[7]][r7]、MovieGen[[8]][r8]，以及JiT[[9]][r9]（取 $m=-0.8,\tau=0.8$）都在训练时按logit-normal采 $t$。
- **uniform派**：FLUX[[6]][r6]就是 $t\sim U[0,1]$，把调节权全交给下一节要讲的timestep shift；跟随FLUX的Lumina-Image 2.0[[11]][r11]也是如此。
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

## 推理时的timestep sampling

推理时，ODE solver通常只执行有限步——每走一步都要调用一次（或多次，如果你用高阶solver）网络 $\boldsymbol{v}_\theta$ 。这个次数有个专门的名字叫NFE（Number of Function Evaluations，函数求值次数）。

做过ml的人想必都听过“训推一致”这个词——但对我们连续的扩散模型来说，即便我们推理时使用了和训练一致的 $s$ 做shift $t'=\frac{st}{1+(s-1)t}$ （这当然是必须要有的），训推也是不一致的——因为训练的t是一个连续分布，推理的t退化成了和NFE预算 $N$ 有关的序列 $1=t_0>t_1>\dots>t_N\approx0$ 。在上一章讲过，在diffusion的真正训练目标（以RF为例，模型学会的速度场是条件期望 $\mathbb{E}[\boldsymbol{\epsilon}-\boldsymbol{x}_0\mid\boldsymbol{x}_t]$ ）下，ODE trajectory一定是弯的，因此NFE越小误差越大，当NFE小到个位数（比如4步）的时候，误差会大到无法接受。

这也就是为什么我们需要**蒸馏**——虽然teacher是连续的，但只要用teacher蒸馏一个训练时也是few-step的student就可以啦。 请看下集！ 

## Wan 2.2的“MoE”

这一整章都在讨论怎么更好的采样t来帮助（同一个）模型学会diffusion。

Wan 2.2表示：那不如干脆搞两个模型吧！这也就是所谓MoE（虽然也叫MoE，但和LLM里的MoE突出一个毫无关系），两个14B 的专家——**high-noise expert** 管去噪早期（$t$ 大、噪声重、SNR低，负责整体 layout），**low-noise expert** 管后期（$t$ 小，负责细节），然后router是一个纯对t的阈值（T2V-A14B是0.875，I2V-A14B是0.9）。

> 当然，两个专家还是share了timestep shift的（t2v的 $s=12$、i2v/ti2v的 $s=5$），只不过是同一条 shift 过的 schedule 切成两段而已。

## Reference

1. Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. [arXiv:2403.03206][r1]
2. Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In ICML, 2023. [arXiv:2301.11093][r2]
3. Diederik P. Kingma and Ruiqi Gao. Understanding diffusion objectives as the ELBO with simple data augmentation. In NeurIPS, 2023. [arXiv:2303.00848][r3]
4. Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022. [arXiv:2206.00364][r4]
5. Wan Team, Alibaba. Wan: Open and advanced large-scale video generative models. arXiv preprint, 2025. [arXiv:2503.20314][r5]
6. Black Forest Labs. FLUX.1. 2024. [github.com/black-forest-labs/flux][r6]
7. Qwen Team, Alibaba. Qwen-Image technical report. arXiv preprint, 2025. [arXiv:2508.02324][r7]
8. Movie Gen Team, Meta. Movie Gen: A cast of media foundation models. arXiv preprint, 2024. [arXiv:2410.13720][r8]
9. Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. In CVPR, 2026. [arXiv:2511.13720][r9]
10. NVIDIA. Cosmos 3: Omnimodal world models for Physical AI. arXiv preprint, 2026. [arXiv:2606.02800][r10]
11. Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Jiakang Yuan, Xinyue Li, Dongyang Liu, Xiangyang Zhu, Manyuan Zhang, Will Beddow, Erwann Millon, Victor Perez, Wenhai Wang, Conghui He, Bo Zhang, Xiaohong Liu, Hongsheng Li, Yu Qiao, Chang Xu, and Peng Gao. Lumina-Image 2.0: A unified and efficient image generative framework. arXiv preprint, 2025. [arXiv:2503.21758][r11]

[r1]: https://arxiv.org/abs/2403.03206
[r2]: https://arxiv.org/abs/2301.11093
[r3]: https://arxiv.org/abs/2303.00848
[r4]: https://arxiv.org/abs/2206.00364
[r5]: https://arxiv.org/abs/2503.20314
[r6]: https://github.com/black-forest-labs/flux
[r7]: https://arxiv.org/abs/2508.02324
[r8]: https://arxiv.org/abs/2410.13720
[r9]: https://arxiv.org/abs/2511.13720
[r10]: https://arxiv.org/abs/2606.02800
[r11]: https://arxiv.org/abs/2503.21758
