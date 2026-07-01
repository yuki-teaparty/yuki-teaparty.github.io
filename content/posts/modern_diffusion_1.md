---
title: "家用现代扩散模型速成 (1)：从Diffusion到Rectified Flow"
date: "2026-06-28 00:30"
slug: modern_diffusion_1
order: 3
series: "家用现代扩散模型速成"
summary: "Rectified Flow还是值得单开一篇讲的。"
---

## 前言

鸽子做了好几年的某个细分3DV领域最近被大模型当路边一条踹死了——虽然过程有些曲折，但2026年的鸽子真的开始做Diffusion了（）

回看23年还非常naïve的鸽子作为纯数学爱好者写的《家用扩散模型》系列，感觉真是恍若隔世——如果23年的鸽子热情拥抱Diffusion，而不是在3DV上吊死，也许一切都会变得不一样...

最近读了nvidia的rCM[[1]][r1]/Causal rCM[[2]][r2]。它有一个很好的综述——而且由于一作出自TSAIL，因此所用的符号和之前DPM-Solver[[3]][r3]完全一致，感觉又回到了23年读DPM的快乐时光——因此借着这个契机，这次尝试自学一下diffusion比较新的进度，计划一路写到能读懂Causal rCM为止。

由于时代变了很多，因此干脆决定重开一个系列：这次就叫《家用现代扩散模型速成》好了。

## Recap：2022年的Diffusion Model

（读了这个就不用读之前的《家用扩散模型》了）

在2022年，一个diffusion的正向过程如下：

$$\boldsymbol{x}_t=\alpha_t \boldsymbol{x}_0+\sigma_t \boldsymbol{\epsilon}$$

其中 $(\alpha_t, \sigma_t)$ 的选取叫noise schedule或者SNR schedule。起这个名字是因为（功率）信噪比（signal-to-noise ratio, SNR）$\mathrm{SNR}_t=\frac{\alpha_t^2}{\sigma_t^2}$ 完全由它决定。更常用的是它的半对数， $\lambda_t:=\log\frac{\alpha_t}{\sigma_t}$ 。

通常来说，我们会使用一个neural network（参数为$\theta$）来从 $\boldsymbol{x}_{t}$ 预测 $\boldsymbol{x}_0$ 。nn代替的目标有三种：
- x-prediction，直接预测 $\boldsymbol{x}_0\sim \boldsymbol{x}_\theta$
- $\epsilon$-prediction，预测 $\boldsymbol{\epsilon}\sim \boldsymbol{\epsilon}_\theta$。此时 $\boldsymbol{x}_\theta =\frac{1}{\alpha_t}(\boldsymbol{x}_t-\sigma_t\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t))$
- v-prediction，预测 $\boldsymbol{v}:=\dot{\alpha}_t\boldsymbol{x}_0+\dot{\sigma}_t\boldsymbol{\epsilon}\sim \boldsymbol{v}_\theta$，也即正向轨迹 $\boldsymbol{x}_t$ 的时间导数 $\dot{\boldsymbol{x}}_t$。此时 $\boldsymbol{x}_\theta=\frac{\dot{\sigma}_t\boldsymbol{x}_t-\sigma_t\boldsymbol{v}_\theta(\boldsymbol{x}_t, t)}{\alpha_t\dot{\sigma}_t-\sigma_t\dot{\alpha}_t}$ 。

> 为什么能这样反解？把 $\boldsymbol{x}_t$ 和 $\boldsymbol{v}$ 看成关于 $(\boldsymbol{x}_0, \boldsymbol{\epsilon})$ 的二元线性方程组，第一式 $\times\,\dot{\sigma}_t$ 减第二式 $\times\,\sigma_t$ 即可消去 $\boldsymbol{\epsilon}$：
>
> $$\dot{\sigma}_t\boldsymbol{x}_t-\sigma_t\boldsymbol{v}=(\alpha_t\dot{\sigma}_t-\sigma_t\dot{\alpha}_t)\,\boldsymbol{x}_0$$
>
> 移项即得。注意到 $\alpha_t\dot{\sigma}_t-\sigma_t\dot{\alpha}_t=-\alpha_t\sigma_t\dot{\lambda}_t$ ，而 $\lambda_t=\log\frac{\alpha_t}{\sigma_t}$ 严格递减（$\dot{\lambda}_t<0$），因此解是良定义的。

关于什么prediction比较好，至今没有定论，比如说Kaiming的JiT (CVPR '26)[[4]][r4]就argue说x-prediction更好一些。

与此同时，loss也有三种——都是naïve的 $\ell_2$，只是作用在不同的prediction上：
- x-loss：$\mathbb{E}\|\boldsymbol{x}_\theta-\boldsymbol{x}_0\|^2$，也叫Denoising Loss。字面意思，它的目标是Denoising/去噪；JiT标题里的Let denoising model denoise就是这个意思。EDM[[5]][r5]也是著名的x-loss拥趸之一。
- $\epsilon$-loss：$\mathbb{E}\|\boldsymbol{\epsilon}_\theta-\boldsymbol{\epsilon}\|^2$，也叫(Denoising) Score Matching loss，这个名字的来源是因为 $\boldsymbol{\epsilon}_\theta$ 本质是 score function:
$$
\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t)=-\sigma_t\nabla_{\boldsymbol{x}}\log q_t(\boldsymbol{x}_t)
$$

几篇著名的score matching论文[[6]][r6][[7]][r7]用的都是$\epsilon$-loss。DDPM[[8]][r8]本质也是 $\epsilon$-loss。

- v-loss：$\mathbb{E}\|\boldsymbol{v}_\theta-\boldsymbol{v}\|^2$，也叫(Conditional) Flow Matching loss，这是因为前面定义的 $\boldsymbol{v}=\dot{\boldsymbol{x}}_t$ 恰好就是（条件）概率流的速度场。出自Flow Matching for Generative Modeling （ICLR '23）[[9]][r9]。

> 严格来讲，这三个loss是同一个目标在不同weights下的样子。由 $\boldsymbol{x}_t=\alpha_t\boldsymbol{x}_\theta+\sigma_t\boldsymbol{\epsilon}_\theta$ 减去 $\boldsymbol{x}_t=\alpha_t\boldsymbol{x}_0+\sigma_t\boldsymbol{\epsilon}$ 得 $\boldsymbol{\epsilon}_\theta-\boldsymbol{\epsilon}=-\frac{\alpha_t}{\sigma_t}(\boldsymbol{x}_\theta-\boldsymbol{x}_0)$，于是
>
> $$\|\boldsymbol{\epsilon}_\theta-\boldsymbol{\epsilon}\|^2=e^{2\lambda_t}\|\boldsymbol{x}_\theta-\boldsymbol{x}_0\|^2,\qquad \|\boldsymbol{v}_\theta-\boldsymbol{v}\|^2=\alpha_t^2\dot{\lambda}_t^2\|\boldsymbol{x}_\theta-\boldsymbol{x}_0\|^2$$
>
> 可见只有系数上的区别。

> 那为什么三个loss都是 $\ell_2$，而不是其他loss比如 $\ell_1$ ？因为给定 $\boldsymbol{x}_t$，$\boldsymbol{x}_0$（以及对应的 $\boldsymbol{\epsilon},\boldsymbol{v}$）其实是一个随机变量分布；我们真正想学的是它们的**条件期望**——而 $\ell_2$ 回归的总体最优解恰好就是条件期望：$\arg\min_f\mathbb{E}\|f(\boldsymbol{x}_t)-Y\|^2=\mathbb{E}[Y|\boldsymbol{x}_t]$ （太伟大了变分法）。如果换成 $\ell_1$ ，学到的就是“条件中位数”了。

三种loss和三种prediction组合，一共有9种schema。

给定一个SNR schedule，在inference的时候，diffusion的反向过程（作为一个ODE；如果关心为什么反向过程是个ODE，可以去看之前的《家用扩散模型》）是deterministic的。例如，在x-pred下，我们解析的可以得到（此公式来自于DPM-Solver++[[10]][r10]）：

$$
\frac{\boldsymbol{x}_{t}}{\sigma_t} - \frac{\boldsymbol{x}_{s}}{\sigma_{s}} = \int_{\lambda_{s}}^{\lambda_{t}}e^{\lambda}\hat{\boldsymbol{x}}_{\theta}(\hat{\boldsymbol{x}}_{\lambda},\lambda)d\lambda, \ t\in [0, s]
$$

其中 $\hat{(\cdot)}$ 是change of variables——把一个对 $t$ 的量换元成对 $\lambda$ 的量。记 $t_\lambda(\cdot)$ 为 $\lambda(t)$ 的反函数（$\lambda$ 严格单调递减，故反函数存在），则

$$
\hat{\boldsymbol{x}}_\lambda := \boldsymbol{x}_{t_\lambda(\lambda)}, \qquad \hat{\boldsymbol{x}}_\theta(\hat{\boldsymbol{x}}_\lambda, \lambda) := \boldsymbol{x}_\theta\big(\boldsymbol{x}_{t_\lambda(\lambda)},\, t_\lambda(\lambda)\big)
$$

既然这是一个ODE，我们就可以用ODE solver来解。最简单的solver是一阶（Euler）：把被积函数里的 $\hat{\boldsymbol{x}}_\theta$ 在整个区间上当作常数，取左端点（即起点 $\boldsymbol{x}_s$ 处的网络预测）$\boldsymbol{x}_\theta(\boldsymbol{x}_s,s)$ 拎出积分号，于是：

$$
\int_{\lambda_s}^{\lambda_t}e^{\lambda}\,d\lambda=e^{\lambda_t}-e^{\lambda_s}=\frac{\alpha_t}{\sigma_t}-\frac{\alpha_s}{\sigma_s}
$$

代回原式并两边乘以 $\sigma_t$，得到一步更新：

$$
\boldsymbol{x}_t=\frac{\sigma_t}{\sigma_s}\boldsymbol{x}_s+\Big(\alpha_t-\frac{\sigma_t}{\sigma_s}\alpha_s\Big)\boldsymbol{x}_\theta(\boldsymbol{x}_s,s)
$$

再把起点处的 $\boldsymbol{\epsilon}_\theta=\frac{\boldsymbol{x}_s-\alpha_s\boldsymbol{x}_\theta}{\sigma_s}$ 代进去，它又能整理成更眼熟的形式：

$$
\boldsymbol{x}_t=\alpha_t\boldsymbol{x}_\theta(\boldsymbol{x}_s,s)+\sigma_t\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_s,s)
$$

于是...你得到了DPM-Solver-1，也就是DDIM[[11]][r11]！

用类似的方法可以得到各种高阶solver，例如DPM-Solver-2，Heun或者RK4，这里就不再赘述了。

## 横空出世的Rectified Flow (ICLR '23)

在2022年，SNR schedule的范式并没有收敛，各种schedule百花齐放。比如说，这里是从EDM (NeurIPS '22)[[5]][r5] 截的图，可以看到充满了花活：

![](/assets/img/posts/diffusion-1p7-samplers/v2-28dc0af39cb126ff8d6b4f6829bb714d.jpg)

Rectified Flow[[12]][r12]改变了这一切，它propose了一个simple yet effective的新范式：$\alpha_t=1-t$、$\sigma_t=t$ 。于是

$$
\boldsymbol{x}_t=(1-t)\boldsymbol{x}_0+t\boldsymbol{\epsilon}
$$

代进前面 $\boldsymbol{v}$ 的定义 $\boldsymbol{v}=\dot{\alpha}_t\boldsymbol{x}_0+\dot{\sigma}_t\boldsymbol{\epsilon}$，由于 $\dot{\alpha}_t=-1$、$\dot{\sigma}_t=1$，velocity退化为 $\boldsymbol{v}=\boldsymbol{\epsilon}-\boldsymbol{x}_0$ 。

也就是说，条件轨迹就是**一条直线**——当然，由于Rectified Flow的训练使用v-pred, v-loss，因此学到的 $\boldsymbol{v}_\theta$ 并不收敛到 $\boldsymbol{v}$ ，或曰v-loss不收敛到0——一个正经的解释是rectified flow必须是causal的，在一个中间的点 $\boldsymbol{x}_t$ 网络并不知道往哪个方向的直线推是正确的）——因此网络在理想情况下也只能学到average，也即条件期望  $\boldsymbol{v}_\theta(\boldsymbol{x}_t,t)\approx\mathbb{E}[\boldsymbol{\epsilon}-\boldsymbol{x}_0\mid\boldsymbol{x}_t]$ 或曰“边际速度场”，对应的是“边际流”，which不可避免是弯的。

> RF原文给出了一种解法，叫 **reflow**：用当前模型把噪声 $\boldsymbol{\epsilon}$ 沿 ODE 跑出对应的 $\boldsymbol{x}_0$，再拿这批模型自己"连"好的 $(\boldsymbol{x}_0,\boldsymbol{\epsilon})$ 配对重训一个新的 RF。新配对不再交叉，边际流随之被**拉直**（rectified）；反复迭代，轨迹越来越接近直线。这就是rectified这个名字的来源。

接近直线的轨迹有什么好处呢？好处就是一阶方法（一阶方法是linear的，对RF来说就退化成 $\boldsymbol{x}_t=\boldsymbol{x}_s+(t-s)\,\boldsymbol{v}_\theta(\boldsymbol{x}_s,s)$ ）的误差小。
自从Rectified Flow一统江湖之后（从Stable Diffusion 3开始，现代diffusion比如Wan和FLUX都只用RF了），大家的solver彻底换成了Euler，高阶solver则冷门了不少。

> 虽然其实在有RF之前，大多数人也是用DDIM的...

## Reference

1. Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale diffusion distillation via score-regularized continuous-time consistency. In ICLR, 2026. [arXiv:2510.08431][r1]
2. Kaiwen Zheng, Guande He, Min Zhao, Jintao Zhang, Huayu Chen, Jianfei Chen, Chen-Hsuan Lin, Ming-Yu Liu, Jun Zhu, and Qianli Ma. Causal-rCM: A unified teacher-forcing and self-forcing open recipe for autoregressive diffusion distillation in streaming video generation and interactive world models. arXiv preprint, 2026. [arXiv:2606.25473][r2]
3. Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In NeurIPS, 2022. [arXiv:2206.00927][r3]
4. Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. In CVPR, 2026. [arXiv:2511.13720][r4]
5. Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022. [arXiv:2206.00364][r5]
6. Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In NeurIPS, 2019. [arXiv:1907.05600][r6]
7. Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021. [arXiv:2011.13456][r7]
8. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. [arXiv:2006.11239][r8]
9. Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023. [arXiv:2210.02747][r9]
10. Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint, 2022. [arXiv:2211.01095][r10]
11. Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. [arXiv:2010.02502][r11]
12. Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023. [arXiv:2209.03003][r12]

[r1]: https://arxiv.org/abs/2510.08431
[r2]: https://arxiv.org/abs/2606.25473
[r3]: https://arxiv.org/abs/2206.00927
[r4]: https://arxiv.org/abs/2511.13720
[r5]: https://arxiv.org/abs/2206.00364
[r6]: https://arxiv.org/abs/1907.05600
[r7]: https://arxiv.org/abs/2011.13456
[r8]: https://arxiv.org/abs/2006.11239
[r9]: https://arxiv.org/abs/2210.02747
[r10]: https://arxiv.org/abs/2211.01095
[r11]: https://arxiv.org/abs/2010.02502
[r12]: https://arxiv.org/abs/2209.03003

