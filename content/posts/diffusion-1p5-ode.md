---
title: "家用扩散模型 (1.5)：从SDE到ODE"
date: "2023-07-18 23:36"
slug: diffusion-1p5-ode
draft: false
order: 1.5
series: "家用扩散模型"
original_url: "https://zhuanlan.zhihu.com/p/643961621"
summary: "在上一期我们速成了SDE，这一期我们要从SDE推导出ODE。"
source: 知乎专栏
---
## 从SDE到ODE

### 关于上一篇的一些补充...

为了提醒这是个矩阵，这次用一个大写字母 $\boldsymbol{D}(x, t) = \frac{1}{2}\boldsymbol{G}(x, t)\boldsymbol{G}^T(x, t)$ 来表示扩散系数。

仔细读了读[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)的附录，发现其实上一篇最后谈reverse-time diffusion equation models的时候，注意到对任意D（或者说任意G）都有

$$
\frac{1}{p(x, t)}\nabla \cdot(p(x, t)\boldsymbol{D}^i) = \nabla\cdot \boldsymbol{D}^i + \frac{1}{p(x, t)}\boldsymbol{D}^i\cdot \nabla p(x, t) = \nabla\cdot \boldsymbol{D}^i +\boldsymbol{D}^i\cdot \nabla\log p(x, t)
$$

这里和 $\boldsymbol{G}(x, t) = g(t)$ 没有关系，对普通的G也是成立的。

### Probability Flow ODE（Section 4.3/Appendix D.1）

这里SDE中使用原文的符号f和G。

众所周知，ODE的性质那是相当的好——好就好在ODE是deterministic的，任何X\_0和X\_t都有一一对应的关系，而且有一系列ODE solver可用。
与此同时，SDE有个噪声dW，因此不是deterministic的，每一步都有随机性。但现在有一个好消息：

对于任意SDE $d\boldsymbol{X} = \boldsymbol{f}(\boldsymbol{X}, t)dt + \boldsymbol{G}(\boldsymbol{X}, t)d\boldsymbol{W}$ ，存在一个ODE $d\boldsymbol{X}(t) = \tilde{\boldsymbol{f}}(\boldsymbol{X}, t)dt$ 使得两个X的概率分布一致。

那么，这个好事是怎么来的呢？我们回忆一下上一期讲过的KFE。

$$
\begin{aligned} \partial_t p(x, t) &= -\nabla \cdot (\boldsymbol{f}(x, t) p(x, t)) + \nabla^2: (\boldsymbol{D}(x, t) p(x, t)) \\ &= -\nabla \cdot (\boldsymbol{f}(x, t) p(x, t))  + \sum_i \frac{\partial}{\partial x_i} (\nabla\cdot (\boldsymbol{D}^ip(x, t))) \\ &= -\nabla \cdot (\boldsymbol{f}(x, t) p(x, t))  + \sum_i \frac{\partial}{\partial x_i} (p(x, t) (\nabla\cdot \boldsymbol{D}^i + \boldsymbol{D}^i\nabla \log p(x, t))) \\ &= -\nabla \cdot (\boldsymbol{f}(x, t) p(x, t))  + \nabla\cdot (p(x, t) (\nabla\cdot \boldsymbol{D} + \boldsymbol{D}\nabla \log p(x, t))) \\ &= -\nabla \cdot ((\boldsymbol{f}(x, t) - \nabla\cdot \boldsymbol{D} - \boldsymbol{D}\nabla \log p(x, t)) p(x, t))  + \nabla^2:\boldsymbol{0} \end{aligned}
$$

然后最后这个式子再KFE转回去，就得到了我们想要的ODE：

$$
d\boldsymbol{X} = (\boldsymbol{f}(x, t) - \nabla\cdot \boldsymbol{D}(x, t) - \boldsymbol{D}(x, t)\nabla \log p(x, t))dt + \boldsymbol{0}d\boldsymbol{W}
$$

> 注：事实上，根据相同的逻辑，不难构造出一个SDE族 $d\boldsymbol{X}(t) = \tilde{\boldsymbol{f_{\boldsymbol{\sigma}}}}(\boldsymbol{X}, t)dt + \boldsymbol{\sigma}(\boldsymbol{X}, t)d\boldsymbol{W}$，使得里面的每个成员都有相同的概率分布——只要少从右边往左边划一些就可以了。

特别的，在扩散模型中， $\boldsymbol{G}(x, t) = g(t)$ ，所以 $\nabla \cdot \boldsymbol{D}=\boldsymbol{0}$ ，于是我们有：

$$
d\mathbf{x} = \{\boldsymbol{f}(\mathbf{x}, t) - \frac{1}{2} \boldsymbol{G}(t)\boldsymbol{G}^T(t)\nabla_{\mathbf{x}} \log p(\mathbf{x}, t)\}dt := \tilde{\boldsymbol{f}} (\mathbf{x}, t) dt
$$

### Exact Likelihood from ODE （Appendix D.2）

这正是Score-based Diffusion在inference时击败DDPM的秘诀——DDPM是在SDE上离散的采样，这个是解析的，所以理论上说可以做得更好（只要中间step数够多）

具体的，对 $d\mathbf{x} = \tilde{\boldsymbol{f}} (\mathbf{x}, t)dt$ ，由instantaneous change of variables formula（见[Neural ODE](https://arxiv.org/pdf/1806.07366.pdf)的Appendix A）有

$$
\log p_0(\mathbf{x}_0) = \log p_T (\mathbf{x}_T) + \int_0^T \nabla\cdot \tilde{\boldsymbol{f}} (\mathbf{x}, t) dt
$$

其中， $\nabla\cdot \tilde{\boldsymbol{f}} (\mathbf{x}, t)$ 的计算非常昂贵，我们采用Skilling-Hutchinson trace estimator，这是一个无偏估计：

$$
\nabla\cdot \tilde{\boldsymbol{f}} (\mathbf{x}, t) =E_{\boldsymbol{\epsilon}\sim p(\boldsymbol{\epsilon})}[\boldsymbol{\epsilon}^T(\nabla\tilde{\boldsymbol{f}}) (\mathbf{x}, t) \boldsymbol{\epsilon}]
$$

这里 $\nabla\tilde{\boldsymbol{f}}$ 是jacobian， $\boldsymbol{\epsilon}$ 服从均值0方差I。

因为这是无偏估计，所以可以通过采样若干次$\boldsymbol{\epsilon}$取平均（这个e^TJe可以用gradient简单的计算）来逼近。

## Score Function—— $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$

好消息是这个函数不是真正的概率，不需要normalize到1，但是又连续——于是一个自然的动机是用一个NN $\boldsymbol{s}_{\boldsymbol{\theta}}$ 来估测。

于是，这个近似的reverse SDE就会变成：

$$
\mathrm{d} \hat{\mathbf{x}}=\left[\boldsymbol{f}(\hat{\mathbf{x}}, t)-g(t)^2 \boldsymbol{s}_{\boldsymbol{\theta}}(\hat{\mathbf{x}}, t)\right] \mathrm{d} t+g(t) \mathrm{d} \overline{\mathbf{w}}, \quad \hat{\mathbf{x}}_{\boldsymbol{\theta}}(T) \sim \pi
$$

这样解出来的x的概率分布（显然是conditioned on $\theta$ 的）记作 $p_{\boldsymbol{\theta}}^{SDE}$ 。

## 优化目标

现在我们万事俱备，只差一个loss。

### Score Matching

因为我们在用一个nn估测一个函数，所以我们很自然的想要优化这样一个目标：

$$
\mathcal{J}_{\mathrm{SM}}(\boldsymbol{\theta} ; \lambda(\cdot)):=\frac{1}{2} \int_0^T \mathbb{E}_{p_t(\mathbf{x})}\left[\lambda(t)\left\|\nabla_{\mathbf{x}} \log p_t(\mathbf{x})-\boldsymbol{s}_{\boldsymbol{\theta}}(\mathbf{x}, t)\right\|_2^2\right] \mathrm{d} t
$$

其中 $\lambda$ 是一个我们选择的weight。原文用J大概是因为上式其实是一个能量。

这个目标是有数学理论保证的。同作者的\[2\]表明，如果取 $\lambda(t)=g(t)^2$ ，我们有一个很好的上界估计：

$$
D_{\mathrm{KL}}\left(p \| p_{\boldsymbol{\theta}}^{\mathrm{SDE}}\right) \leqslant \mathcal{J}_{\mathrm{SM}}\left(\boldsymbol{\theta} ; g(\cdot)^2\right)+D_{\mathrm{KL}}\left(p_T \| \pi\right)
$$

当 $\boldsymbol{s}_{\boldsymbol{\theta}} \equiv \nabla_{\boldsymbol{x}} \log q_t(\boldsymbol{x}), q_T\sim \pi$ 时等号成立，且 $p_{\boldsymbol{\theta}}^{SDE}=p_{\boldsymbol{\theta}}^{ODE}=q$ 。

当然现实中如果s不精确的时候，SDE和ODE的p就对不上，但似乎empirically差不多。

### Denoising Score Matching

实际上 $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ 很难估计，所以常用的是带条件（初始值）的版本：

$$
\mathcal{J}_{\mathrm{DSM}}(\boldsymbol{\theta} ; \lambda(\cdot)):=\frac{1}{2} \int_0^T \mathbb{E}_{p(\mathbf{x}) p_{0 t}\left(\mathbf{x}^{\prime} \mid \mathbf{x}\right)}\left[\lambda(t)\left\|\nabla_{\mathbf{x}^{\prime}} \log p_{0 t}\left(\mathbf{x}^{\prime} \mid \mathbf{x}\right)-\boldsymbol{s}_{\boldsymbol{\theta}}\left(\mathbf{x}^{\prime}, t\right)\right\|_2^2\right] \mathrm{d} t
$$

> 这里 $p_{0 t}\left(\mathbf{x}^{\prime} \mid \mathbf{x}\right)$ 是transition density。按鸽子理解，这里的符号翻译一下应该是  
> $\mathbf{x} := \mathbf{x}_0, \mathbf{x}^\prime := \mathbf{x}_t,   p_{0 t}\left(\mathbf{x}^{\prime} \mid \mathbf{x}\right) := p(\mathbf{x}_t, t|\mathbf{x}_0, 0)$

[可以证明](https://kexue.fm/archives/9509)，DSM和SM只相差了一个和 $\theta$ 无关的常数，这里就不展开了。

DSM的好处如下：

-   $p_{0 t}\left(\mathbf{x}^{\prime} \mid \mathbf{x}\right)$ 是tractable的——如果扩散模型中取 $\boldsymbol{f}(\mathbf{x}, t)$ 对x线性，那么这个概率就是一个Gaussian。

-   因为是Gaussian，所以即使加上 $\nabla$ 之后，这个式子也是有closed form的。

-   这里这个“期望的积分”虽然看着吓人，但是可以蒙特卡洛的估计——不断采样（t, x, x'），其中x来自初始分布p\_0，而t从\[0, T\]中随机抽取，x'按转移概率采样出来。

关于SM和DSM的详细讨论，详见同作者的tutorial [\[3\]](https://arxiv.org/pdf/2101.03288.pdf)。

### DSM的importance sampling

之前提到了如何计算DSM。本来如果只是对积分里面这个期望E进行蒙特卡洛采样（给定t抽x），理论上说是无偏的。但在随机抓一个t采样代替积分之后，虽然采样快了十倍甚至九倍（x），但是产生了巨大的方差。

为了降低方差，我们的t不能从\[0, T\]中均匀抽取，而是要遵循一个分布。具体的分布和我们选取的SDE f和g有关。

## Evidence Lower Bound (ELBO)

大致来说，在贝叶斯体系中，我们现在有个观察值x，然后有个latent code z，它的分布p(z)我们规定好了（典型比如说gaussian），然后我们有一个模型p(x, z)，或者说p(x | z)。

现在我们想要inference：给定一个x，想要p(z | x)。但是这玩意是intractable的，不能直接用贝叶斯公式算（因为求和p(x)=\\int p(z, x) dz这一步贵到无法接受），所以我们就找一个函数q(z)来逼近它（也就是所谓变分——类似的操作[在物理里我们经常见到](https://zhuanlan.zhihu.com/p/139018146)，大家上高中或者带学的时候其实多少都学过），然后自然而然地，对于两个分布我们就会想着最小化KL散度。

现在我们像这样构造一个函数ELBO(p, q)：

> 注意p和q在现实中是两个不同的nn，有各自不同的参数，但这里懒得打所以省略掉了。

$$
\begin{aligned} \log p(\mathbf{x}) & =\mathbb{E}_q[\log p(\mathbf{x})] \\ & =\mathbb{E}_q\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{p(\mathbf{z} \mid \mathbf{x})}\right] \\ & =\mathbb{E}_q\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{p(\mathbf{z} \mid \mathbf{x})} \frac{q(\mathbf{z})}{q(\mathbf{z})}\right] \\ & =\mathbb{E}_q\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})} \frac{q(\mathbf{z})}{p(\mathbf{z} \mid \mathbf{x})}\right] \\ & =\underbrace{\mathbb{E}_q\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})}\right]}_{\mathrm{ELBO}}+\underbrace{\mathbb{E}_q\left[\log \frac{q(\mathbf{z})}{p(\mathbf{z} \mid \mathbf{x})}\right]}_{\mathrm{KL}} \end{aligned}
$$

于是：

-   ELBO是我们可以算出来的，因为q(z)和p(x, z)我们都已知。
-   虽然log p(x)和KL我们都算不出来，但如果我们最大化ELBO，有同时最大化log p(x)（所谓Maximum Likelihood Estimation；等效的，可以理解为，最小化KL(p\* || p)，其中p\*是真实数据分布），和最小化KL(q || p)的双重功效。

-   这正是我们想要的——类比一下我们熟悉的语言，可以把前一项想象为generator，后一项想象为discriminator。

-   众所周知，KL是非负的，所以ELBO就是log p(x)的下界（所以才得到了ELBO这个名字），成立当且仅当KL取0（也就是q(z)=p(z|x)）。
-   VAE其实就是在Maximize ELBO。

### Score-based Diffusion的negative log likelihood(NLL) bound

\[2\]中证明了下式（可以认为是一种连续意义上的ELBO），说明我们这样train出来的东西是有保证的。

![](/assets/img/posts/diffusion-1p5-ode/v2-7550646bf2b5efd53c22b45523d2683d.jpg)

## 如何破除高大上数学名词光环——以Langevin Dynamics为例子

Langevin Dynamics是DDPM宣称自己采样背后的数学动机，一听起来就非常高大上。

Wiki一下可以知道，它的本意是指Langevin方程，aka. ${\displaystyle m{\frac {d\mathbf {v} }{dt}}=-\lambda \mathbf {v} +{\boldsymbol {\eta }}\left(t\right)}$ 的动力学。

然后具体到DDPM，我们知道DDPM离散形式的前向MC是（这里使用DDPM原论文的符号）

$$\mathbf{x}_i=\sqrt{1-\beta_i}\mathbf{x}_{i-1}+\sqrt{\beta_i}\mathbf{z}_{i-1}$$

N趋于无穷的时候，把这个东西看成连续的，可以想象成有一个函数 $\beta(\frac{i}{N})=\frac{\beta_i}{N}$ 对每个step都趋于无穷小。

所以泰勒展开一下，连续形式的前向SDE是 

$$d\mathbf{x}=-\frac{1}{2}\beta(t)\mathbf{x}dt+\sqrt{\beta(t)}d\boldsymbol{W}$$

于是反向SDE是 

$$d\mathbf{x}=(-\frac{1}{2}\beta(t)\mathbf{x}-\beta(t)\nabla_{\mathbf{x}} \log p_t(\mathbf{x}))dt+\sqrt{\beta(t)}d\bar{\boldsymbol{W}}$$

然后你再离散回去（注意这里dt是负的），假装对dt做了一个很小的积分，于是你形式化的得到

$$\mathbf{x}_{i-1}-\mathbf{x}_i=\frac{1}{2}\beta_i\mathbf{x}_i +\beta_i\boldsymbol{s}^*_\theta(\mathbf{x}_i, i)+\sqrt{\beta_i}\mathbf{z}_i$$

然后脑补一下反向的泰勒（记住这里beta是无穷小），可以写成

$$\mathbf{x}_{i-1}=\frac{1}{\sqrt{1-\beta_i}}(\mathbf{x}_i+\beta_i\boldsymbol{s}^*_\theta(\mathbf{x}_i, i))+\sqrt{\beta_i}\mathbf{z}_i$$

这就是所谓Langevin Dynamics。这个式子也出现在\[1\]的2.2中, aka.“ancestral sampling”。

到这里就已经和DDPM形式上相当像了。这里和DDPM原文$-\frac{1}{\sqrt{1-\bar{\alpha}_t}}$差了一个系数，这是怎么回事呢？
这是因为DDPM里算的其实是 $\mathbf{x}_t\left(\mathbf{x}_0, \boldsymbol{\epsilon}\right)=\sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{1-\bar{\alpha}_t} \boldsymbol{\epsilon}$ 。
用Gaussian的PDF计算一下，就会发现 $\nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t)=\frac{\nabla_{\mathbf{x}}p_t(\mathbf{x}_t)}{p_t(\mathbf{x}_t)} = ... = -\frac{1}{\sqrt{1-\bar{\alpha}}}\boldsymbol{\epsilon}$

所以两者其实是一码事，也就是说DDPM这里所谓估 $\boldsymbol{\epsilon}_\theta$ 其实就是在估测Score Function。

题外话，在现实中：
-   Langevin Equation的 $-\mathbf{\lambda}\mathbf{v}$ 这一项其实是摩擦力（阻尼，damping），使用KFE立得 $t\to\infty$ 的时候会趋近于一个期望为0，方差随着时间变大的纯Gaussian Noise，整个drift项都被阻尼磨掉了。
-   如果是overdamping的情况，通常写作 $\lambda \frac{d x}{d t}=-\frac{\partial U(x)}{\partial x}+\eta(t)$ ，最后稳态会得到一个很漂亮的Boltzmann分布，翻译成ML的语言，就是大名鼎鼎的softmax。

## 参考资料

\[1\] [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)

\[2\] [Maximum Likelihood Training of Score-Based Diffusion Models](https://arxiv.org/abs/2101.09258)

\[3\] [How to Train Your Energy-Based Models](https://arxiv.org/abs/2101.03288)
