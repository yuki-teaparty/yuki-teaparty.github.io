---
title: "家用扩散模型 (2.1)：Variational Score Distillation"
date: "2023-11-26 00:45"
slug: diffusion-2p1-vsd
draft: false
order: 2.1
series: "家用扩散模型"
original_url: "https://zhuanlan.zhihu.com/p/668704605"
summary: "有关Variational Score Distillation及其变体"
source: 知乎专栏
---

## 动机

假设我们训（下）好（载）了一个pretrained diffuser $\boldsymbol{\epsilon}_\psi(\boldsymbol{x}_t, t, c)$（也可以认为是score function，之前已经介绍过了这两者是一样的），能帮我们轻松的计算$p(\boldsymbol{x}_0|c)$。

> 其实一点也不轻松，即使是之前介绍的DPM Solver 2m++，也要10~20 steps（

现在我们想要解决如下问题：对生成函数$x=g(z;\theta, c)$，条件c和隐分布z，优化g的参数$\theta \in \Theta$使得x在pretrained diffuser的分布内（常见的应用比如，我们想要学一个nerf让它渲染的图片x看起来真实——我们今天要介绍的论文相当程度上围绕这个主题展开）。

这时候，一个自然而然的想法是——既然我们已经有了pretrained diffuser，那如果我们直接做denoising score matching呢？也就是说，先把生成的图片加噪声

$$
\boldsymbol{x}_t=\alpha_t\boldsymbol{x}_0+\sigma_t\epsilon,\quad \boldsymbol{x}_0=g(z;\theta, c)
$$

然后最小化DSM Loss。这意味着极值处的$\nabla_\theta$是0:

$$
0=\nabla_\theta\mathcal{L}_{\text{DSM}}=\mathbb{E}_{t, \epsilon,z} \left[\omega(t)(\boldsymbol{\epsilon}_\psi(\boldsymbol{x}_t, t, c)-\epsilon)(\nabla_\theta\boldsymbol{\epsilon}_\psi(\boldsymbol{x}_t, t, c))\right]
$$

根据链式法则，

$$
\nabla_\theta\boldsymbol{\epsilon}_\psi(\boldsymbol{x}_t, t, c)=\nabla_{\boldsymbol{x}}\boldsymbol{\epsilon}_\psi(\boldsymbol{x}_t, t, c)\cdot \alpha_t\frac{\partial g(z, \theta, c)}{\partial\theta}
$$

-   后半段是我们自己设计控制的小模型，我们可以用backprop轻松算出梯度。
-   但前半段是在一个极其巨大的pretrained diffuser上求Jacobian，且不论这玩意计算起来有多昂贵，光是数值稳定性就是一大难题

所以直接最小化DSM Loss非常困难。那怎么办呢？我们需要Score Distillation。

## Variational Score Distillation \[2,3\]

我们用一个新Diffuser，$\boldsymbol{\epsilon}_\phi(\boldsymbol{x}_t, t, c)$，来预测g生成样本的概率q的分数$-\sigma_t\nabla_{\boldsymbol{x}}\log q_t(\boldsymbol{x}_t, t, c)$。

我们的优化目标是(推导过程见下文)

$$
\nabla_\theta\mathcal{L}_{\text{VSD}}=\mathbb{E}_{t,z,\epsilon} \left[\omega(t)(\boldsymbol{\epsilon}_{\psi}(\boldsymbol{x}_t, t, c)-\boldsymbol{\epsilon}_\phi(\boldsymbol{x}_t, t, c))\frac{\partial g(z, \theta, c)}{\partial\theta} \right]
$$

这里$\boldsymbol{\epsilon}_{\psi}$来自预训练的大模型。

这个函数就是可以计算的了。实现上我们交替优化$\phi$和$\theta$：

-   优化$\phi$的时候使用DSM Loss（也就是$\phi$对$\boldsymbol{\epsilon}$——但计算这个DSM Loss的时候因为x是从g采样来的，实际上还是用了generator），从而让$\boldsymbol{\epsilon}_\phi(\boldsymbol{x}_t, t, c)$逼近我们想要的真值$-\sigma_t\nabla_{\boldsymbol{x}}\log q_t(\boldsymbol{x}_t, t, c)$。
-   优化$\theta$的时候算VSD Loss，让生成分布和真实分布靠拢。

当我们sample的时候，只要先采样一个z，然后用g生成一个x，这意味着我们一步就能完成inference，再也不需要走昂贵的diffusion sampling了（

至于为什么叫Variational Score Distillation，那是因为还有一个不variational的Score Distillation Sampling\[1\]，请见下文。

> 题外话：Variational Score Distillation（VSD）是\[2\]中的名字。\[3\]中有一个非常接近的term，叫Integral Kullback-Leibler divergence（IKL），仔细对比一下就会发现\[2\]和\[3\]只差了个只和t有关的系数，可以简单的塞进weight里。这是CMU的Zhengyang Geng大佬发现的。\[2\]和\[3\]的发布日期其实也相当接近，可以说是比较机缘巧合。

### 推导思路

> 详见\[3\]的A.2和\[2\]的C.4。

我们记隐分布为$z\sim q(z)$，生成的分布为$x=g(z; \theta)\sim q_\theta(x)$，而真实数据的分布是$x\sim p(x)$（这里省略了c和t，但是不本质）

我们显然想要生成分布和真实分布尽量接近，一个自然的想法就是令两者的KL为0：

$$
\begin{aligned}
0=&\nabla_\theta KL(q_\theta(x)\|p(x)) \\
=& \nabla_\theta \mathbb{E}_{x\sim q_\theta(x)}[\log q_\theta(x)-\log p(x)]\\
=& \nabla_\theta \mathbb{E}_{z\sim q(z)}[\log q_\theta(g_\theta(z))-\log p(g_\theta(z))] \\
=&\mathbb{E}_{z\sim q(z)}[ \nabla_\theta \log q_\theta(g_\theta(z))- \nabla_\theta \log p(g_\theta(z))] \\
=&\mathbb{E}_{z\sim q(z)}[ \nabla_\theta \log q_\theta(x)|_{x=g_\theta(z)}+\frac{\partial x}{\partial \theta}\nabla_x\log q_\theta(x)|_{x=g_\theta(z)}- \frac{\partial x}{\partial \theta}\nabla_x \log p(x)|_{x=g_\theta(z)}] \\
\end{aligned}
$$

而

$$
\begin{aligned} &\mathbb{E}_{z\sim q(z)}[ \nabla_\theta \log q_\theta(x)|_{x=g_\theta(z)}]\\ =&\mathbb{E}_{x\sim q_\theta(x)}[ \nabla_\theta \log q_\theta(x)]=\int q_\theta(x) \frac{\nabla_\theta q_\theta(x)}{q_\theta(x)} dx \\ =&\nabla_\theta \int q_\theta(x)dx\\ =&\nabla_\theta \boldsymbol{1}=0 \end{aligned}
$$

所以

$$
\begin{aligned} \nabla_\theta KL(q_\theta\|p(x)) = \mathbb{E}_{z\sim q(z)}[(\nabla_x\log q_\theta(x)- \nabla_x \log p(x))\frac{\partial x}{\partial \theta}|_{x=g_\theta(z)}] \\ \end{aligned}
$$

到这里已经长出了VSD的主要形状（注意$\nabla_x \log p(x)$和$\boldsymbol{\epsilon}$差了个负号），剩下来的权重部分就只和t有关了。

## Score Distillation Sampling \[1\]

SDS的优化目标是

$$
\nabla_\theta\mathcal{L}_{\text{SDS}}=\mathbb{E}_{t,z,\epsilon} \left[\omega(t)(\boldsymbol{\epsilon}_{\psi}(\boldsymbol{x}_t, t, c)-\boldsymbol{\epsilon})\frac{\partial g(z, \theta, c)}{\partial\theta} \right]
$$

这里$\boldsymbol{\epsilon}_{\psi}$来自预训练的大模型。

考虑VSD这个问题。给定条件c和先验z，事实上我们可以有很多个$\theta$，使得$g(z;\theta, c)$符合要求。因此我们需要的其实是一个variational的$\theta$分布——但SDS其实是$\theta \sim \delta(\theta-\theta^*)$，此时有$q_\theta \sim \mathcal{N}(\boldsymbol{x}_t|\alpha_t g(z; \theta^*, c), \sigma_t^2 \boldsymbol{I})$，于是$-\sigma_t \nabla_x\log q_\theta(x) = \boldsymbol{\epsilon}$。

因此，SDS可以视为VSD在不Variational时的特殊情况，它的效果更差也自然是可以理解的了。

> SDS本身的动机其实是从$\nabla_\theta\mathcal{L}_{\text{DSM}}$中扔掉难以计算的巨大Jacobian项，事后才想着从KL方面圆回来（见\[1\]的appendix），而且推导过程十分相像，实际上和\[2\]\[3\]只差一层窗户纸，只能说\[1\]没有直接进到VSD真是机缘巧合（  
>   
> 但话又说回来，实际上大家都负担不起train一个新diffuser，所以还是SDS用的多...

## VSD和GAN的关系

> 见\[3\]的A.4

VSD的这个交替优化是不是让你想到了什么呢？没错，这玩意看着像GAN！甚至GAN的著名trick之一就是往data里掺gaussian noise。

最传统的GAN的discriminator output $h(\cdot)$是一个sigmoid，然后出来的结果做Binary Cross Entropy。使用同样的参数符号，可以想象$\theta$是generator的参数，而$\phi$是discriminator的参数（方便类比），我们有：

$$
\mathcal{L}_D = -\mathbb{E}_{x\sim p(x)}[\log h_\phi(x)] - \mathbb{E}_{z\sim q(z), x=g_\theta(z)}[\log (1-h_\phi(x))]
$$

它在h把所有假样本判成0而真样本判成1的时候取得极小值0。

$$
\mathcal{L}_G = -\mathbb{E}_{z\sim q(z), x=g_\theta(z)}[\log h_\phi(x)]
$$

它在h把所有假样本判成1的时候取得极小值0。

和VSD类比一下：

-   D loss的目标是让$h_\phi(x)$逼近$\frac{p(x)}{q_\theta(x)+p(x)}$。

这是因为如果简单列个算式，D loss的极值处有

$$
0=\nabla_\phi \mathcal{L}_D = \int (-\frac{p(x)}{h_\phi(x)} + \frac{q_\theta(x)}{1-h_\phi(x)}) \nabla_\phi h_\phi(x) dx
$$

-   G loss的目标本身很难处理（因为这个log概率和很难处理），但我们既然已经知道了$h_\phi(x)$的真实目标，可以偷偷的换一个目标一致的loss：

$$
\mathcal{L}_G^{KL} = \mathbb{E}_{z\sim q(z), x=g_\theta(z)}[\log \frac{1-h_\phi(x)}{h_\phi(x)}]
$$

于是G loss的极值处有（把$h_\phi(x)$换成其理想值$\frac{p(x)}{q_\theta(x)+p(x)}$）

$$
\begin{aligned}
0 &= \nabla_\theta \mathcal{L}_G^{KL} \\
&= \mathbb{E}_{z\sim q(z)} [\nabla_\theta \log \frac{1-h_\phi(x)}{h_\phi(x)}]\\
&\approx \mathbb{E}_{z\sim q(z)} [\nabla_\theta \log q_\theta(g_\theta(z)) - \nabla_\theta \log p(g_\theta(z))] \\
&= \mathbb{E}_{z\sim q(z)}[(\nabla_x\log q_\theta(x) - \nabla_x \log p(x))\frac{\partial x}{\partial \theta}|_{x=g_\theta(z)}] 
\end{aligned}
$$

这样就长得和VSD（的不加noise, t==0的版本）异曲同工（甚至可以说是一模一样）了。

> 所以其实有点好奇，如果用$\mathcal{L}_G^{KL}$代替$\mathcal{L}_G$，train GAN会变得更稳定吗？  
>   
> 可惜生产中大家人均WGAN-GP，早就没有人用传统GAN了...

## 参考资料

\[1\] [DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988)

\[2\] [ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation](https://arxiv.org/abs/2305.16213)

\[3\] [Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models](https://arxiv.org/abs/2305.18455)
