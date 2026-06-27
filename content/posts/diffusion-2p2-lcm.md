---
title: "扩散模型囫囵吞枣 (2.2)：Latent Consistency Models"
date: "2023-11-26 22:40"
slug: diffusion-2p2-lcm
order: 2.2
original_url: "https://zhuanlan.zhihu.com/p/668940718"
summary: "免责声明：鸽子只是一个平凡的数学爱好者，如果blog里出现了错误还请大佬们指正... 前情提要如果是鸽子 之前blog的读者，应该还记得Diffusion Model作为SDE，和ODE的对应关系：每一个SDE都有一个唯一对应的ODE，而…"
source: 知乎专栏
---
免责声明：鸽子只是一个平凡的数学爱好者，如果blog里出现了错误还请大佬们指正...

## 前情提要

如果是鸽子[之前blog](https://zhuanlan.zhihu.com/p/643961621)的读者，应该还记得Diffusion Model作为[SDE](https://zhida.zhihu.com/search?content_id=236783868&content_type=Article&match_order=1&q=SDE&zhida_source=entity)，和ODE的对应关系：每一个SDE都有一个唯一对应的ODE，而一个ODE对应一个SDE族。

[之前讲EDM\[5\]的时候提到过](https://zhuanlan.zhihu.com/p/645971110)，为了让SDE的转移概率是Gaussian，通常来说SDE两项对x都必须是线性的，所以一共也没有几种可行设置。去年最流行的经典Diffusion设置自然是EDM（事实上[DDIM](https://zhida.zhihu.com/search?content_id=236783868&content_type=Article&match_order=1&q=DDIM&zhida_source=entity)也用了同款取值），查EDM论文表1可知其各项的取值是

$$
\sigma(t)=t, s(t)=1
$$

翻译成SDE函数里的f和g就是

$$
f(t)=0, g(t)=\sqrt{2t}
$$

于是，对应的ODE是（记$\mathbf{s}_{\theta}(\mathbf{x}_t, t) :\approx \nabla_{\mathbf{x}}\log p_t(\mathbf{x})$是score function对应的nn）

$$
\frac{d}{dt}\mathbf{x}_t = -t \mathbf{s}_{\theta}(\mathbf{x}_t, t)
$$

截至2022年，解这个ODE（也就是从noise恢复原图）的路数主要有两类：

-   一类是数值法硬解这个ODE（比如Euler，Heun或者DPM-Solver++，当然也包括Ancestral Sampling之类不ODE的方法）——但即使2022年最好的Solver也需要10+步迭代来得到一个好结果。
-   另一类是蒸馏（Distillation，比如[这篇blog](https://zhuanlan.zhihu.com/p/668704605)的SDS和VSD），能蒸出来当然很好（因为从generator你就能1步出图了），但是首先蒸馏本身很贵，其次你出的图quality不太能进一步提高（

那么，有没有什么办法，能比较便宜的1步得到可以接受的近似解，同时又能trade compute for quality呢？

## EDM的“[Denoiser](https://zhida.zhihu.com/search?content_id=236783868&content_type=Article&match_order=1&q=Denoiser&zhida_source=entity)”

在介绍EDM那篇blog里我们唐突引入了Denoiser但没有介绍其动机，所以在开始介绍CM之前先简单补一下。

在EDM的设定下，由$\sigma(t)$和$s(t)$的定义可知$p_{0\sigma}(\mathbf{x}_\sigma|\mathbf{x}_0)\sim \mathcal{N}(\mathbf{x}_0, \sigma^2\mathbf{I})$，也就是说我们这个本质是往一个不变的原图$\mathbf{x}_0$上直接加强度为$\sigma$的Gaussian noise。

> 值得指出的一点是，虽然鸽子之前的所有blog里都使用了0，但SDE/ODE这个连续时间的formulation是不能把时刻t倒到0的（见Song之前的几篇文章），所以其实只能停在一个略大于0的时刻$\epsilon=0.002$，Denoiser各项的系数也需要相应的修正以让目标是$\mathbf{x}_\epsilon$而非$\mathbf{x}_0$。但感觉大家都把0和$\epsilon$随便混着用，明白意思就好（

如果在EDM这个设定下，我们想要找到一个“Denoiser” D，以把DSM Loss形式上写成对真值，而非对score或对noise的Loss：

$$
\mathbb{E}_{\sigma\sim p_{\mathrm{{train}}}}\left[\lambda(\sigma)\ \mathbb{E}_{\mathbf{y}\sim p_{\mathrm{data}}}\ \mathbb{E}_{\mathbf{n}\sim N(0,\sigma^{2})}\ \|D_{\theta}(\mathbf{y}+\mathbf{n};\sigma)-\mathbf{y}\|_{2}^{2}\right]
$$

那么Denoiser应该形如

$$
D(\mathbf{x}, \sigma):=\mathbf{x}+\sigma^2\nabla_{\mathbf{x}}\log p_\sigma(\mathbf{x},\sigma)
$$

> 这个乍一看有点唬人（原文证明了一大圈），但你用传统思路想一下，传统上我们用$\mathbf{\epsilon}_\theta(\mathbf{x}_t, t)=-\sigma_t\nabla_{\mathbf{x}}\log p_t(\mathbf{x}_t, t)$来解释score function，而$\mathbf{\epsilon}_\theta(\mathbf{x}_t, t)$是用来和$\mathbf{\epsilon}$做DSM的，所以这玩意形式上就是$D(\mathbf{x}, \sigma) \sim \mathbf{x} - \sigma \mathbf{\epsilon}$，是不是就觉得合理多了（

鸽子乍一看，第一反应是哇这个形式这么好，那如果DSM Loss收敛到0了，我是不是提供任意一个image加上已知强度的噪音$\mathbf{y}+\mathbf{n}$，喂给D就能single step denoise出$\mathbf{y}$呢？

很可惜，这是错误的，因为DSM并不收敛到0；事实上对一个有限的训练集，因为我们的分布是Gaussian的，你可以解析的把D的表达式算出来：

$$
D(\mathbf{x}, \sigma)=\frac{\sum_i \mathcal{N}(\mathbf{x}; \mathbf{y}_i, \sigma^2 I) \mathbf{y}_i}{\sum_i \mathcal{N}(\mathbf{x}; \mathbf{y}_i, \sigma^2 I)}
$$

概括来说，$\sigma$越接近0，D越接近原图；而$\sigma$越大，D越接近整个训练集的均值，看上去就像一团浆糊，某种程度上也像把原图Gaussian Blur拉满的样子（EDM Figure 1b）。

> 注意D并非在逼近$\mathbf{x} - \sigma \mathbf{\epsilon}$，后者对大$\sigma$看起来应该像一坨Gaussian Noise（EDM Figure 1a），而不是像一团浆糊。

除此之外，EDM发现直接搞个neural network来预测$\mathbf{s}_{\theta}$是不行的，需要一些preconditioning（Section 5）：

$$
D(\mathbf{x};\sigma)=c_{\mathrm{skip}}(\sigma)\mathbf{x} + c_{\mathrm{out}}(\sigma)\mathbf{F}_{\theta}(c_{\mathrm{in}}(\sigma)\mathbf{x};c_{\mathrm{noise}}(\sigma))
$$

这里$\mathbf{F}$是真正的raw neural network。

## Consistency Models \[1\]

### 动机

CM的思路很自然，因为ODE是deterministic的，对每个初始点都是一条连续的trajectory，因此我们可以搞一个迭代式的Solver：

-   1步就能从noise出一个可以接受的sample
-   多迭代几步可以得到更精细的sample

那有什么现成的函数结构follow ODE trajectory呢？理论上说你按着ODE Solver的循环走的话，理想情况下确实也是沿着trajectory走，但并不能一步就先到一个可以接受的sample（因为ODE solver里t的取值总归是线性或者对数线性的），所以我们需要一些更激进的策略，更具体地说是想要一个nn直接吃noise出原图。

有了，我们刚才在EDM那里做梦的时候，梦里的Denoiser（DSM=0的情况下）是不是好像就满足这个要求，如果它真能把所有加了噪音的图都变回$\mathbf{x}_{\epsilon}$的话。

于是作者大笔一挥：现在你就以Consistency Model这个新名字重生了（

> 所以CM的模型结构长得和Diffusion一模一样（甚至如果蒸馏的话，模型weight就直接initialize成原模型的weight了）——毕竟连Denoiser形式都长得一样嘛（

为了和EDM的那个Denoiser $D$ 区分开，这里我们就管CM的真 $\cdot$ Denoiser叫$f$了，定义为

$$
f(\mathbf{x};\sigma)=c_{\mathrm{skip}}(\sigma)\mathbf{x} + c_{\mathrm{out}}(\sigma)\mathbf{F}_{\theta}(\mathbf{x};\sigma)
$$

于是顺理成章的我们有如下简洁的贪心Multi-step sampling方法：

```python
x = f(noise, T)
while 开心就好：
  选一个单调递减的时刻序列值t_i和一个N(0, 1)的noise z
  x_noise = x + \sqrt{t_i^2 - \epsilon^2} z
  x = f(x_noise, t_i)
```

现在只剩下一个问题：刚才已经说了用DSM train出$f$是做梦，那你这个$f$怎么来呢？

### Consistency Distillation

一种答案是蒸。

很不严谨的来说，DSM实际上相当于每一项对第一项取loss：

$$
\mathcal{L}_{\text{DSM}}(\mathbf{x}, n) = \left<f_\theta(\mathbf{x}_{t_{n}}, t_{n}), \mathbf{x}_{\epsilon}\right>_{\text{L2}}
$$

这个loss太松了。但我们可以把Loss改成相邻两项，而非每一项对第一项：

$$
\mathcal{L}_{\text{CD}}(\mathbf{x}, n) = \left<f_\theta(\mathbf{x}_{t_{n+1}}, t_{n+1}), f_\theta(\mathbf{x}_{t_{n}}, t_{n})\right>_{\text{某种metric，比如L2或者LPIPS}}
$$

> 严谨的证明就懒得截图了，鸽子确实也没有读。很不严谨的说，鸽子感觉这近似于要求f对t的导数处处尽可能小，收敛的理想情况下应该f对t变成常数，也就是“consistency”。

这里给定$\mathbf{x}_{t_{n+1}}\sim \mathcal{N}(\mathbf{x}, t_{n+1}^2 I)$从data采样，$\mathbf{x}_{t_{n}}$其实是未知的，但我们知道首先它应该在ODE trajectory上，而且我们知道这个ODE的表达式的估测值（给定一个已经训好的模型，我们能拿到$\mathbf{s}_{\phi}$），因此可以用ODE solver做一步数值解来逼近，记作$\hat{\mathbf{x}}_{t_{n}}^\phi\approx \mathbf{x}_{t_{n}}$。

除此之外，另一个变化是这里$\mathbf{x}_{t_{n}}$项用的并非$\theta$（online network），而是$\theta$的[EMA](https://zhida.zhihu.com/search?content_id=236783868&content_type=Article&match_order=1&q=EMA&zhida_source=entity)，记作$\theta^-$（target network），以使得结果更稳定。

> 熟悉的小伙伴可能一看到EMA和stopgrad立刻想到了contrastive learning——事实上作者说这套语言确实来自著名的[BYOL](https://zhida.zhihu.com/search?content_id=236783868&content_type=Article&match_order=1&q=BYOL&zhida_source=entity)\[4\]。  
>   
> 题外话：然后鸽子今天才发现Kaiming的[SimSiam](https://zhida.zhihu.com/search?content_id=236783868&content_type=Article&match_order=1&q=SimSiam&zhida_source=entity)\[6\]指出EMA并不是BYOL不collapse的根本原因（虽然他们证明了EMA确实能涨点），而stopgrad和predictor才是BYOL不collapse的原因（

于是，正式的loss term是

$$
\mathcal{L}_{\text{CD}}(\theta, \theta^-, \phi) := \mathbb{E}_{\mathbf{x}, n}[\lambda (t_n) \left<f_\theta(\mathbf{x}_{t_{n+1}}, t_{n+1}), f_{\theta^-}(\hat{\mathbf{x}}_{t_{n}}^\phi, t_{n})\right>_{\text{某种metric}}]
$$

> 实践上作者们发现LPIPS比L1和L2好用，Heun比Euler好用（合理）。

### Consistency Training

之前需要pretrained model，主要是为了得到ODE表达式中的$\nabla \log p_t(\mathbf{x}_t, t)$项；但given我们是Gaussian，这玩意其实可以解析的无偏估计，这样就可以把pretrained model扔了：

$$
\nabla \log p_t(\mathbf{x}_t, t) = -\mathbb{E}\left[\frac{\mathbf{x}_t - \mathbf{x}}{t^2}|\mathbf{x}_t\right]
$$

所以以下的Loss理论上也是work的：

$$
\mathcal{L}_{\text{CT}}(\theta, \theta^-) := \mathbb{E}_{\mathbf{x}, n, \mathbf{z}\sim \mathcal{N}(0, I)}[\lambda (t_n) \left<f_\theta(\mathbf{x} + t_{n+1}\mathbf{z}, t_{n+1}), f_{\theta^-}(\mathbf{x} + t_{n}\mathbf{z}, t_{n})\right>_{\text{某种metric}}]
$$

> 当然，训练起来就慢多了...另一方面CT也没有teacher可以用来初始化weight。在实验里可以看到CD指标显著的比CT好（

## Latent Consistency Models \[2\]

[之前](https://zhuanlan.zhihu.com/p/645326315)介绍过LDM。LCM的思路和LDM一模一样，只不过把DSM Loss换成了CD Loss（称为Latent Consistency Distillation，LCD，Algorithm 1）。

如果LCD已经收敛的差不多了，也可以把Loss改成CT Loss对$\theta$进行finetune（Latent Consistency Fine-tuning，LCF，Algorithm 4），这样就可以把teacher diffusion model扔掉了。

### Skipping Time Step (Sec 4.3)

这是一项比起CM的重大改进。之前提到CD是$t_{n+1}$和$t_n$做loss，因为这两步非常接近，所以导致CD收敛很慢。作者们发现其实可以多跳几步，变成$t_{n+k}$和$t_n$做loss，可以极大的增加收敛速度，同时几乎不影响指标。

从$\mathbf{x}_{t_{n+k}}$计算$\mathbf{x}_{t_n}$对各solver的closed form formula见Appendix E。

> 注意[CFG](https://zhida.zhihu.com/search?content_id=236783868&content_type=Article&match_order=1&q=CFG&zhida_source=entity)(Classifier-free guidance)是加在noise prediction上的：  
> $\tilde{\mathbf{\epsilon}}_\theta(z_t, \omega, c, t):= (1+\omega)\mathbf{\epsilon}_\theta(z_t, c, t) - \omega \mathbf{\epsilon}_\theta(z_t, \emptyset, t)$  
> 所以ODE的式子也需要相应修改。

实验发现对经典1k step的Diffusion Model，k=20比较好，更大的k会导致累计误差比较严重，从而影响performance。

LCF和LCD一样也可以跳步。

### Encode CFG scale $\omega$ in LCM

作者们仿效了\[7\]，把$\omega$的傅里叶级数像$t$一样喂给LCM。

## LCM-LoRA \[3\]

LCM什么都好，但有一个巨大问题：之前提到LCM的模型结构和对应的LDM是一样的，可finetune SDXL也太贵了。

然后很自然的就会想到：既然LCM是用LDM initialize的，那我...贴个LoRA是不是也是一样的（

结论是确实，不但如此，还能和Style LoRA线性叠加，十分甚至九分的科学（

> [有好心人](https://wrong.wang/blog/20231111-consistency-is-all-you-need/)试了一下发现Finetune LCM-LoRA的时候记得把$\omega$弄小点，然后诚意推荐了$\omega=1.5$（

## 参考资料

\[1\] [Consistency Models](https://arxiv.org/abs/2303.01469)

\[2\] [Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378)

\[3\] [LCM-LoRA: A Universal Stable-Diffusion Acceleration Module](https://arxiv.org/abs/2311.05556)

\[4\] [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733)

\[5\] [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)

\[6\] [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566)

\[7\] [On Distillation of Guided Diffusion Models](https://arxiv.org/abs/2210.03142)

> 本文使用 [Zhihu On VSCode](https://zhuanlan.zhihu.com/p/106057556) 创作并发布
