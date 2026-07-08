---
title: "家用扩散模型 (2)：Latent Diffusion Model"
date: "2023-08-13 16:22"
slug: diffusion-2-ldm
draft: false
order: 2
series: "家用扩散模型"
original_url: "https://zhuanlan.zhihu.com/p/645326315"
summary: "VQVAE，LDM和RoPE"
source: 知乎专栏
---
## 前文导航

在之前的三篇中，我们大致介绍了扩散模型背后的数学原理以及如何从一个训完的扩散模型中采样。

但是，还有一个最重要的问题没有解答：那我怎么估算 $\boldsymbol{\epsilon}_\theta(x, t)$ 呢？

有请本文的主角，generative model第一次火出圈的LDM \[1\] （aka Stable Diffusion 1.0）

## TL; DR

![](/assets/img/posts/diffusion-2-ldm/v2-489a1288e20b32a1b901e25a12c53317.jpg)

这篇文章的思路非常简单：

-   首先train一个VAE把high-res的图片压到一个low-res的latent space（注意被压成feature map了还是2D，和许多VAE直接压成1D latent code不同）。backbone使用了VQGAN。

> 虽然作者提了一下loss可以用传统VAE的KL loss，或者用VQGAN的quantization layer，但不管选哪个regularization，backbone结构都是一样的——因为VQGAN虽然名义上是GAN，但其实还是有encoder-decoder，不要像鸽子一样望文生义。

-   然后就可以在latent space上做diffusion了，因此我们可以通过cross attention把各种condition fuse进来，从而让各种task都复用同一个backbone。例如，如果是从text prompt出图，就把encoder从图的encoder换成文字的encoder。具体可以看同属SD系列 的[img2img源码](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py)和[text2img源码](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)。

## 从image space到latent space：VQGAN系列

首先，回忆一下没有V的AE（autoencoder）是img $x$ --> encoder $e$ --> latent $z_e(x)$ --> decoder --> output。

传统VAE可以看成，对posterior $p(z|x) = \mathcal{N}(z; \mu, \sigma^2)$ 优化ELBO。

Vector Quantization (VQ) 做了什么呢？它把$z_e(x)$给最近邻到codebook里的nearest neighbour code $z_q$。

因此VQVAE可以看成，对posterior $p(z=z_k|x) :=\boldsymbol{1}_{k=\arg \min_j\|z_e(x)-e_j\|}$ 优化ELBO。Posterior变成了一个冲击——因此他就不variational了。

回忆一下我们的ELBO，它由reconstruction项和KL项组成，reconstruction项是一致的，但代替Gaussian的KL的是VQ regularization $\|z_q-z\|_2^2$ 。

注意这个loss本身不可导（因为argmin没有导数），所以需要一个trick叫stop-gradient (字面意思，就是让$sg(x)=x$, $\frac{\partial}{\partial x}sg(x)=0$)，因此真正的loss term是 

$$\|z_q-\text{sg}(z)\|_2^2+\beta\|\text{sg}(z_q)-z\|_2^2$$

其中参数 $\beta$ 通常取0.25。详见[这里](https://kexue.fm/archives/6760#%E7%BB%B4%E6%8A%A4%E7%BC%96%E7%A0%81%E8%A1%A8)的介绍。

除了直接用这个MSE loss来update codebook之外，也可以用Exponential Moving Averages（EMA）来update codebook（相当于加了个momentum），详见VQVAE Appendix A。

那这么绕了一圈，换了一个discrete posterior，好处是什么呢？——好处是我们**tokenize**了latent space feature map! 这样你就可以做next token prediction了。这就是我们要介绍的VQGAN\[2\]：

![](/assets/img/posts/diffusion-2-ldm/v2-2d5c4ad5250d771edd29d71ef4f4275c.jpg)

这里就是train了一个autoregressive model，然后从第一个token开始不断做next token prediction，最后从所有token decoder回图。

当然，token prediction的顺序当然会影响结果，但实际上只要从左上开始按顺序一路采样就行。作者们在附录里作了一番实验，结果惊奇的发现按顺序效果最好（而不是那些特意设计的采样顺序）

另外还有一个补丁：transformer的长度是有限制的（不能给太大的context window），但CNN Encoder-Decoder不能层数太多，不然VQVAE无法recon。因此作者们使用一个sliding attention window让某一个patch的token概率只和邻近的patch有关。

Google此后又做了ViT-VQGAN\[3\]。相对VQGAN，首先backbone encoder/decoder换成ViT，其次是对vq层codebook的改进，包括两条：
-   normalized code，这样code到code的l2 distance就可以被解释为cosine similarity。
-   传统VQGAN一直受到codebook usage不足的困扰（大多数code根本没人用），作者提出只要简单地把高维的code project到低维（比如32维），然后在低维做最近邻，就能有效解决usage的问题。
> 这也非常合理，想象一下一个1024维的球面，球面上随便点两个vector几乎肯定是正交的，但在32d上就好多了。

## U-Net Denoiser

> 一个更好的介绍在这里，是SD源码的逐行解析：[U-Net for Stable Diffusion](https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html)

## Unconditioned的情况： $\boldsymbol{\epsilon}_\theta(x, t)$

U-Net大家都十分熟悉了，四舍五入约等于带skip connection的encoder-decoder。

但是t是怎么混进U-Net的呢——答案是把t看成一个position信息。

在SD中使用的是最传统的sinusoidal encoding（也叫Absolute Position Encoding），来自17年的万恶之源Attention is all you need\[4\]。它把 t 编码成一个向量：前一半维度是 $\cos(\omega_k\cdot t)$，后一半是 $\sin(\omega_k\cdot t)$，其中频率  $\omega_k=\theta^{-2k/d}$（$d$ 是 embedding 的总长度，$k$ 从 0 取到 $d/2$，max period $\theta$ 一般取 10000，这也是 SD 中的取值）。

使用三角函数的目的是使得t平移dt对encoding的变换（aka "relative position"）是线性且独立于t的：

$$
\begin{bmatrix}
\cos(\omega_k \cdot \phi) & \sin(\omega_k \cdot \phi) \\
-\sin(\omega_k \cdot \phi) & \cos(\omega_k \cdot \phi)
\end{bmatrix}
\begin{bmatrix}
\sin(\omega_k \cdot t) \\
\cos(\omega_k \cdot t)
\end{bmatrix}
=
\begin{bmatrix}
\sin(\omega_k \cdot (t + \phi)) \\
\cos(\omega_k \cdot (t + \phi))
\end{bmatrix}
$$

> 原文的说法是We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions（section 3.5）  

> 注：虽然在NLP里的本意是为了相对位置，但对于diffusion其实也是很有道理的——回忆一下 $\boldsymbol{\epsilon}_\theta(x, t)$ 其实是尝试approximate噪声 $\boldsymbol{\epsilon}$ ，而噪声是Wiener过程，它也是Markov的，两个噪声的差只和时间差有关。

而使用geometric progression的$\omega_k$，乍一看不太自然——因为大家熟悉的Fourier变换需要线性的 $\omega_k$ ，搞成geometric就不能碰瓷Fourier了。
但其实这个选择也有些道理：类比一下，如果采样int的话，一个自然的想法是采样n进制上的各位，位的底是指数增长的，而位上的值也是一个周期函数。那如果采样float呢？我们这个 $\omega_k$ 实际上可以看成 $\theta^{2/d}$ 进制的编码（

> 虽然这个思路最早是在reddit上发现的，但后来我发现[苏老师](https://kexue.fm/tag/%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81/)也想到了一样的解释（

得到position encoding之后，过两层MLP就得到t的embedding。

> 注意embedding和encoding的区别，encoding是人为选择的，而embedding是mlp学出来的（

然后就把t的embedding插入回去。插入方法非常的暴力，unet每一层input先conv2d(3,1,1)到hidden layer（dim和t\_emb一样），然后**加上**t\_emb，最后conv2d(3,1,1)回去。其他的部分就和传统unet一样了。

为什么是加呢？这个也是NLP那边传过来的操作，主要也是为了方便model理解relative position（注意到对linear layer来说concat和加其实是一样的）。

那么可不可以换个别的方法呢？有的有的，时代已经变了，现在人人都用RoPE了。

### RoPE\[5\]

之前我们提到了relative position。

这个性质其实可以更进一步的formulate为：我们希望对于 $\boldsymbol{QK}^T$ 中的每一项， $(\boldsymbol{QK}^T)_{mn}=\boldsymbol{q}_m^T\boldsymbol{k}_n$ 的值只和原信息x1/x2，以及位置差 $m-n$ 有关（注意\[5\]中的符号和上一张图mn的顺序是反的）。

我们记给x1/x2添加位置信息的过程为 $\boldsymbol{q}_m=\boldsymbol{f}_q(\boldsymbol{x}_m, m),\boldsymbol{k}_n=\boldsymbol{f}_k(\boldsymbol{x}_n, n)$ ，其中m和n是x1和x2对应位置的值，那么我们想要的其实是

$$
\langle\boldsymbol{f}_q(\boldsymbol{x}_m, m),\boldsymbol{f}_k(\boldsymbol{x}_n, n)\rangle=g(\boldsymbol{x}_m, \boldsymbol{x}_n, m-n)
$$

加性的position encoding当然也可以做到，但是需要一系列操作来显式的提供相对位置信息（最老的传统sinusodial+multi-head attention其实是不行的）。

但实际上这个式子有一个更漂亮的基于复数的解法：

![](/assets/img/posts/diffusion-2-ldm/v2-522097d5f20311a2fcabba624a870888.jpg)

这里R就是Rotary Position Encoding (RoPE)。

> 和传统position encoding是加在x上不同，R是乘上去的（旋转矩阵），这也就是RoPE标题中Rotary的含义。

好消息是RoPE是一个绝对位置编码，却能传递相对位置信息。

更好的消息是这个矩阵十分稀疏，所以乘法非常便宜：

$$
Rx=\begin{pmatrix}x_0 \\ x_1 \\ x_2 \\ x_3 \\ \vdots \\ x_{d-2} \\ x_{d-1}  \end{pmatrix}\otimes\begin{pmatrix}\cos m\theta_0 \\ \cos m\theta_0 \\ \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \cos m\theta_{d/2-1}  \end{pmatrix} + \begin{pmatrix}-x_1 \\ x_0 \\ -x_3 \\ x_2 \\ \vdots \\ -x_{d-1} \\ x_{d-2}  \end{pmatrix}\otimes\begin{pmatrix}\sin m\theta_0 \\ \sin m\theta_0 \\ \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2-1} \\ \sin m\theta_{d/2-1}  \end{pmatrix}
$$

## Conditioned的情况： $\boldsymbol{\epsilon}_\theta(x, t, c)$

首先c作为condition是肯定没法塞进unet的，我们假设已经有一个encoder（使用原文的符号，记作 $\tau$ ）把它变成一个latent code了。

然后我们使用cross attention把unet中的一些层（在源代码中是attention\_levels）的值hijack出来，和condition fuse起来然后再喂回unet。

至于什么是cross attention...

![](/assets/img/posts/diffusion-2-ldm/v2-d2c3478130a40347bb207aa7d3d04ccf.jpg)

这图画的真是好。特别的，如果x1=x2就变成self-attention。图片来源是[这里](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html#:~:text=In%20cross%2Dattention%2C%20we%20mix,decoder%20part%20on%20the%20right)。

  

## 参考资料

\[1\] [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

\[2\] [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841)

\[3\] [Vector-quantized Image Modeling with Improved VQGAN](https://arxiv.org/abs/2110.04627)

\[4\] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

\[5\] [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
