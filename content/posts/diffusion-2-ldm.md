---
title: "家用扩散模型速成 (2)：Latent Diffusion Model"
date: "2023-08-13 16:22"
slug: diffusion-2-ldm
order: 2
original_url: "https://zhuanlan.zhihu.com/p/645326315"
summary: "前文导航 康拉德：家用扩散模型速成 (1)：SDE从入门到弃疗 康拉德：家用扩散模型速成 (1.5)：从SDE到ODE 康拉德：家用扩散模型速成 (1.7)：2022年的采样算法在之前的三篇中，我们大致介绍了扩散模型背后的数学原理…"
source: 知乎专栏
---
## 前文导航

[康拉德：家用扩散模型速成 (1)：SDE从入门到弃疗](https://zhuanlan.zhihu.com/p/641768442)

[康拉德：家用扩散模型速成 (1.5)：从SDE到ODE](https://zhuanlan.zhihu.com/p/643961621)

[康拉德：家用扩散模型速成 (1.7)：2022年的采样算法](https://zhuanlan.zhihu.com/p/645971110)

在之前的三篇中，我们大致介绍了扩散模型背后的数学原理以及如何从一个训完的扩散模型中采样。

但是，还有一个最重要的问题没有解答：那我怎么估算 $\bm{\epsilon}_\theta(x, t)$ 呢？

有请本文的主角，generative model第一次火出圈的LDM \[1\] （aka Stable Diffusion 1.0）

> 鸽子虽然多少懂一些数学，但鸽子根本不懂深度学习（尤其是NLP），所以本文很可能比前几篇更不可靠。  
> 本文假设读者和鸽子一样认识上古模型（比如VAE或者U-Net），但对于五年内的新论文完全不熟悉（

// 有一说一，当年整个新奥尔良大街小巷的餐馆里，漫山遍野都是边恰生蚝边讨论LDM的...可惜鸽子那时候只是个彩笔咸鱼，光顾着到处加原神联机好友了，没有好好用功读书，悔之晚矣（所以宵宫做错了什么，为什么米忽悠要这么对她）

![](/assets/img/posts/diffusion-2-ldm/v2-f4ec724c0a464a5b8086e6eee04a1023.jpg)

## TL; DR

![](/assets/img/posts/diffusion-2-ldm/v2-489a1288e20b32a1b901e25a12c53317.jpg)

这篇文章的思路非常简单：

-   首先train一个VAE把high-res的图片压到一个low-res的latent space（注意被压成feature map了还是2D，和许多VAE直接压成1D latent code不同）。backbone使用了VQGAN。

-   小贴士：虽然作者提了一下loss可以用传统VAE的KL loss，或者用VQGAN的quantization layer，但不管选哪个regularization，backbone结构都是一样的——因为VQGAN虽然名义上是GAN，但其实还是有encoder-decoder，不要像鸽子一样顾名思义(

-   然后就可以在latent space上做diffusion了，因此我们可以通过cross attention把各种condition fuse进来，从而让各种task都复用同一个backbone。

-   例如，如果是从prompt出图，就把encoder从图的encoder换成文字的encoder。具体可以看同属SD系列 的[img2img源码](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py)和[text2img源码](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)。

## 从image space到latent space：VQGAN系列

VQGAN实际上也是大名鼎鼎的老牌模型了，知乎上有不少介绍可以搜到，大佬们肯定讲的比我好。

以下是在今天之前完全不懂transformer的鸽子速成VQGAN的心路历程，仅供参考。

> 附：鸽子胡扯之Transformer一分钟从入门到放弃  
> \- 首先大家都知道CNN最重要的inductive bias是卷积这个kernel有locality。  
> \- 然后attention你盯着这个式子看一会儿，会意识到实际上它是对于query Q中的每个元素根据key K，算一个softmax对value V加权平均——所以才叫QKV——然后你就意识到attention某种意义上是一种fc，它没有locality，inductive bias变小了。  
> \- 如果把CNN和transformer结合起来，自然可以又得益于locality（比如说得到feature map），又能得到long-range（或曰global）information。  
> \- 那可能有人要说了，那我全上mlp，inductive bias不是更小——正确的，但是代价是inductive bias越小的东西参数就越多，模型就越难train（大概也是“天下没有免费午餐”的一部分）  
> \- 特别的，如果时空穿越回十年之前，当年大家image classification的时候把图片卷成feature map然后拍扁过fc（比如VGG），这就是CNN和mlp结合起来，某种意义上是CNN和transformer结合的前身（  
> \- 如果未来某一天老黄能发明特别牛逼的显卡，每秒train 1919810个batch，每个batch有114514个sample，大概以后就mlp is all we need了（

### Vector Quantization (VQ) for latent space

-   首先，如果只有encoder-decoder，latent code直接从decoder出来，那就是个经典autoencoder（甚至没有V）。
-   VQ layer做了什么呢？它把latent code z（严格地说是输入x经过encoder出的latent code z\_e(x)）给最近邻到codebook里最接近的code z\_q。

-   对比VAE，传统VAE可以看成，对posterior $p(z|x) = \mathcal{N}(z; \mu, \sigma^2)$ 优化ELBO。
-   而VQVAE可以看成，对posterior $p(z=z_k|x) :=\bm{1}_{k=\arg \min_j\|z_e(x)-e_j\|}$ 优化ELBO。

-   所以某种意义上它其实就不variational了。

-   因此，ELBO的解析式中，reconstruction项是一致的，但代替KL regularization的是VQ regularization $\|z_q-z\|_2^2$ 。
-   注意这个loss本身不可导（因为argmin没有导数），所以需要一个trick叫stop-gradient (字面意思，就是让sg(x)=x, d/dx sg(x)=0)，真正的loss term是 $\|z_q-\text{sg}(z)\|_2^2+\beta\|\text{sg}(z_q)-z\|_2^2$ 。

-   其中参数 $\beta$ 通常取0.25。详见[这里](https://kexue.fm/archives/6760#%E7%BB%B4%E6%8A%A4%E7%BC%96%E7%A0%81%E8%A1%A8)的介绍。

-   除了直接用这个MSE loss来update codebook之外，也可以用Exponential Moving Averages（EMA）来update codebook（相当于加了个momentum），详见VQVAE Appendix A。

-   鸽子猜想这就是Stable Diffusion里EMA-VAE和MSE-VAE的来源（？）

-   那这么绕了一圈，换了一个discrete posterior，好处是什么呢？——好处是我们**tokenize**了latent space feature map!

-   然后我们就可以用它干各种tokenize之后才能做的事情，比如说请各种NLP大法上身（

### VQGAN\[2\]

![](/assets/img/posts/diffusion-2-ldm/v2-2d5c4ad5250d771edd29d71ef4f4275c.jpg)

如图，这里作者们就请了GPT2的transformer上身：为了生成一张图，我们先用transformer学一个自编码模型autoregressive model——用人话说就是先采样第一个token，然后从第一个token开始一步一步根据概率采样其他token（也就是figure 2里这个连乘符号）——然后把采样完得到的latent code过decoder，是不是就成功生成了一张图。这就是VQGAN。

-   token顺序当然会影响结果，但实际上只要从左上开始按顺序一路采样就行。事实上作者们在附录里作了一番实验，结果惊奇的发现顺着infer效果最好（而不是那些特意设计的采样顺序），感觉非常神秘（
-   transformer的长度是有限制的，但CNN Encoder-Decoder不能层数太多，不然train vq layer的时候reconstruction不出来。因此作者们使用一个sliding attention window让某一个patch的token概率只和邻近的patch有关。

### ViT-VQGAN\[3\]

Google做的改进版VQGAN。改进有两项：

-   首先有请爆火的ViT来充当encoder和decoder。
-   其次是对vq层codebook的改进，包括两条：

-   normalized code，这样code到code的l2 distance就可以被解释为cosine similarity。
-   传统VQGAN一直受到codebook usage不足的困扰（大多数code根本没人用），作者提出只要简单地把高维的code project到低维（比如32维），然后在低维做最近邻，就能有效解决usage的问题。

-   这也非常合理，想象一下一个1024维的球面，球面上随便点两个vector几乎肯定是正交的，但在32d上就好多了。

## U-Net Denoiser

> 一个比鸽子更好的介绍在这里，是SD源码的逐行解析：[U-Net for Stable Diffusion](https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html)

## Unconditioned的情况： $\bm{\epsilon}_\theta(x, t)$

U-Net大家都十分熟悉了，四舍五入约等于带skip connection的encoder-decoder，所以能吃一个图x当然是很容易理解的，鸽子作为落后时代的老古董自然是秒懂。

但是t是怎么混进unet的呢——答案是把t看成一个position信息，然后就可以继续请各种NLP大法上身了（

### Embedding of t

在SD中使用的是最传统的sinusoidal encoding，来自17年的万恶之源Attention is all you need\[4\]：

将t映射为一半 $\cos (\omega_k\cdot t)$ 一半 $\sin (\omega_k\cdot t)$ ，其中 $\omega_k = \text{max_period}^{-2k/d}$ ，d是embedding的总长度，k从0到d/2，max\_period一般取10000（这也是SD中的取值）。

> 这种设置也被叫做Absolute Position Encoding。

使用三角函数的目的是使得t平移dt对encoding的变换（aka "relative position"）是线性且独立于t的：

$$
\begin{bmatrix}         \cos(\omega_k .\phi) & \sin(\omega_k .\phi) \\         - \sin(\omega_k . \phi) & \cos(\omega_k .\phi)     \end{bmatrix} .\begin{bmatrix} 	    \sin(\omega_k . t) \\ 	    \cos(\omega_k . t) 	\end{bmatrix} = \begin{bmatrix} 	    \sin(\omega_k . (t + \phi)) \\ 	    \cos(\omega_k . (t + \phi)) 	\end{bmatrix}
$$

> 原文的说法是We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions（section 3.5）  
> 鸽子注：虽然在NLP里的本意是为了相对位置，但对于diffusion其实也是很有道理的——回忆一下 $\bm{\epsilon}_\theta(x, t)$ 其实是尝试approximate噪声 $\bm{\epsilon}$ ，而噪声是Wiener过程，它也是Markov的，两个噪声的差只和时间差有关。

而使用geometric progression来选择 $\omega_k$ ，鸽子乍一看感觉有些神秘——因为大家熟悉的Fourier变换需要线性的 $\omega_k$ ，搞成geometric就不能碰瓷Fourier了 。但其实这个选择也有些道理：类比一下，如果采样int的话，一个自然的想法是采样n进制上的各位，位的底是指数增长的，而位上的值也是一个周期函数。那如果采样float呢？我们这个 $\omega_k$ 实际上可以看成 $10000^{2/d}$ 进制的编码（

> 虽然这个思路最早是在reddit上发现的，但后来我发现[苏老师](https://kexue.fm/tag/%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81/)也想到了一样的解释（

得到position encoding之后，过两层linear就得到t的embedding。

> 注意这个词和encoding的区别，encoding不是学出来的而是人为选择的，而embedding是学出来的（

### 把t\_emb插入u-net

插入方法非常的暴力，unet每一层input先conv2d(3,1,1)到hidden layer（dim和t\_emb一样），然后**加上**t\_emb，最后conv2d(3,1,1)回去。其他的部分就和传统unet一样了。

为什么是加呢？这个也是NLP那边传过来的操作，主要也是为了方便model理解relative position。

那么可不可以换个别的方法呢？乘法也是可以的，见下文“题外话”。至于concat，实际上concat也是一种加法...

## Conditioned的情况： $\bm{\epsilon}_\theta(x, t, c)$

首先c作为condition是肯定没法塞进unet的，我们假设已经有一个encoder（使用原文的符号，记作 $\tau$ ）把它变成一个latent code了。

然后我们使用cross-attention把unet中的一些层（在源代码中是attention\_levels）的值hijack出来，和condition fuse起来然后再喂回unet。

然后...就没了，LDM的思路就是这么简洁易懂（

### 一分钟速成Cross-attention

<figure>
  <img src="/assets/img/posts/diffusion-2-ldm/v2-d2c3478130a40347bb207aa7d3d04ccf.jpg" alt="这图画的真是好（特别的，如果x1=x2就变成self-attention）">
  <figcaption>这图画的真是好（特别的，如果x1=x2就变成self-attention）</figcaption>
</figure>

（图片来源是[这里](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html#:~:text=In%20cross%2Dattention%2C%20we%20mix,decoder%20part%20on%20the%20right)）

## 题外话：RoPE\[5\]

之前我们提到了relative position。

这个性质其实可以更进一步的formulate为：我们希望对于 $\bm{QK}^T$ 中的每一项， $(\bm{QK}^T)_{mn}=\bm{q}_m^T\bm{k}_n$ 的值只和原信息x1/x2，以及位置差 $m-n$ 有关（注意\[5\]中的符号和上一张图mn的顺序是反的）。

我们记给x1/x2添加位置信息的过程为 $\bm{q}_m=\bm{f}_q(\bm{x}_m, m),\bm{k}_n=\bm{f}_k(\bm{x}_n, n)$ ，其中m和n是x1和x2对应位置的值，那么我们想要的其实是

$$
\langle\bm{f}_q(\bm{x}_m, m),\bm{f}_k(\bm{x}_n, n)\rangle=g(\bm{x}_m, \bm{x}_n, m-n)
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

  

## 参考资料

\[1\] [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

\[2\] [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841)

\[3\] [Vector-quantized Image Modeling with Improved VQGAN](https://arxiv.org/abs/2110.04627)

\[4\] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

\[5\] [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
