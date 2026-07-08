---
title: "家用扩散模型 (1)：SDE从入门到弃疗"
date: "2023-07-13 00:05"
slug: diffusion-1-sde
original_url: "https://zhuanlan.zhihu.com/p/641768442"
summary: "论没学过随机过程，也不想看一万个离散概率求和的人应该如何速成扩散模型数学。"
source: 知乎专栏
---
## （2026年的）前言

（2023年）入行diffusion的人在点开开山大作DDPM的第一瞬间，会发现里面是一大堆复杂的离散概率来回求和，动不动一整屏幕公式，看起来一个头两个大。

然而，经过Mizore的研究，发现完全有办法在不学这一大堆离散玩意的情况下搞懂diffusion，因为在连续的语境下其实diffusion还挺优美的——这就是宋飏老师的传奇大作[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)（ICLR '21），也是“家用扩散模型“这个系列的最初动机。
  
以下所有过程都非常不严谨——严谨的推导和证明请看参考资料。需要稍微还记得一点工科数学，包括微积分1，微积分2，线性代数和概率论。

如果想要啃离散大部头，热情推荐[苏剑林老师的中文blog](https://www.kexue.fm/tag/%E6%89%A9%E6%95%A3)。喜欢看英文版离散方法的可以看[翁荔老师的blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)。

### SDE的动机

一个传统的，没有噪声的Ordinary Differential Equation（ODE）形式上长这样：

$$d\boldsymbol{X}(t) = \boldsymbol{\mu}(\boldsymbol{X}, t)dt$$

有边际条件之后，两头对t积分我们就得到了通解，所以微分形式和积分形式可以相互转换。

假设我们要加入一个随机噪声，形式上姑且写作$\boldsymbol{\xi}(t)=\frac{d}{dt}\boldsymbol{W}(t)$ ，那么我们可以依样画葫芦，得到一个Stochastic Differential Equation (SDE)：

$$d\boldsymbol{X}(t) = \boldsymbol{\mu}(\boldsymbol{X}, t)dt + \boldsymbol{\sigma}(\boldsymbol{X}, t)d\boldsymbol{W}$$

有了SDE+边际条件之后，我们两头对t积分就能得到通解——理论上说是这样，但这里有一些问题需要解决：

-   怎样定义噪声dW——在这里我们将会选择Wiener过程（一个更大名鼎鼎的名字是“布朗运动”）
-   怎样定义噪声/随机变量的积分$\int_0^{T} \boldsymbol{\sigma}d\boldsymbol{W}$ ？
-   “这玩意存在/唯一吗”也是个重要问题，但本文一点也不数学，所以这里就不管了。

### Wiener过程 $W(t)$

首先考虑1D的情况。1D的 $W(t)$ 满足如下性质：

-   $W(0)=0$
-   如果 $t \ge s \ge 0$，那么 $W(t)-W(s)$ 是一个高斯分布 $N(0, t-s)$。这里的动机近似的来自大数定律。
-   对一个时间序列，$W(t_1), W(t_2-t_1), \dots$ 每一项都是相互独立的——总之这个过程是马尔可夫的。

推论：

- $E(W(t))=0,\quad E(W^2(t)) = t$
- $E(W(t)W(s))=\min(s, t)$

事实上 $W$ 是不可导的，但我们近似的还是有 $E(\xi(t)\xi(s)) = \delta_0(s-t)$ 。

高维的情况类似，总归意思已经传达到了。

### Itô积分 $\int_0^{T} \boldsymbol{G}d\boldsymbol{W}$

和经典的黎曼积分很像，大致上就是把G拆成很多段，然后求和：

$$
\int_0^{T} GdW=\sum_k G_k(W(t_{k+1})-W(t_k))
$$

推论：

-   $E(\int_{0}^{T}GdW)=0$
-   $E(\int_{0}^{T}GdW\int_{0}^{T}HdW)=E(\int_{0}^{T}GHdt)$

把G和W推广到高维的时候就是简单的做一个内积，类似的有：

-   $E(\int_{0}^{T}\boldsymbol{G}d\boldsymbol{W})=0$
-   $E(\|\int_{0}^{T}\boldsymbol{G}d\boldsymbol{W}\|^2)=E(\int_{0}^{T}\|\boldsymbol{G}\|^2dt)$

### 伊藤引理 Itô's Lemma

$d\boldsymbol{X}(t) = \boldsymbol{\mu}(\boldsymbol{X}, t)dt + \boldsymbol{\sigma}(\boldsymbol{X}, t)d\boldsymbol{W}$ 这个式子非常好，因为你对X(t)算mean和var都非常好算。

但是很多时候式子是关于另一个数u(X, t)的——也就是说我们需要一个链式法则。

-   Itô product rule: d(X1X2) = X1dX2 + X2dX1 + dX1dX2。
-   一般的，对SDE $d\boldsymbol{X}= \boldsymbol{b}dt + \boldsymbol{G}d\boldsymbol{W}$ ，symbolically的有：

$$
d(u(\boldsymbol{X}, t)) = \frac{\partial u}{\partial t} dt + \sum_i \frac{\partial u}{\partial x_i}  dX^i + \frac{1}{2} \sum_{i, j} \frac{\partial^2 u}{\partial x_i\partial x_j}  dX^idX^j +...
$$

然后舍弃高阶无穷小（(dW)^2和dt是一个量级，所以比dt更高阶的无穷小都可以扔了）：

$$
(dt)^2=0, dtdW^k=0, dW^kdW^l=\delta_{kl}dt
$$

得到的式子就是Itô's Lemma：

$$
du(\boldsymbol{X}, t) = (\frac{\partial u}{\partial t} + \boldsymbol{b}\cdot \nabla u + \frac{1}{2}\boldsymbol{G}\boldsymbol{G}^T:\nabla^2u)dt + \nabla u\cdot \boldsymbol{G}d\boldsymbol{W}
$$

其中

-   $A:B=\sum_{i, j}A_{ij}B_{ij} = Tr(A^TB)$ 是矩阵内积
-   $\nabla$ 是对x的各个维度的梯度
-   $\nabla^2$ 是对x的各个维度的Hessian（注意不是Laplacian）

### Kolmogorov Backward Equation和Fokker-Planck Equation

// Kolmogorov当年其实同时提出来了两个式子，backward的和forward的，backward的是已知时间t算之前的时间s，forward的是已知时间t算之后的时间s。后来大家发现forward版的版权已经在物理学里被F-P抢注了，所以就只有backward一说。

对SDE $d\boldsymbol{X}(t) = \boldsymbol{b}(\boldsymbol{X}, t)dt + \boldsymbol{\sigma}(\boldsymbol{X}, t)d\boldsymbol{W}$ ：

我们记infinitesimal generator L，使得 $(\mathcal{L} f)(x, t) = \boldsymbol{b}(x, t)\cdot \nabla f(x)+ \boldsymbol{a}(x, t):\nabla^2f(x)$ ，其中 $\boldsymbol{a}= \frac{1}{2}\boldsymbol{\sigma}\boldsymbol{\sigma}^T$ 。一般来说这里的b叫drift（也记作 $\mu$ ），而a叫diffusion coefficient。

首先考虑backward的情况。将伊藤引理的式子两边从时刻s到时刻t积分，然后两边取在X\_s=y时刻的期望 $E^{y,s}[\cdot] = E[\cdot | \boldsymbol{X}_s=y]$ ，得到：

$$
E^{y, s}[u(\boldsymbol{X}_t, t)]-u(y, s) = \int_s^t E^{y, s}[(\partial_r u+ \mathcal{L}u)(\boldsymbol{X}_r, r)]dr
$$

因为是期望，所以dW项就消失了。

现在考虑 $u(y, s):=E^{y, s}[f(\boldsymbol{X}_t)]=E^{y, s}[u(\boldsymbol{X}_t, t)]$ (这是因为此时由定义，u(X\_t, t)恒等于f(X\_t))——注意此时f是任意函数。两边同取 $\lim_{t \rightarrow s}{\frac{1}{t-s}}$ (其中t>s)，LHS收敛到0，我们得到

-   $\partial_s u(y, s) + \mathcal{L}u(y, s) = 0, s<t, u(y, t)= f(y)$ （1）

记 $p(x, t|y, s)$ 为转移概率，有 $u(y, s)=\int f(x)p(t, x|s, y)dx$ ，代入上式，由于f是任意函数，所以我们就消掉了f的积分，从而得到：

-   $\partial_s p + \mathcal{L}p = 0, s<t, p(x, t|y, t)=\delta(x-y)$ （2）

(1)和(2)即为Kolmogorov Backward Equation。

然后考虑forward的情况。我们取 $\mathcal{L}$ 的伴随 $\mathcal{L}^*: \langle\mathcal{L}f, g\rangle=\langle f, \mathcal{L}^*g\rangle$

通过一系列分部积分（详询参考资料），可以得到 $(\mathcal{L}^* g)(x, t) = -\nabla \cdot (\boldsymbol{b}(x, t) g(x)) + \nabla^2: (\boldsymbol{a}(x, t) g(x))$

（注意这里g在括号内，和f不同）

然后用类似的方法，就可以得到正向的Kolmogorov Forward Equation或Fokker-Planck Equation:

-   $\partial_t \rho(x, t) = \mathcal{L}^*\rho (x, t), t>0, \rho(x, 0)=\rho_0(x)$ （3），这里\\rho是t时刻X\_t的概率密度。
-   $\partial_t p = \mathcal{L}^*p, t>0, p(x, s|y,s) =\delta(x-y)$ （4），这里p是p(x, t|y, s)

### Reverse-time diffusion equation models

我们把KBE（2）倒着写（也就是取反向的s>t，对t求导数）

$$
-\partial_t p(y, s|x, t) = \boldsymbol{b}(x, t)\cdot \nabla p(y, s|x, t)+ \boldsymbol{a}(x, t):\nabla^2 p(y, s|x, t)
$$

然后KFE（3）告诉我们 $-\partial_t p(x, t) = \nabla \cdot (\boldsymbol{b}(x, t) p(x, t)) - \nabla^2: (\boldsymbol{a}(x, t) p(x, t))$，t>0 。

又， $p(x, t, y, s) = p(y, s|x, t)p(x, t)$ ，这三个式子联立，只留下p(x, t, y, s)，我们得到：

$-\partial_t p(x, t, y, s) = \nabla \cdot (\overline{\boldsymbol{b}}(x, t) p(x, t, y, s)) - \nabla^2 : (\boldsymbol{a}(x, t)p(x, t, y, s))$ （5）

其中 $\overline{\boldsymbol{b}^i}=\boldsymbol{b}^i - \frac{2}{p(x, t)}\nabla\cdot (p(x, t)\boldsymbol{a}^i)$ ，这里 $\boldsymbol{a}^i$ 是a的第i行， $\boldsymbol{b}^i$ 是b的第i个元素。

> 大致的思路：记 $p_1=p(y, s|x, t), p_2=p(x, t)$  
> 由梯度的运算法则，有 $\nabla \cdot(p_1p_2\boldsymbol{b})=p_2\boldsymbol{b}\cdot \nabla p_1 + p_1\nabla \cdot(p_2\boldsymbol{b})$  
> 按矩阵内积的定义展开，有  
> $\nabla^2: (p_1p_2\boldsymbol{a}) = p_2\boldsymbol{a}:\nabla^2 p_1 + 2\sum_i \frac{\partial}{\partial_{x_i}}(p_1\nabla\cdot (p_2\boldsymbol{a}^i)) - p_1\nabla^2:(p_2\boldsymbol{a})$  
> 所以  
> $-\partial_t(p_1p_2) = -p_2\partial_tp_1-p_1\partial_tp_2 = p_2\boldsymbol{b}\cdot \nabla p_1 + p_2\boldsymbol{a}:\nabla^2p_1 + p_1\nabla \cdot (p_2\boldsymbol{b})-p_1\nabla^2 : (p_2\boldsymbol{a})$  
> 化简一下就得到（5）。

那么好了，（5）是不是看着很眼熟呢？

对啦，（5）长得和（3）形式上一模一样，实际上它就是以下反向过程SDE的KFE：

$$
d\boldsymbol{X}(t) = \overline{\boldsymbol{b}}(\boldsymbol{X}, t)dt + \boldsymbol{\sigma}(\boldsymbol{X}, t)d\overline{\boldsymbol{W}}
$$

这里 $d\overline{\boldsymbol{W}}=d\boldsymbol{W}+\frac{1}{p(\boldsymbol{X_t},t)}\nabla\cdot (p(\boldsymbol{X_t}, t)\boldsymbol{\sigma}^i(\boldsymbol{X_t}, t))dt$

## 你说得对，但是扩散模型是由...

现在我们回到扩散模型。根据原文，正向过程的方差是个和X无关的函数，它的SDE是：

$$
d\boldsymbol{X}(t) = \boldsymbol{f}(\boldsymbol{X}, t)dt +g(t)d\boldsymbol{W}
$$

所以反向过程SDE是：

$$
d\boldsymbol{X}(t) = \overline{\boldsymbol{f}}(\boldsymbol{X}, t)dt + g(t)d\overline{\boldsymbol{W}}
$$

其中 $\overline{\boldsymbol{f}^i}=\boldsymbol{f}^i - \frac{1}{p(\boldsymbol{X}_t, t)}\frac{\partial}{\partial x_i}(p(\boldsymbol{X}_t, t)g^2(t))$

化简一下这个式子，就得到 $\overline{\boldsymbol{f}}=\boldsymbol{f} - \nabla (g^2(t)\log p(\boldsymbol{X}_t, t))$

这就是原文中的3.2 GENERATING SAMPLES BY REVERSING THE SDE。

## 下期预告

-   其实SDE还有很多别的用处（比如Black-Scholes Model，大概做金融的人需要努力学SDE），不过和我们diffusion的关系也不大（
-   其实这里推了老半天，和diffusion基本上没什么关系，感觉作为家用扩散模型的第一篇非常不成功（

## 参考资料

### SDE部分

-   [https://www.cmor-faculty.rice.edu/~cox/stoch/SDE.course.pdf](https://www.cmor-faculty.rice.edu/~cox/stoch/SDE.course.pdf)
-   [https://cims.nyu.edu/~holmes/teaching/asa19/handout\_Lecture10\_2019.pdf](https://cims.nyu.edu/~holmes/teaching/asa19/handout_Lecture10_2019.pdf)
-   [https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck\_equation](https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation)
-   [https://core.ac.uk/download/pdf/82826666.pdf](https://core.ac.uk/download/pdf/82826666.pdf)

