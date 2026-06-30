---
title: "家用扩散模型 (1.7)：2022年的采样算法"
date: "2023-07-30 17:54"
slug: diffusion-1p7-samplers
order: 1.7
series: "家用扩散模型"
original_url: "https://zhuanlan.zhihu.com/p/645971110"
summary: "一些2022年常用的采样算法，包括DPM-solver++和EDM"
source: 知乎专栏
---

## 前言

本文介绍一些2022年常用的采样算法，包括DPM-solver++\[2\]\[3\]和EDM\[1\]，都是2022年的论文。

这里有一个（2022年）Stable Diffusion上常用sampler原理和效果比对的文章：  [Stable Diffusion Samplers: A Comprehensive Guide](https://stable-diffusion-art.com/samplers/)  

## 转移概率：为什么扩散模型SDE里的f(x, t)的选取一般对x是线性的？

在[上一期](/blog/posts/diffusion-1p5-ode.html)中我们提过，为了让 $p_{0 t}\left(\mathbf{x}_t \mid \mathbf{x}_0\right)$ 是Gaussian，我们的f其实一般选择是线性的。

为什么呢？如果我们有一个线性的f：

$$
d\mathbf{x} = f(t)\mathbf{x}dt + g(t)d\mathbf{w}
$$

记（这个符号来自\[1\]）

$$
s(t)=\exp \left(\int_0^t f(\xi) \mathrm{d} \xi\right), \sigma^2(t)=\int_0^t \frac{g(\xi)^2}{s(\xi)^2} \mathrm{~d} \xi
$$

我们有

$$
p_{0 t}\left(\mathbf{x}_t \mid \mathbf{x}_0\right) \sim \mathcal{N}(s(t)\mathbf{x}_0, s^2(t)\sigma^2(t)\mathbf{I})
$$

即服从高斯分布。

> 事实上，这是名为Linear SDE的一大类SDE，这一类SDE都有解析解。  
> 大致的推导思路：注意到 $d(s^{-1}(t))=-f(t)s^{-1}(t)dt$ ，由Ito's product rule我们有  
> $d(s^{-1}(t)x(t))=x(t)d(s^{-1}(t))+ s^{-1}(t)dx(t) + d(s^{-1}(t))dx(t)\\= s^{-1}(t) (dx(t)-f(t)x(t)dt)=s^{-1}(x)g(t)dw$  
> 这里dtdx项因为都是高阶无穷小被消掉了。既然只剩下一项dw，那你两头积分当然立得Gaussian（  
>   
> 详见[这个讲义](http://www.stat.uchicago.edu/~zhongjian/_downloads/d05d9dbaec0b82dcdf7e708bcb7c6735/lec4.pdf)。

## DPM-solver\[2\] / DPM-solver++ \[3\]

以下使用DPM系列的符号：

$$
p_{0 t}\left(\boldsymbol{x}_t \mid \boldsymbol{x}_0\right) \sim \mathcal{N}(\alpha_t\boldsymbol{x}_0, \sigma^2_t\mathbf{I})
$$

翻译成人话： $\boldsymbol{x}_t=\alpha(t)\boldsymbol{x}_0+\sigma(t)\boldsymbol{\epsilon}$ ， $\boldsymbol{\epsilon}$ 是单位高斯噪声。

使用类似DDPM的方式（详见上一期），可以用一个nn $\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t)$ 来从 $\boldsymbol{x}_t$ 预测 $\boldsymbol{x}_0\sim \boldsymbol{x}_\theta:=\frac{1}{\alpha_t}(\boldsymbol{x}_t-\sigma_t\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t))$ ——某种意义上可以认为这个nn在预测噪声 $\boldsymbol{\epsilon}$ 。可以推导出，这种情况下我们有

$$
\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t)=-\sigma_t\nabla_{\boldsymbol{x}}\log q_t(\boldsymbol{x}_t)
$$

（可以记住这个漂亮的结论）

### 既然这个转移概率的closed form对f和g这么丑，我们干脆别用f和g了吧！

我们有

$$
f(t)=\frac{d}{dt}\log\alpha_t, g^2(t)=\frac{d}{dt}\sigma^2_t-2\sigma_t^2\frac{d}{dt}\log\alpha_t
$$

### 那么，Loss呢？

按上一期的式子化简一下，把系数提出去，可以写成

$$
\mathcal{J}_{\mathrm{DSM}}(\boldsymbol{\theta} ; \omega(\cdot)):=\frac{1}{2} \int_0^T \mathbb{E}_{q_0(\boldsymbol{x}_0), q(\boldsymbol{\epsilon})}\left[\omega(t)\left\|\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t) - \boldsymbol{\epsilon}\right\|_2^2\right] \mathrm{d} t
$$

### 对应的ODE，及其解法

按上一期的式子化简一下，有

$$
\frac{d}{dt}\boldsymbol{x}_t = f(t)\boldsymbol{x}_t+\frac{g^2(t)}{2\sigma_t}\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t, t)
$$

把f和g用 $\alpha$ 和 $\sigma$ 代入，经过一通操作（详见\[2\]），我们得到一个精确的表达式：

$$
\boldsymbol{x}_{t}=\frac{\alpha_{t}}{\alpha_{s}}\boldsymbol{x}_{s}-\alpha_{t}\int_{\lambda_{s}}^{\lambda_{t}}e^{-\lambda}\hat{\boldsymbol{\epsilon}}_{\theta}(\hat{\boldsymbol{x}}_{\lambda},\lambda)d\lambda, t\in [0, s]
$$

（式3.5，\[2\] / 式7，\[3\]）

这个式子看起来可能有点怪，但其实可以写成这样： 

$$
\frac{\boldsymbol{x}_{s}}{\alpha_{s}}-\frac{\boldsymbol{x}_{t}}{\alpha_t}=\int_{\lambda_{s}}^{\lambda_{t}}e^{-\lambda}\hat{\boldsymbol{\epsilon}}_{\theta}(\hat{\boldsymbol{x}}_{\lambda},\lambda)d\lambda
$$

是不是就看起来像一个ODE的差分形式了。

其中，$\lambda_t:=\log\frac{\alpha_t}{\sigma_t}$ 是信噪比（SNR）的对数（如果经常读diffusion论文，会发现SNR也是一个经常出现的名词），$\hat{(\cdot)}$ 是将对t的函数变换为对 $\lambda$ 的函数（ $\lambda$ 是严格单调递减的，所以存在一个反函数）。

如果t和s足够接近，我们可以把这项积分项泰勒展开（对 $\lambda$ 求n阶导），从而求出一个近似值。例如，一阶近似（记作DPM-solver-1）是

$$
\tilde{\boldsymbol{x}}_{t}=\frac{\alpha_{t}}{\alpha_{s}}\tilde{\boldsymbol{x}}_{s}-\alpha_{t} (\frac{\sigma_s}{\alpha_s}-\frac{\sigma_t}{\alpha_t}) \boldsymbol{\epsilon}_{\theta}(\tilde{\boldsymbol{x}}_s,s) + \mathcal{O}((\lambda_t-\lambda_s)^2)
$$

> 这个实际上在ODE里叫[Euler法](https://en.wikipedia.org/wiki/Euler_method)。

通过类似的方式可以得到二阶近似，记作DPM-solver-2。

但\[3\]中作者们发现直接二阶展开上式有些数值不稳定，所以他们变换了一下形式（式8, \[3\])：

$$
\boldsymbol{x}_{t}=\frac{\sigma_{t}}{\sigma_{s}}\boldsymbol{x}_{s}+\sigma_{t}\int_{\lambda_{s}}^{\lambda_{t}}e^{\lambda}\hat{\boldsymbol{x}}_{\theta}(\hat{\boldsymbol{x}}_{\lambda},\lambda)d\lambda, t\in [0, s]
$$

注意这里用了 $\boldsymbol{x}_\theta:=\frac{1}{\alpha_t}(\boldsymbol{x}_t-\sigma_t\boldsymbol{\epsilon}_\theta)$，以及$\alpha$ 换成了 $\sigma$。

这个式子的一阶展开依然是一样的（DDIM），但二阶展开有一些区别，记作DPM++2。

\[3\]中提出了2S和2M两种二阶方法（S代表single step，M代表multi step），细节总归不本质，但总之大家都用DPM++ 2M就完了。

当然，既然有ODE版，肯定也有SDE（Ancestral Sampling）版，详见\[3\]。

### 和DDIM的联系

作者指出，实际上DDIM的SDE形式就是SDE-DPM-Solver++1，而ODE形式就是DPM-Solver-1。

> 这也和Stable Diffusion的官方指南相吻合——他们推荐的SDE sampler正是DDIM或DPM++ SDE(2M)，而ODE sampler的推荐是DPM++ 2M，也就是说，不管怎么样，他们总归推荐的正是 $\boldsymbol{x}_{t}=\frac{\sigma_{t}}{\sigma_{s}}\boldsymbol{x}_{s}+\sigma_{t}\int_{\lambda_{s}}^{\lambda_{t}}e^{\lambda}\hat{\boldsymbol{x}}_{\theta}(\hat{\boldsymbol{x}}_{\lambda},\lambda)d\lambda$ 的某种一阶或二阶近似。

如果一时半会儿看不出来两者的联系，可以注意到DPM-Solver-1

$$
\tilde{\boldsymbol{x}}_{t}=\frac{\alpha_{t}}{\alpha_{s}}\tilde{\boldsymbol{x}}_{s}-\alpha_{t} (\frac{\sigma_s}{\alpha_s}-\frac{\sigma_t}{\alpha_t}) \boldsymbol{\epsilon}_{\theta}(\tilde{\boldsymbol{x}}_s,s)
$$

事实上可以被写成 

$$
\tilde{\boldsymbol{x}}_{t}=\alpha_{t}\boldsymbol{x}_{\theta}(\tilde{\boldsymbol{x}}_s,s) + \sigma_{t}  \boldsymbol{\epsilon}_{\theta}(\tilde{\boldsymbol{x}}_s,s)
$$

这样一看就看的很明显，DDIM是Final image + Direction + Noise的组合，显然第一项是final image，第二项是direction（别忘了$\sigma_{t}\boldsymbol{\epsilon}_{\theta}$其实是score function）。

然后Ancestral版是 

$$
\tilde{\boldsymbol{x}}_{t}=\alpha_{t}\boldsymbol{x}_{\theta}(\tilde{\boldsymbol{x}}_s,s) + \sqrt{\sigma_{t}^2-\eta^2} \boldsymbol{\epsilon}_{\theta}(\tilde{\boldsymbol{x}}_s,s) +\eta \boldsymbol{z}_{s}
$$

这个 $\eta$ 项是怎么来的呢？回忆一下我们在上一期里是怎么从SDE推出ODE的，少划一些noise过去就立得这个SDE族。

### 虽然但是，DDIM到底比DDPM加速在哪里呢？

按照DDPM的那一套，你正向多少步就得反向多少步。DDIM里全程就不依赖正向过程（可以想象为对那个积分方程泰勒展开，当然中间的断点可以随便取啦），所以反向步数可以取得相当少。

## EDM\[1\]

以下使用EDM的符号：

$$
p_{0 t}\left(\mathbf{x}_t \mid \mathbf{x}_0\right) \sim \mathcal{N}(s(t)\mathbf{x}_0, s^2(t)\sigma^2(t)\mathbf{I})
$$

### 既然这个转移概率的closed form对f和g这么丑，我们干脆别用f和g了吧！（异 曲 同 工）

我们有

$$
f(t)=\dot{s}(t)/s(t), g(t)=s(t)\sqrt{2\dot{\sigma}(t)\sigma(t)}
$$

和DPM对比一下会发现是一样的，只能说log函数真是神奇啊（

### 那么，Loss呢？

一般的，EDM使用一个diffuser来拟合x\_t到x\_0的反向过程：

$$
D(\mathbf{x};\sigma)=c_{\mathrm{skip}}(\sigma)\mathbf{x} + c_{\mathrm{out}}(\sigma)\mathbf{F}_{\theta}(c_{\mathrm{in}}(\sigma)\mathbf{x};c_{\mathrm{noise}}(\sigma))
$$

其中所有的c都是只和 $\sigma(t)$ 有关的函数，某种意义上类似preconditioner。

于是，Denoising Score Matching变为

$$
\begin{aligned}
\mathcal{L}(D_{\theta}) &=\mathbb{E}_{\sigma\sim p_{\mathrm{train}}}\left[\lambda(\sigma)\ \mathbb{E}_{\mathbf{y}\sim p_{\mathrm{data}}}\ \mathbb{E}_{\mathbf{n}\sim N(0,\sigma^{2})}\ \|D_{\theta}(\mathbf{y}+\mathbf{n};\sigma)-\mathbf{y}\|_{2}^{2}\right] \\
&=\mathbb{E}_{\sigma,\mathbf{y},\mathbf{n}}\left[\lambda(\sigma)\ \|D_{\theta}(\mathbf{y}+\mathbf{n};\sigma)-\mathbf{y}\|_{2}^{2}\right] \\
&=\mathbb{E}_{\sigma,\mathbf{y},\mathbf{n}}\left[w(\sigma)\|F_{\theta}\left(c_{\mathrm{in}}(\sigma)(\mathbf{y}+\mathbf{n});c_{\mathrm{noise}}(\sigma)\right)-F_{\mathrm{target}}(\mathbf{y},\mathbf{n};\sigma)\|_{2}^{2}\right]
\end{aligned}
$$

其中 $w(\sigma)=\lambda(\sigma)c_{\mathrm{out}}^2(\sigma)$， $F_{\mathrm{target}}(\mathbf{y},\mathbf{n};\sigma)=\frac{1}{c_{\mathrm{out}}(\sigma)}\mathbf{y}-c_{\mathrm{skip}}(\sigma)(\mathbf{y}+\mathbf{n})$ 。

### 对应的ODE，及其解法

通过一些不太有趣的计算，对应的ODE为

$$
\frac{\mathrm{d}x}{\mathrm{d}t}=\biggl(\frac{\dot{\sigma}(t)}{\sigma(t)}+\frac{\dot{s}(t)}{s(t)}\biggr)x-\frac{\dot{\sigma}(t)s(t)}{\sigma(t)}D_{\theta}\biggl(\frac{x}{s(t)};\sigma(t)\biggr)
$$

然后就可以抓一个二阶方法来解了（Heun）。虽然但是，据Stable Diffusion的介绍，Heun似乎比较慢，所以大概也许不是很推荐（

### 一些常见的扩散模型对应的 $\sigma$ 的取值

![](/assets/img/posts/diffusion-1p7-samplers/v2-28dc0af39cb126ff8d6b4f6829bb714d.jpg)

## 参考资料

-   \[1\] [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
-   \[2\] [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927)
-   \[3\] [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095)
