---
title: "家用现代扩散模型速成 (4)：从Consistency Models到MeanFlow"
date: "2026-07-11 01:00"
slug: modern_diffusion_4
summary: "带forward-divergence味道的轨迹蒸馏，以及这条线如何走到不要teacher的1-NFE from scratch。"
---

## 前言：蒸馏的另一条路

借用rCM[[9]][r9]的taxonomy，可以按照监督样本从哪里来，把few-step distillation近似类比成forward divergence和reverse divergence两种范式：
- 上一章我们讲了reverse divergence：从student自己的on-policy样本出发，拿teacher的score把它往teacher分布上推。上章介绍的例子是VSD和DMD，它们的目标确实来自KL divergence。
- 这一章是forward divergence：在真实数据（以及optional的teacher trajectory sample上）做回归，要把target distribution的每个mode都盖住（mode covering）。

但事实上，本章这些方法的核心trajectory-consistency项没有一个直接写成divergence（甚至有几个压根没有teacher——所以其实并非一种distillation）；不过CTM[[4]][r4]额外加入的DSM和GAN auxiliary losses，确实分别带有KL和f-divergence/IPM的解释。
它们真正的共同点是学习一个finite-time transport operator $\Phi_{t\to s}(x_t)=x_s$ 来代替diffusion本身predict的局部量。特别的，如果在真实数据+teacher上训练，就会看起来像forward divergence。

## Recap：(discrete) Consistency Models (ICML '23)

> 和上章的情况类似，鸽子把之前《家用扩散模型》的专栏refactor了一下，这样读者朋友们就不用回去看老版了（

CM[[1]][r1]同样由宋飏老师发明。先用 $\epsilon\to0$ 的理想化记法来说，它的目标是训练一个函数 $\boldsymbol f_\theta(\boldsymbol x_t,t)\to \boldsymbol x_0, \forall t$，让我们之前在第一章提过，和forward SDE在每个时刻具有相同marginal distribution的那个确定性ODE（学名叫probability flow(PF) ODE）trajectory上的每个点都map回同一个干净样本。

> 但0处numerically不stable（回忆一下，t=0时SNR变成无穷了），因此实现上并非 $\forall t\in [0,T]$，而是需要从一个很小但非零的 $\epsilon$ 开始。

也许这里有读者会问：“形式上，这不就是想train一个denoiser吗？”

是的，但别忘了第一章提过，普通diffusion的x-loss学到的是条件期望 $\mathbb E[\boldsymbol x_0\mid\boldsymbol x_t]$，而不是“这次和 $\boldsymbol x_t$ 配对的那张 $\boldsymbol x_0$”。所以，x-loss不会收敛到0：$t\to0$ 时，$\boldsymbol x_t$ 几乎已经把答案写在脸上，条件期望接近原图；$t\to T$ 时，同一个 $\boldsymbol x_t$ 背后可能对应大量不同的原图，条件期望就会把它们average成一团浆糊（EDM Figure 1b[[7]][r7]）。普通diffusion学到的是这个Bayes denoiser，并不是“把同一条PF-ODE轨迹上的每个点都送回同一个端点”的consistency function。

![](/assets/img/posts/modern-diffusion-4/edm1.png)

> 虽然形式上CM写作一个x-pred的denoiser，但里面的raw network仍然可以沿用v-pred形式的parameterization。例如RF下可以写成 $\boldsymbol f_\theta(\boldsymbol x_t,t)=\boldsymbol x_t-t\boldsymbol F_\theta(\boldsymbol x_t,t)$，这样 $t=0$ 时边界条件自动满足。这里说的是输出形式；经过CM loss训练后，$\boldsymbol F_\theta$ 不再保证等于PF-ODE的瞬时速度。

既然普通x-loss训不出这样的目标函数，那么CM就需要一个新的loss。

在刚才 $\epsilon\to0$ 的理想化记法下，CM保证了 $\boldsymbol f_\theta(\boldsymbol x_0,0) = \boldsymbol x_0$，因此我们只需要保证

$$
\frac{\mathrm{d}}{\mathrm{d}t}\boldsymbol f_\theta(\boldsymbol x_t,t)=\frac{\partial \boldsymbol f_\theta}{\partial t}+\nabla_{\boldsymbol x}\boldsymbol f_\theta\,\frac{\mathrm{d}\boldsymbol x_t}{\mathrm{d}t}=\boldsymbol 0
$$

翻译成离散版，就是同一条轨迹上相邻两个节点的输出应该相等：

$$
\boldsymbol f_\theta(\boldsymbol x_{t_{n+1}},t_{n+1})
\approx
\boldsymbol f_\theta(\boldsymbol x_{t_n},t_n)
$$

这也是consistency一词的来源。

这里还有一个问题，就是对采样的 $\boldsymbol x_{t_{n+1}}=\boldsymbol x_0+t_{n+1} \boldsymbol \epsilon$ ，如何构建 $x_{t_n}$ 。

一个简单的方法是直接不要teacher了，直接取 $\boldsymbol x_{t_n}=\boldsymbol x_0+t_n\boldsymbol\epsilon$，这个叫consistency training（CT）。注意，这两个点连出的straight noising path通常也不是PF-ODE trajectory；它是一个finite-difference surrogate，只有在 $\Delta t\to0$ 时才给出PF velocity的unbiased estimator，有限步长下仍然有bias。
如果真的有一个teacher，固定同一组 $(\boldsymbol x_0,\boldsymbol\epsilon)$ 连出的straight noising path一般不是teacher PF-ODE穿过 $\boldsymbol x_{t_{n+1}}$ 的那条trajectory。为了构造teacher trajectory上的相邻点，我们从 $\boldsymbol x_{t_{n+1}}$ 出发，用teacher跑一步ODE solver得到 $\hat{\boldsymbol x}_{t_n}$；这个叫consistency distillation（CD）。

> 在CIFAR-10、ImageNet 64×64这些benchmark上，CT不一定比CD差；但在sCM[[3]][r3]的latent ImageNet 512×512实验里，sCD确实scale得更稳。因此后面讨论大模型蒸馏时，我们会主要关注CD这条线。

具体来说，在 $[\epsilon,T]$ 上取一串离散节点

$$
\epsilon=t_1<t_2<\dots<t_N=T
$$

并要求 $\boldsymbol f_\theta$ 满足边界条件

$$
\boldsymbol f_\theta(\boldsymbol x,\epsilon)=\boldsymbol x, \forall x
$$

训练时，让相邻两个节点的预测对齐：

$$
\mathcal L_{\text{dCM}}=\mathbb E_{n,\,\boldsymbol x_0,\,\boldsymbol\epsilon}\Big[\lambda(t_n)\,d\big(\boldsymbol f_\theta(\boldsymbol x_{t_{n+1}},t_{n+1}),\ \boldsymbol f_{\theta^-}(\hat{\boldsymbol x}_{t_n},t_n)\big)\Big]
$$

其中$d$ 是metric，设定上可以是 $\ell_2$、LPIPS，或者iCT[[2]][r2]里的Pseudo-Huber。 $\theta^- = \operatorname{stopgrad}(\operatorname{EMA}(\theta))$ 是target网络。

> EMA可以stabilize target，在self-supervised learning里很常见，比如著名的MoCo和BYOL。不过EMA其实是optional的，比如iCT的loss就没用EMA。

## Continuous-time CMs：sCM（ICLR '25）

dCM有什么问题呢？
- 节点串 $t_1<\dots<t_N$（后续工作LCM发明了一次跳 $k$ 步的变体）是超参。取密了，相邻点的loss信号趋近于0，收敛很慢；取疏了，离散化误差直接进模型。iCT为此攒了一堆trick，突出一个炼丹。
- 每次造 $\hat{\boldsymbol x}_{t_n}$ 都要teacher跑一步solver，而solver自己还有截断误差。
- dCM的多步采样是“从 $\boldsymbol x_{t_n}$ 1-NFE跳回 $\boldsymbol x_0$、重新加一份更小的噪声、再跳回 $\boldsymbol x_0$”的zigzag。它确实可以改善质量——CM原文[[1]][r1]的2-NFE FID就比1-NFE好——但这并不是沿同一条PF-ODE继续精修：每一步都会重新掷噪声，已经生成的信息会被随机扰动，因此也没有ODE solver那种“步长越小，沿同一条轨迹的数值解越准”的保证。

我们回忆CM的核心思想：

$$
\frac{\mathrm{d}}{\mathrm{d}t}\boldsymbol f_\theta(\boldsymbol x_t,t)=\frac{\partial \boldsymbol f_\theta}{\partial t}+\nabla_{\boldsymbol x}\boldsymbol f_\theta\,\frac{\mathrm{d}\boldsymbol x_t}{\mathrm{d}t}=\boldsymbol 0
$$

仔细看这个式子：$\nabla_{\boldsymbol x}\boldsymbol f_\theta$ 是对 $\boldsymbol x$ 的jacobian，$\partial_t\boldsymbol f_\theta$ 是对 $t$ 的jacobian，所以它本质上是个jacobian-vector product（JVP）。

> 鸽子和大多数读者第一次见到JVP可能是在微信公众号三大顶会宣传Kaiming的MeanFlow的时候，虽然JVP其实早已有之——总而言之，可以把JVP当成一个不那么便宜，但因为不用explicitly construct Jacobian，因此价格可以接受的primitive，设定上是pytorch autograd包自带的。

> 然而，真正麻烦的不是JVP的数学，而是现代大模型严重依赖FlashAttention和分布式并行，框架自带的JVP并不能直接覆盖这套kernel。sCM[[3]][r3]在Appendix F给出了FA的 JVP；rCM[[9]][r9]后来实现了兼容大模型并行的FA2 JVP，把它scale到10B以上的image/video model；Causal-rCM[[10]][r10]再把这个kernel扩展到自回归视频需要的custom mask。在此之前，consistency JVP主要还是在ImageNet规模的模型上验证；到这里才算真正进入10B级大模型。

记target network沿teacher ODE的tangent为

$$
\boldsymbol\tau_{\theta^-}(\boldsymbol x_t,t)
:=
\frac{\mathrm d\boldsymbol f_{\theta^-}(\boldsymbol x_t,t)}{\mathrm dt}
=
\nabla_{\boldsymbol x}\boldsymbol f_{\theta^-}(\boldsymbol x_t,t)\,
\boldsymbol v_{\text{teacher}}(\boldsymbol x_t,t)
+
\partial_t\boldsymbol f_{\theta^-}(\boldsymbol x_t,t)
$$

其中 $\boldsymbol v_{\text{teacher}}(\boldsymbol x_t,t)=\frac{\mathrm d\boldsymbol x_t}{\mathrm dt}$。这正好是 $\boldsymbol f_{\theta^-}$ 在输入 $(\boldsymbol x_t,t)$ 上、沿方向 $(\boldsymbol v_{\text{teacher}},1)$ 的JVP：

$$
\boldsymbol\tau_{\theta^-}(\boldsymbol x_t,t)
=
\operatorname{JVP}\!\left(
\boldsymbol f_{\theta^-},
(\boldsymbol x_t,t),
(\boldsymbol v_{\text{teacher}}(\boldsymbol x_t,t),1)
\right)
$$

现在我们回忆一下dCM的loss。令 $t_{n+1}=t+\Delta t$、并且取 $d(\boldsymbol x,\boldsymbol y)=\|\boldsymbol x-\boldsymbol y\|_2^2$ ，在teacher ODE上Taylor：

$$
\boldsymbol f_{\theta^-}(\boldsymbol x_t,t)
=
\boldsymbol f_{\theta^-}(\boldsymbol x_{t+\Delta t},t+\Delta t)
-
\Delta t\,\boldsymbol\tau_{\theta^-}(\boldsymbol x_{t+\Delta t},t+\Delta t)
+
O(\Delta t^2)
$$

代回dCM loss，并且姑且暂时先忘了EMA的事，即取 $\theta^-=sg(\theta)$，于是零阶项刚好抵消。把只依赖于schedule的系数吸收进 $w(t)$，有

$$
\lim_{\Delta t\to0}\frac{1}{2\Delta t}\nabla_\theta\mathcal L_{\mathrm{dCM}}
=
\nabla_\theta\,
\mathbb E_{\boldsymbol x_0,\boldsymbol\epsilon,t}
\left[
w(t)\,
\boldsymbol f_\theta(\boldsymbol x_t,t)^\top
\boldsymbol\tau_{\theta^-}(\boldsymbol x_t,t)
\right].
$$

所以，下面的instantaneous CM objective严格来说是一个gradient-equivalent pseudo-objective：

$$
\mathcal L_{\text{inst}}(\theta)
=
\mathbb E_{\boldsymbol x_0,\boldsymbol\epsilon,t}
\left[
w(t)\,
\boldsymbol f_\theta(\boldsymbol x_t,t)^\top
\boldsymbol\tau_{\theta^-}(\boldsymbol x_t,t)
\right]
$$

虽然这已经是一个可导的objective了，但如果不把teacher tangent stop-gradient，对JVP继续求导就会引入mixed/second-order derivative。
把 $\boldsymbol\tau_{\theta^-}$ detach掉之后，在不做normalization时，可以用类似上一章DMD的MSE trick写出gradient-equivalent的假loss。sCM再把tangent替换成normalized tangent——这一步会有意修改原始梯度以换取稳定性——并加入adaptive weighting，得到

$$
\mathcal L_{\text{sCM}}(\theta,\phi)
=
\mathbb E_{\boldsymbol x_0,\boldsymbol\epsilon,t}
\left[
\frac{e^{w_\phi(t)}}{D}
\left\|
\boldsymbol F_\theta(\boldsymbol x_t,t)
-
\boldsymbol F_{\theta^-}(\boldsymbol x_t,t)
-
\frac{\boldsymbol g}{\|\boldsymbol g\|_2+c}
\right\|_2^2
-w_\phi(t)
\right]
$$

其中

$$
\boldsymbol g
=
w(t)
\boldsymbol\tau_{\theta^-}(\boldsymbol x_t,t)
$$

注意 $f$ 变成了 $F$，中间只依赖 $t$ 的schedule/prior系数被吸收进 $w(t)$；$w_\phi(t)$ 则是另一个learned adaptive weight。分母的regularization叫tangent normalization，目标是避免少数大tangent直接主宰step size。后来的rCM[[9]][r9]保留了同样的 $\|\boldsymbol g\|_2+c$，但观察到normalized residual的量级已经接近常数，因此去掉了sCM的adaptive weighting。

和其他一系列改进相结合之后（包括一个基于三角函数的新schedule TrigFlow，adaptive weight，warmup等等，不过这些不是本文的主题，这里就不赘述了），sCM成功scale到1.5B，并且在ImageNet 512×512的相应评测协议下做到了2-NFE FID=1.88，是当时极少数两步进入2.0以内的结果之一。

## Consistency Trajectory Models (ICLR '24)

之前提到，sCM的 $\boldsymbol f$ 只会把点跳到0。想多步，就只能跳回0、重新加噪，再跳到0。CTM[[4]][r4]把CM extend到从 $t$ 跳到任意 $s\le t$ 的网络：

$$
\boldsymbol g_\theta(\boldsymbol x_t,t,s)\to \boldsymbol x_t+\int_t^s\frac{\mathrm{d}\boldsymbol x_\tau}{\mathrm{d}\tau}\,\mathrm{d}\tau
$$

类似CM，CTM的参数化要保证 $s=t$ 时是恒等映射：

$$
\boldsymbol g_\theta(\boldsymbol x_t,t,s)=\frac{s}{t}\,\boldsymbol x_t+\Big(1-\frac{s}{t}\Big)\,G_\theta(\boldsymbol x_t,t,s)
$$

特别的，
- $s=0$ 时，CTM退化为CM。
- $s\to t$ 时，完整的jump当然退化成恒等映射 $\boldsymbol g_\theta(\boldsymbol x_t,t,t)=\boldsymbol x_t$；但藏在parameterization里的raw network $G_\theta$ 并没有丢掉。对ground-truth jump做一阶Taylor展开，最优raw network在对角线上正好是EDM坐标下的denoiser：

$$
G^*(\boldsymbol x_t,t,t)
=
\mathbb E[\boldsymbol x_0\mid\boldsymbol x_t]
=
\boldsymbol x_t-t\,\frac{\mathrm d\boldsymbol x_t}{\mathrm dt}
$$

> 注：EDM是 $\alpha_t=1, \sigma_t=t$，和RF的 $\alpha_t=1-t,\ \sigma_t=t$ 不同。

因此，可以认为CTM是DM和CM之间的某种插值。

那么，任意区间的jump怎么学？最直接的做法当然是让teacher solver真的从 $t$ 积到 $s$，再拿结果监督student，但这样每个iteration都要求解一整段ODE，蒸馏完说不定比从头训练还贵。CTM用的是**soft consistency**：在 $s\le u<t$ 里随机抽一个中间点 $u$，teacher只负责 $t\to u$，剩下的 $u\to s$ 交给EMA student：

$$
\hat{\boldsymbol x}_u^\phi
=
\operatorname{Solver}_\phi(\boldsymbol x_t,t,u),
\qquad
\boldsymbol g_\theta(\boldsymbol x_t,t,s)
\approx
\boldsymbol g_{\theta^-}(\hat{\boldsymbol x}_u^\phi,u,s).
$$

在理想teacher、并且student收敛时，左右都是同一个PF-ODE上的 $\boldsymbol x_s$：左边让online student一口气跳过去，右边先让teacher走一段，再让stopgrad student补完。LPIPS当然不是只有 $t=0$ 才能算；论文把两边都用stopgrad student搬到0，是为了让feature distance落在语义清楚、彼此可比的clean-data space上：

$$
\begin{aligned}
\boldsymbol x_{\mathrm{est}}
&=
\boldsymbol g_{\theta^-}\!\left(
\boldsymbol g_\theta(\boldsymbol x_t,t,s),s,0
\right),\\
\boldsymbol x_{\mathrm{target}}
&=
\boldsymbol g_{\theta^-}\!\left(
\boldsymbol g_{\theta^-}(\hat{\boldsymbol x}_u^\phi,u,s),s,0
\right),\\
\mathcal L_{\mathrm{CTM}}
&=
\mathbb E_{t,s,u,\boldsymbol x_t}
\left[d(\boldsymbol x_{\mathrm{est}},\boldsymbol x_{\mathrm{target}})\right].
\end{aligned}
$$

这个随机的 $u$ 是soft的来源：$u=s$ 时整段 $t\to s$ 都由teacher solver提供，是昂贵但直接的global consistency；$u=t-\Delta t$ 时teacher只走一小步，是便宜的local consistency；再取 $s=0$，就退回CM的consistency distillation。

只靠这个loss还有一个小洞：$s\to t$ 时，$G_\theta$ 前面的系数 $1-s/t\to0$，raw network在对角线附近几乎收不到梯度。于是CTM额外加一个普通的denoising score matching，直接钉住刚才那条 $s=t$ 对角线：

$$
\mathcal L_{\mathrm{DSM}}
=
\mathbb E\left[
\left\|\boldsymbol x_0-G_\theta(\boldsymbol x_t,t,t)\right\|_2^2
\right].
$$

最终训练目标是 $\mathcal L_{\mathrm{CTM}}+\lambda_{\mathrm{DSM}}\mathcal L_{\mathrm{DSM}}+\lambda_{\mathrm{GAN}}\mathcal L_{\mathrm{GAN}}$。前两个负责把trajectory和score学对，GAN则是和上一章DMD2一样帮助student超越teacher的工具。

在Inference time，作为CM和DM的插值，CTM可以和CM一样跳到0再zigzag回来，和DM一样sample PF-ODE，或者选某个中间版。

## MeanFlow (NeurIPS '25) / Improved MeanFlow (CVPR '26)

> 和上一章统一把DMD、DMD2叫作DMD一样，本文把MeanFlow[[5]][r5]和Improved MeanFlow[[6]][r6]统一叫作MeanFlow。只有需要区分两种训练目标时，才分别叫它们 $u$-space和$v$-space objective。

MeanFlow[[5]][r5]可以看成sCM和CTM的结合。回到RF的schedule

$$
\boldsymbol x_t=(1-t)\boldsymbol x_0+t\boldsymbol\epsilon
$$

定义从 $r$ 到 $t$ 的平均速度：

$$
\boldsymbol u(\boldsymbol x_t,r,t):=\frac{1}{t-r}\int_r^t\boldsymbol v(\boldsymbol x_\tau,\tau)\,\mathrm{d}\tau
$$

于是从 $t$ 回到 $r$ 的一步更新是

$$
\boldsymbol x_r=\boldsymbol x_t-(t-r)\,\boldsymbol u(\boldsymbol x_t,r,t)
$$

这不是Euler近似，而是平均速度的定义。取 $t=1,r=0$，喂一个噪声进去就能1-NFE出图。

MeanFlow的 $\boldsymbol u$ 可以看成CTM的速度版：在RF坐标下，$\boldsymbol g(\boldsymbol x_t,t,r)=\boldsymbol x_t-(t-r)\boldsymbol u(\boldsymbol x_t,r,t)$；CM则是 $r=0$ 的切片。区别只在于怎么训练。

把平均速度的定义两边乘上 $(t-r)$，再沿轨迹对 $t$ 求全导（$r$ 不动），得到MeanFlow Identity：

$$
\boldsymbol u+(t-r)\frac{\mathrm{d}\boldsymbol u}{\mathrm{d}t}=\boldsymbol v(\boldsymbol x_t,t)
$$

也就是

$$
\boldsymbol u(\boldsymbol x_t,r,t)=\boldsymbol v(\boldsymbol x_t,t)-(t-r)\frac{\mathrm{d}\boldsymbol u}{\mathrm{d}t}
$$

其中

$$
\frac{\mathrm{d}\boldsymbol u}{\mathrm{d}t}=\partial_t\boldsymbol u+\boldsymbol v\cdot\nabla_{\boldsymbol x}\boldsymbol u
$$

还是JVP，只不过网络比sCM多吃了一个 $r$。

最直接的训练方法，是把identity右边当target、在 $u$-space做回归。记这个基础目标为

$$
\mathcal L_u=\mathbb E\Big\|\boldsymbol u_\theta(\boldsymbol x_t,r,t)-\operatorname{sg}\Big(\boldsymbol v-(t-r)\big(\partial_t\boldsymbol u_\theta+\boldsymbol v\cdot\nabla_{\boldsymbol x}\boldsymbol u_\theta\big)\Big)\Big\|_2^2
$$

$\boldsymbol v$ 用单样本的条件速度 $\boldsymbol\epsilon-\boldsymbol x_0$ 顶替。$r=t$ 时，identity退化成 $\boldsymbol u=\boldsymbol v$，loss也退化成普通的FM loss；实际训练会混采 $r=t$ 和 $r<t$。

> 严格来说，identity里的 $\boldsymbol v$ 是边际速度场 $\mathbb E[\boldsymbol\epsilon-\boldsymbol x_0\mid\boldsymbol x_t]$。target里的 $\boldsymbol v$ 用条件速度顶替没问题，这是 $\ell_2$ 学条件期望的老规矩；但JVP方向里的 $\boldsymbol v$ 也这样顶替其实是个近似，原文对此有专门讨论，实践上work。

只靠这个基础目标，ImageNet 256×256上就已经能做到from scratch、1-NFE、FID 3.43（XL/2）。不要预训练teacher，不要蒸馏，不要GAN——CM家族折腾了两年的事情，被一个恒等式从头训出来了。鸽子读到这里的时候，第一反应是：这也能行？

> MeanFlow出自CMU的Zhengyang Geng、Zico Kolter和MIT的Kaiming He。第一章提过的JiT也是Kaiming组的，看来他最近确实很喜欢“把生成模型做简单”。

不过这个 $u$-space objective还有两个地方膈应：

1. **target里掺着自己。** $\mathcal L_u$ 的target里含有 $\boldsymbol u_\theta$ 自己的JVP。网络一抖，target就跟着抖；$(t-r)$ 越大，自举项越重，也就越抖。
2. **CFG被烘死。** guidance scale $\omega$ 被直接烘进训练目标，训练时写死，推理时想换就得重训。

MeanFlow的完整修法很直接：把identity反过来用。网络照旧输出 $\boldsymbol u_\theta$，但不再从 $\boldsymbol v$ 造一个带JVP的 $\boldsymbol u$ target，而是先把自己的输出换算成瞬时速度预测：

$$
\hat{\boldsymbol v}:=\boldsymbol u_\theta+(t-r)\Big(\partial_t\boldsymbol u_\theta+\boldsymbol v\cdot\nabla_{\boldsymbol x}\boldsymbol u_\theta\Big)
$$

然后回归干净的FM真值：

$$
\mathcal L_v=\mathbb E\big\|\hat{\boldsymbol v}(\boldsymbol x_t,r,t)-(\boldsymbol\epsilon-\boldsymbol x_0)\big\|_2^2
$$

这样target里就没有网络了，bootstrap消失，问题重新变成一个标准回归。

这和第一章里“9种schema只是换个空间放loss”其实是同一种操作：identity没变、网络输出也没变，只是回归空间从 $u$ 换成了 $v$。$u$-space target里掺着网络自己的JVP，比较脏；$v$-space target就是干净的conditional velocity。

第二刀是把 $\omega$ 当成条件喂给网络。训练时随机采guidance scale，推理时想用多大的CFG就填多大，CFG重新变成运行时的旋钮。

最终结果是ImageNet 256×256、from scratch、1-NFE FID 1.72，相比 $u$-space版本的3.43直接砍半，一步生成和多步diffusion之间的差距基本抹平。这一下确实有点漂亮，属于鸽子喜欢的那种“只换个地方回归，训练性质突然变了”。

> MeanFlow团队把这类方法叫作 *fastforward generative models*。“fastforward”这个词起得很传神：多步采样是在播放ODE，一步生成就是把它快进掉。顺带一提，后面还有SoFlow、TiM[[8]][r8]等工作继续沿着这条线往前走，本系列先按下不表。

## 小结与下集预告

把这条forward线串起来，大概是这样：

| | object | 机器 | teacher |
|---|---|---|---|
| dCM ('23) | 跳到0 | 离散跳步 + solver | 要（CD）/不要（CT） |
| CTM ('23) | **任意区间 $t\to s$** | 离散 + solver + GAN | 要 |
| sCM ('24) | 跳到0 | **连续时间JVP** | 都行 |
| MeanFlow ('25) | 任意区间（平均速度，$v$-space回归） | 连续时间JVP | **不要** |

共同的味道是forward divergence：沿着teacher的轨迹，或者数据本身定义的轨迹做回归。好处是稳、mode covering、不容易塌；坏处也很明显，$\ell_2$ 学的是条件期望，高频细节天然容易糊。

这正好是上一章reverse divergence的镜像：reverse那边更锐，甚至有机会反超teacher，但mode collapse如影随形；forward这边更稳，但上限和细节受条件期望限制。

把两边缝在一起，就是连续时间consistency打底管多样性，score蒸馏（DMD）做正则管锐度——这就是rCM[[9]][r9]。再把整套配方搬进自回归视频，就是Causal-rCM[[10]][r10]。前面三章铺的坑，到这里终于要全部回收了。请看下集！

## Reference

1. Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, 2023. [arXiv:2303.01469][r1]
2. Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. In ICLR, 2024. [arXiv:2310.14189][r2]
3. Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. In ICLR, 2025. [arXiv:2410.11081][r3]
4. Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ODE trajectory of diffusion. In ICLR, 2024. [arXiv:2310.02279][r4]
5. Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. In NeurIPS, 2025. [arXiv:2505.13447][r5]
6. Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J. Zico Kolter, and Kaiming He. Improved mean flows: On the challenges of fastforward generative models. arXiv preprint, 2025. [arXiv:2512.02012][r6]
7. Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022. [arXiv:2206.00364][r7]
8. Zidong Wang, Yiyuan Zhang, Xiaoyu Yue, Xiangyu Yue, Yangguang Li, Wanli Ouyang, and Lei Bai. Transition models: Rethinking the generative learning objective. arXiv preprint, 2025. [arXiv:2509.04394][r8]
9. Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale diffusion distillation via score-regularized continuous-time consistency. In ICLR, 2026. [arXiv:2510.08431][r9]
10. Kaiwen Zheng, Guande He, Min Zhao, Jintao Zhang, Huayu Chen, Jianfei Chen, Chen-Hsuan Lin, Ming-Yu Liu, Jun Zhu, and Qianli Ma. Causal-rCM: A unified teacher-forcing and self-forcing open recipe for autoregressive diffusion distillation in streaming video generation and interactive world models. arXiv preprint, 2026. [arXiv:2606.25473][r10]

[r1]: https://arxiv.org/abs/2303.01469
[r2]: https://arxiv.org/abs/2310.14189
[r3]: https://arxiv.org/abs/2410.11081
[r4]: https://arxiv.org/abs/2310.02279
[r5]: https://arxiv.org/abs/2505.13447
[r6]: https://arxiv.org/abs/2512.02012
[r7]: https://arxiv.org/abs/2206.00364
[r8]: https://arxiv.org/abs/2509.04394
[r9]: https://arxiv.org/abs/2510.08431
[r10]: https://arxiv.org/abs/2606.25473
