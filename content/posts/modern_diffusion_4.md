---
title: "家用现代扩散模型速成 (4)：Consistency Models and (Improved) MeanFlow"
date: "2026-07-11 01:00"
slug: modern_diffusion_4
summary: "D(teacher||student)的蒸馏，以及这条线如何走到不要teacher的1-NFE from scratch。"
---

## 前言

上一章我们跟着Causal-rCM[[10]][r10]把蒸馏分成两条路：Reverse divergence（$D(\text{student}\|\text{teacher})$，mode seeking、on-policy）的代表VSD→DMD已经讲完；这一章轮到**Forward divergence**（$D(\text{teacher}\|\text{student})$，mode covering、offline）：让student把teacher的PF-ODE轨迹一点不落地贴住——代表就是Consistency Model全家桶。

有个微妙的点值得先说破。上一章说forward线"把teacher的轨迹当ground truth"，听起来天生就是蒸馏。但"轨迹"这个监督信号其实有两个来源：可以问teacher（Consistency **Distillation**, CD），也可以直接用noise schedule加数据样本造条件速度（Consistency **Training**, CT）——回忆第一章念过的经，$\ell_2$ 回归会自动帮你把条件速度平均成边际速度。所以这条线走到头，teacher是可选项：本章后半的MeanFlow/iMF干脆完全from scratch，"蒸馏"悄悄变成了一个独立的"一步生成范式"。这也是为什么本章标题里没有"蒸馏"两个字（）

主线还是那句经：**diffusion在连续语境下才优雅**。23年的dCM是这条线的离散原罪，sCM负责赎罪；CTM贡献了一个正确的object，但机器还是离散的；MeanFlow把两者合体，iMF负责售后。全程只有一台引擎：**JVP（Jacobian-vector product，雅可比–向量积）**。

## Recap: 2023年的(discrete) Consistency Models (dCM)

TL; DR：它是一个 v-pred、x-loss，然后**在同一条 PF-ODE 轨迹的相邻两点上强制自洽**。

具体地，dCM[[1]][r1] 在 $[\epsilon, T]$ 上取一串离散节点 $\epsilon=t_1<t_2<\dots<t_N=T$，学一个 consistency function $\boldsymbol f_\theta(\boldsymbol x_t, t)$，要求它把同一条 PF-ODE 轨迹上的任意点都映回同一个起点（干净数据），并带边界条件 $\boldsymbol f_\theta(\boldsymbol x_\epsilon, \epsilon)=\boldsymbol x_\epsilon$。训练目标就是让轨迹上相邻两节点的预测对齐：

$$
\mathcal L_{\text{dCM}}=\mathbb E_{n,\,\boldsymbol x_0,\,\boldsymbol\epsilon}\Big[\lambda(t_n)\,d\big(\boldsymbol f_\theta(\boldsymbol x_{t_{n+1}},t_{n+1}),\ \boldsymbol f_{\theta^-}(\hat{\boldsymbol x}_{t_n},t_n)\big)\Big]
$$

其中 $\hat{\boldsymbol x}_{t_n}$ 是从 $\boldsymbol x_{t_{n+1}}$ 出发、拿预训练 teacher 跑一步 ODE solver 得到的相邻（更干净）的点；$\theta^-=\operatorname{stopgrad}(\operatorname{EMA}(\theta))$ 是 target 网络；$d$ 是个距离度量（$\ell_2$ / LPIPS / 后来 iCT[[2]][r2] 用的 Pseudo-Huber）。所谓 "x-loss"，就是这个 $d$ 直接度量在 clean-$\boldsymbol x$ 空间上——两端各自先把网络的 v 输出换算成 $\boldsymbol x_0$ 预测，再比。轨迹相邻两点一旦处处对齐，$\boldsymbol f_\theta$ 就等于"一步跳到 $\epsilon$"，于是推理时 1 次 NFE 就能出图。

详细的介绍见《家用扩散模型》。

dCM能work，但一身的离散化味道：

- 节点串 $t_1<\dots<t_N$（以及LCM那种跳 $k$ 步的变体里的 $k$）全是超参：取密了，相邻两点的loss差分信号趋近于0、收敛奇慢；取疏了，离散化误差直接进模型。后续的iCT[[2]][r2]为此攒了一堆trick（Pseudo-Huber、EMA schedule、$N$ 的curriculum……），能train但突出一个炼丹。
- 造 $\hat{\boldsymbol x}_{t_n}$ 要teacher跑一步solver，solver自己还带截断误差。
- 推理想拿多步换质量？dCM的多步是"跳回0、加噪回中间、再跳回0"的zigzag，每一轮都在重新掷骰子，步数堆上去质量并不单调变好。

而本系列从第一章就开始念的经是：diffusion在连续语境下才优雅。CM当然也应该有个 $\Delta t\to 0$ 的版本——事实上CM原论文[[1]][r1]的附录里就推过continuous-time CM，梯度形式很漂亮，就是**根本训不动**。于是23年大家集体退回离散版炼丹去了。直到sCM。

## Continuous-time CMs (sCM)

Consistency的本意是"$\boldsymbol f$ 沿同一条PF-ODE轨迹是常数"。离散版把它写成相邻节点对齐；连续版直接写成微分条件：

$$
\frac{\mathrm{d}}{\mathrm{d}t}\boldsymbol f_\theta(\boldsymbol x_t,t)=\frac{\partial \boldsymbol f_\theta}{\partial t}+\frac{\mathrm{d}\boldsymbol x_t}{\mathrm{d}t}\cdot\nabla_{\boldsymbol x}\boldsymbol f_\theta=\boldsymbol 0\quad\text{（沿PF-ODE轨迹）}
$$

其中 $\frac{\mathrm{d}\boldsymbol x_t}{\mathrm{d}t}$ 是轨迹的速度场——蒸馏时问teacher（sCD），from scratch时用条件速度顶替（sCT）。关键的观察是：这个全导数**不需要有限差分**——它就是 $\boldsymbol f$ 沿方向 $\big(\frac{\mathrm{d}\boldsymbol x_t}{\mathrm{d}t},\,1\big)$ 的方向导数，一次forward-mode自动微分（`torch.func.jvp` / `jax.jvp`）精确算出。$N$、$k$、step schedule，全部蒸发。

loss呢？对 $\mathcal L_{\text{dCM}}$ 做Taylor展开。相邻点 $(\hat{\boldsymbol x}_{t},t)$ 和 $(\boldsymbol x_{t+\Delta t},t+\Delta t)$ 在同一条轨迹上，所以

$$
\boldsymbol f_{\theta^-}(\hat{\boldsymbol x}_{t},t)=\boldsymbol f_{\theta^-}(\boldsymbol x_{t+\Delta t},t+\Delta t)-\Delta t\,\frac{\mathrm{d}\boldsymbol f_{\theta^-}}{\mathrm{d}t}+O(\Delta t^2)
$$

取 $d=\ell_2^2$、并令 $\theta^-=\operatorname{sg}(\theta)$（sCM顺手把target的EMA也扔了），代回去：

$$
\mathcal L=\mathbb E\Big\|\boldsymbol f_\theta-\boldsymbol f_{\theta^-}+\Delta t\,\frac{\mathrm{d}\boldsymbol f_{\theta^-}}{\mathrm{d}t}\Big\|_2^2
$$

（所有量都在同一点 $(\boldsymbol x_{t+\Delta t},t+\Delta t)$ 取值。）注意前向值上 $\boldsymbol f_\theta=\boldsymbol f_{\theta^-}$，于是

$$
\nabla_\theta\mathcal L=2\Delta t\,\mathbb E\Big[\Big(\frac{\mathrm{d}\boldsymbol f_{\theta^-}}{\mathrm{d}t}\Big)^{\!\top}\frac{\partial\boldsymbol f_\theta}{\partial\theta}\Big]+O(\Delta t^2)
$$

除以 $2\Delta t$、取极限，就得到连续时间CM的"梯度"。眼熟吗——"手里有个想要的梯度 $\boldsymbol g$，把它包成stopgrad loss"，上一章末尾拆Causal-rCM里DMD loss的时候刚见过同款把戏：

$$
\mathcal L_{\text{sCM}}=\mathbb E\Big[w(t)\,\big\|\boldsymbol f_\theta(\boldsymbol x_t,t)-\operatorname{sg}\big[\boldsymbol f_\theta(\boldsymbol x_t,t)-\boldsymbol g\big]\big\|_2^2\Big],\qquad \boldsymbol g=\frac{\mathrm{d}\boldsymbol f_{\theta^-}(\boldsymbol x_t,t)}{\mathrm{d}t}
$$

这就是CM原论文附录里那个训不动的continuous-time CM。sCM[[3]][r3]（OpenAI的Lu & Song——对，就是DPM-Solver的Lu和score SDE的Song）干的事情，就是找到它为什么炸、修好、然后scale上去。三板斧：

**(1) TrigFlow：把系数全变成三角函数。** 取 $\alpha_t=\cos t$、$\sigma_t=\sin t$、$t\in[0,\frac{\pi}{2}]$——第一章那套一般schedule的一个特例（数据事先归一到std $\sigma_d$）：

$$
\boldsymbol x_t=\cos(t)\,\boldsymbol x_0+\sin(t)\,\boldsymbol z,\qquad \boldsymbol z\sim\mathcal N(\boldsymbol 0,\sigma_d^2\boldsymbol I)
$$

一致性函数参数化为

$$
\boldsymbol f_\theta(\boldsymbol x_t,t)=\cos(t)\,\boldsymbol x_t-\sin(t)\,\sigma_d\,F_\theta\!\Big(\frac{\boldsymbol x_t}{\sigma_d},t\Big)
$$

$t=0$ 时 $\boldsymbol f_\theta=\boldsymbol x_0$，边界条件自动满足；而且此时PF-ODE的速度场恰好就是 $\sigma_d F(\cdot)$ 的形式，$F$ 是个干干净净的v-pred网络。TrigFlow的意义不是玄学审美：所有系数都变成 $\sin/\cos$ 之后，接下来拆tangent不会被一堆 $c_{\text{skip}}/c_{\text{out}}$ 的导数糊一脸——顺手还统一了EDM、FM、CM三家的符号。

**(2) 拆tangent，找到炸点，对症下药。** 把 $\boldsymbol g$ 沿TrigFlow展开：

$$
\frac{\mathrm{d}\boldsymbol f_{\theta^-}}{\mathrm{d}t}=\underbrace{-\sin(t)\,\boldsymbol x_t+\cos(t)\,\frac{\mathrm{d}\boldsymbol x_t}{\mathrm{d}t}-\cos(t)\,\sigma_d F_{\theta^-}}_{\text{现成的量，乖得很}}\ \underbrace{-\ \sin(t)\,\sigma_d\,\frac{\mathrm{d}F_{\theta^-}}{\mathrm{d}t}}_{\text{炸点}}
$$

元凶是网络自身的时间全导数 $\frac{\mathrm{d}F_{\theta^-}}{\mathrm{d}t}$：JVP会把time embedding的导数一起带出来，在部分噪声档方差大得离谱。药方三味：

- **Tangent normalization**：$\boldsymbol g\leftarrow\boldsymbol g/(\|\boldsymbol g\|+c)$（$c=0.1$），把这个梯度场逐点钉在单位球附近，方差不再爆炸；
- **Tangent warmup**：炸点那一项的系数从0线性升到1，先把稳的部分学出来再放开；
- **修time embedding**：Fourier embedding的尺度调小等一系列修正——因为对 $t$ 求导会把这些模块的高频震荡原样倒进tangent里。

**(3) Adaptive weighting。** 不同 $t$ 的loss量级差好几个数量级，手调 $w(t)$ 不现实。sCM借EDM2[[7]][r7]的uncertainty weighting把权重当参数学：

$$
\mathcal L(\theta,\varphi)=\mathbb E\Big[\frac{e^{w_\varphi(t)}}{D}\,\|\cdot\|_2^2-w_\varphi(t)\Big]
$$

对 $w_\varphi$ 逐点求极值得 $e^{w_\varphi}\propto1/\|\cdot\|^2$——自动学出"loss的倒数"当权重，各噪声档被拉回同一量级（本质是异方差回归那一套）。

三板斧下去，连续时间CM第一次被**稳定地**scale上去：1.5B参数、ImageNet 512×512，2-NFE的FID 1.88，和teacher的差距缩到10%以内。没有 $N$、没有schedule、没有跳步误差——非常对本系列的胃口。

> sCD（蒸馏）和sCT（from scratch）sCM都做了，后来rCM接走的是sCD这一支。另外JVP在工程上并不白给：FlashAttention这类fused kernel默认没有forward-mode导数，要scale到video级得自己写JVP kernel——这个坑由Causal-rCM来填，下集见。

## Consistency Trajectory Models (CTM)

sCM把机器修好了，但object没变：$\boldsymbol f$ 只会"跳到0"。这个设计有个先天残疾——想多步的时候只能zigzag（跳到0、加噪回 $t'$、再跳到0），每轮加噪都在丢掉已经算出来的信息。CTM[[4]][r4]于是把object升级：让网络直接学"从 $t$ 跳到**任意** $s\le t$"，也就是PF-ODE的解算子本身：

$$
\boldsymbol g_\theta(\boldsymbol x_t,t,s)\approx\boldsymbol x_t+\int_t^s\frac{\mathrm{d}\boldsymbol x_\tau}{\mathrm{d}\tau}\,\mathrm{d}\tau
$$

> 时间线澄清：CTM（23年10月）其实比sCM早一年，和dCM/iCT是同代人。这里按"先修机器、再换object"的逻辑顺序讲，纯粹因为好讲（

（CTM生活在EDM坐标系里，$t$ 直接就是噪声水平 $\sigma$。）参数化保证 $s=t$ 时恒等：

$$
\boldsymbol g_\theta(\boldsymbol x_t,t,s)=\frac{s}{t}\,\boldsymbol x_t+\Big(1-\frac{s}{t}\Big)\,G_\theta(\boldsymbol x_t,t,s)
$$

这个**two-time object**的妙处在两个端点：

- $s=0$：整段积分一步跳完——这就是CM的 $\boldsymbol f$；
- $s\to t$：一阶Taylor展开给出 $G_\theta(\boldsymbol x_t,t,t)=\boldsymbol x_t-t\,\frac{\mathrm{d}\boldsymbol x_t}{\mathrm{d}t}$，在EDM坐标下这恰好是denoiser（x-pred）——也就是普通的diffusion/score model。

**CM和diffusion原来是同一个object的两个切片**——这个观察本身就值一篇论文。

训练用soft consistency matching：直接跳 $t\to s$，和"先让teacher的solver走一小段到 $u\in[s,t)$、再由stopgrad网络跳 $u\to s$"对齐：

$$
\boldsymbol g_\theta(\boldsymbol x_t,t,s)\ \approx\ \boldsymbol g_{\operatorname{sg}(\theta)}\big(\texttt{Solver}_\phi(\boldsymbol x_t,\,t\to u),\,u,\,s\big)
$$

（两边还会再被sg网络送到 $s=0$、在clean-$\boldsymbol x$ 空间上比距离，仍然是x-loss。）另外再挂一个DSM loss锚住 $s\to t$ 端，一个GAN loss提锐度。有了任意区间跳转，多步采样也不用zigzag了：CTM的 $\gamma$-sampling在"纯deterministic长跳"（$\gamma=0$）和"CM式回零加噪"（$\gamma=1$）之间连续插值，质量/多样性可调。

平心而论，CTM的成绩单（CIFAR-10 1-NFE FID 1.73）是靠LPIPS加GAN堆出来的，训练也还是离散跳步+teacher solver那一套，2023年的时代眼泪一样不少。但它留下的遗产是那个two-time object：**网络多吃一个"目标时刻"，diffusion和CM就变成了一族**。idea超前，机器落后——它在等一台连续时间的机器。

> CTM在forward主干上挂GAN，DMD2在reverse主干上挂GAN——两边都默认"纯蒸馏的锐度不够用"。26年的rCM给出的答案是把GAN换成score蒸馏正则，这是下集的事。

## MeanFlow

现在把两个礼物放在一起：sCM给了**连续时间的机器**（JVP加驯服术），CTM给了**对的object**（two-time跳转）。2025年的MeanFlow[[5]][r5]把它们合了体——顺便把teacher也扔了。

回到RF坐标（$\boldsymbol x_t=(1-t)\boldsymbol x_0+t\boldsymbol\epsilon$）。定义从 $r$ 到 $t$ 的**平均速度**：

$$
\boldsymbol u(\boldsymbol x_t,r,t):=\frac{1}{t-r}\int_r^t\boldsymbol v(\boldsymbol x_\tau,\tau)\,\mathrm{d}\tau
$$

按定义，一步跳转是**精确**的——不是Euler近似，平均速度的定义就是把积分摊平：

$$
\boldsymbol x_r=\boldsymbol x_t-(t-r)\,\boldsymbol u(\boldsymbol x_t,r,t)
$$

取 $t=1$、$r=0$：喂一个噪声进去，1-NFE直接出图。和前文对表：$\boldsymbol u$ 就是CTM的 $\boldsymbol g$ 的速度版（RF坐标下 $\boldsymbol g(\boldsymbol x_t,t,r)=\boldsymbol x_t-(t-r)\boldsymbol u(\boldsymbol x_t,r,t)$），而CM的 $\boldsymbol f$ 是 $r=0$ 的切片。区别全在怎么训：MeanFlow不要teacher、不要solver、不要GAN，只要一个恒等式。

**MeanFlow Identity.** 把定义两边乘 $(t-r)$，再沿轨迹对 $t$ 求全导（$r$ 按住不动）：左边乘积法则，右边微积分基本定理：

$$
\boldsymbol u+(t-r)\frac{\mathrm{d}\boldsymbol u}{\mathrm{d}t}=\boldsymbol v(\boldsymbol x_t,t)
\quad\Longrightarrow\quad
\boldsymbol u(\boldsymbol x_t,r,t)=\boldsymbol v(\boldsymbol x_t,t)-(t-r)\,\frac{\mathrm{d}\boldsymbol u}{\mathrm{d}t}
$$

其中 $\frac{\mathrm{d}\boldsymbol u}{\mathrm{d}t}=\partial_t\boldsymbol u+\boldsymbol v\cdot\nabla_{\boldsymbol x}\boldsymbol u$——又是熟悉的配方：$\boldsymbol u_\theta$ 沿方向 $(\boldsymbol v,\,0,\,1)$（$\boldsymbol x$ 随 $\boldsymbol v$ 动、$r$ 不动、$t$ 走1）的JVP。和sCM同一台引擎，只是网络多吃了一个 $r$。

训练就是让 $\boldsymbol u_\theta$ 回归identity的右边（右边整体stopgrad）：

$$
\mathcal L_{\text{MF}}=\mathbb E\Big\|\boldsymbol u_\theta(\boldsymbol x_t,r,t)-\operatorname{sg}\Big(\boldsymbol v-(t-r)\big(\partial_t\boldsymbol u_\theta+\boldsymbol v\cdot\nabla_{\boldsymbol x}\boldsymbol u_\theta\big)\Big)\Big\|_2^2
$$

$\boldsymbol v$ 拿单样本的条件速度 $\boldsymbol\epsilon-\boldsymbol x_0$ 顶替。$r=t$ 时identity退化成 $\boldsymbol u=\boldsymbol v$，loss退化成普通的FM loss；实际训练混采 $r=t$ 和 $r<t$ 两种样本。

> 严格说identity里的 $\boldsymbol v$ 是边际速度场 $\mathbb E[\boldsymbol\epsilon-\boldsymbol x_0\mid\boldsymbol x_t]$。target里的 $\boldsymbol v$ 用条件速度顶替没问题（$\ell_2$ 学条件期望的老规矩），但JVP**方向**里的 $\boldsymbol v$ 也这么顶替其实是个近似——原文对此有专门讨论，实践上work。

结果：ImageNet 256×256 **from scratch、1-NFE、FID 3.43**（XL/2），不要预训练teacher、不要蒸馏、不要GAN。CM家族折腾了两年的事情，被一个恒等式从头训出来了。

不过MeanFlow的CFG处理得比较僵硬：带guidance的速度场连同scale $\omega$ 一起被烘进训练目标，$\omega$ 训练时写死，推理想换？重训吧。这个伏笔马上收。

> MeanFlow出自CMU的Zhengyang Geng、Zico Kolter和MIT的Kaiming He——第一章提过的JiT也是Kaiming组的。看得出来他最近对"把生成模型做简单"这件事相当上头。

## Improved MeanFlow (iMF)

MeanFlow很美，但真训过的人知道它有两个膈应的地方（iMF[[6]][r6]的诊断）：

1. **Target里掺着自己。** 看 $\mathcal L_{\text{MF}}$：回归目标里含 $\boldsymbol u_\theta$ 自己的JVP。这是bootstrap——网络一抖，target跟着抖；$(t-r)$ 越大，target里自举项的占比越高，越抖。整个consistency家族其实都有这毛病（sCM的 $\boldsymbol g$ 里也是 $F_{\theta^-}$ 自己），sCM靠normalization和warmup把它摁住；iMF的观察是：MeanFlow根本可以不这样。
2. **CFG被烘死。** 上一节刚吐槽过。

iMF的修法漂亮在几乎什么都没加——把identity**反过来用**。MeanFlow是"从 $\boldsymbol v$ 造 $\boldsymbol u$ 的target"，target里掺着网络；iMF反之：网络照旧输出 $\boldsymbol u_\theta$，但用identity把它**换算成对瞬时速度的预测**

$$
\hat{\boldsymbol v}:=\boldsymbol u_\theta+(t-r)\Big(\partial_t\boldsymbol u_\theta+\boldsymbol v\cdot\nabla_{\boldsymbol x}\boldsymbol u_\theta\Big)
$$

然后拿 $\hat{\boldsymbol v}$ 去回归**干净的、与网络无关的**FM真值：

$$
\mathcal L_{\text{iMF}}=\mathbb E\big\|\hat{\boldsymbol v}(\boldsymbol x_t,r,t)-(\boldsymbol\epsilon-\boldsymbol x_0)\big\|_2^2
$$

target彻底不含网络，bootstrap消失，整个问题回到一个标准回归。第一章的读者应该会心一笑：这和"9种schema里换个空间放loss"是同一类操作——**同一个identity，换个空间做回归，训练性质天差地别**。MeanFlow在 $u$-空间回归（target脏），iMF在 $v$-空间回归（target干净）。

第二刀：**把 $\omega$ 当条件喂**。iMF把guidance scale做成in-context的条件（像 $t$、$r$ 一样喂进网络），训练时随机采 $\omega$，推理时想要多强的guidance就填多大——CFG从"烘进权重"变回"运行时的旋钮"。

结果：ImageNet 256×256 **1-NFE FID 1.72**，仍然from scratch、不蒸馏——把MeanFlow的数字直接砍半，一步生成和多步diffusion的差距基本抹平。

> iMF论文的副标题是 *On the Challenges of Fastforward Generative Models*。"fastforward"这个词起得传神：多步采样是在**播放**ODE，一步生成是把它**快进**掉。顺带一提，iMF之后这个方向还在高速产出——SoFlow用有限差分绕开JVP、TiM[[8]][r8]把diffusion/CM/MeanFlow统一成"任意区间transition"的三个切面……本系列先按下不表。

## 小结与下集预告

串起来看，这条forward线是一场"object × 机器"的双人舞：

| | object | 机器 | teacher |
|---|---|---|---|
| dCM ('23) | 跳到0 | 离散跳步 + solver | 要（CD）/不要（CT） |
| CTM ('23) | **任意区间 $t\to s$** | 离散 + solver + GAN | 要 |
| sCM ('24) | 跳到0 | **连续时间JVP** | 都行 |
| MeanFlow ('25) | 任意区间（平均速度） | 连续时间JVP | **不要** |
| iMF ('25) | 任意区间（$v$-空间回归） | 连续时间JVP | 不要 |

共同的味道是forward divergence：回归式地贴住轨迹（teacher的，或数据定义的），稳、mode covering、不塌缩。但上限也写在脸上：$\ell_2$ 回归学到的是条件期望，高频细节天然容易发糊——这恰好是上一章reverse divergence（锐、能反超teacher，但mode collapse如影随形）的镜像。

一边稳而糊，一边锐而塌。把互补的两个缝在一起——**连续时间一致性打底管多样性，score蒸馏（DMD）做正则管锐度**——就是rCM[[9]][r9]；再把整套配方搬进自回归视频，就是Causal-rCM[[10]][r10]。请看下集！

## Reference

1. Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, 2023. [arXiv:2303.01469][r1]
2. Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. In ICLR, 2024. [arXiv:2310.14189][r2]
3. Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. In ICLR, 2025. [arXiv:2410.11081][r3]
4. Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ODE trajectory of diffusion. In ICLR, 2024. [arXiv:2310.02279][r4]
5. Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. In NeurIPS, 2025. [arXiv:2505.13447][r5]
6. Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J. Zico Kolter, and Kaiming He. Improved mean flows: On the challenges of fastforward generative models. arXiv preprint, 2025. [arXiv:2512.02012][r6]
7. Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In CVPR, 2024. [arXiv:2312.02696][r7]
8. Zidong Wang, Yiyuan Zhang, Xiaoyu Yue, Xiangyu Yue, Yangguang Li, Wanli Ouyang, and Lei Bai. Transition models: Rethinking the generative learning objective. arXiv preprint, 2025. [arXiv:2509.04394][r8]
9. Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale diffusion distillation via score-regularized continuous-time consistency. In ICLR, 2026. [arXiv:2510.08431][r9]
10. Kaiwen Zheng, Guande He, Min Zhao, Jintao Zhang, Huayu Chen, Jianfei Chen, Chen-Hsuan Lin, Ming-Yu Liu, Jun Zhu, and Qianli Ma. Causal-rCM: A unified teacher-forcing and self-forcing open recipe for autoregressive diffusion distillation in streaming video generation and interactive world models. arXiv preprint, 2026. [arXiv:2606.25473][r10]

[r1]: https://arxiv.org/abs/2303.01469
[r2]: https://arxiv.org/abs/2310.14189
[r3]: https://arxiv.org/abs/2410.11081
[r4]: https://arxiv.org/abs/2310.02279
[r5]: https://arxiv.org/abs/2505.13447
[r6]: https://arxiv.org/abs/2512.02012
[r7]: https://arxiv.org/abs/2312.02696
[r8]: https://arxiv.org/abs/2509.04394
[r9]: https://arxiv.org/abs/2510.08431
[r10]: https://arxiv.org/abs/2606.25473
