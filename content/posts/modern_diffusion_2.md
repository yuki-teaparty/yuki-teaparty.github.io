---
title: "家用现代扩散模型速成 (2)：weighting scheme"
date: "2026-06-30 22:00"
slug: modern_diffusion_2
order: 3.05
series: "家用现代扩散模型速成"
summary: "RF把schedule钉死之后还剩的超参：训练时timestep按什么分布撒，推理时步点怎么排、Wan/Flux各自怎么shift，以及为什么最后还是Euler。"
draft: true
---

## 前言

上一篇讲到，自SD3以后RF一统江湖，schedule被钉死成 $\alpha_t=1-t,\sigma_t=t$ 的一条直线。schedule没得选了，solver也退化成Euler了——那训练的时候到底还剩下什么自由度？

答案其实还不少，分两拨：**训练**时 $t$ 按什么分布撒（等价于给loss怎么加权），和**推理**时那个ODE用什么solver去解（步点放哪、用几阶、要不要加随机性）。

前一拨在离散时代（DDPM）几乎不存在——那时候 $t$ 是 $\{1,\dots,1000\}$ 上的均匀分布，撒法就一种，写死在for循环里。可一旦进了连续时间，$t\in[0,1]$ 就是一个连续随机变量，它的分布 $\pi(t)$ 成了一个可以自由设计的东西，而这个设计对效果的影响大到离谱。上一篇把schedule和solver都"收敛"了，但收敛不代表没有超参——只是超参从"schedule长什么样"挪到了"怎么撒 $t$、怎么解ODE"上。这一篇就把这些旋钮一个个拧一遍。

## 从离散到连续：t schedule

先把训练目标写清楚。以RF的v-loss为例：

$$
\mathcal{L}=\mathbb{E}_{t\sim\pi(t)}\;\mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{\epsilon}}\big[\,\|\boldsymbol{v}_\theta(\boldsymbol{x}_t,t)-\boldsymbol{v}\|^2\,\big]
$$

在离散DDPM里，对 $t$ 的期望就是 $\frac1T\sum_{t=1}^T$ 这样一个朴素求和（离散扩散最丑陋的地方之一（）；在连续时间里，它变成

$$
\mathbb{E}_{t\sim\pi}[\ell(t)]=\int_0^1 \pi(t)\,\ell(t)\,dt
$$

其中 $\ell(t)$ 是给定 $t$、对 $(\boldsymbol{x}_0,\boldsymbol{\epsilon})$ 求完期望后剩下的那部分loss。

这里有一个一眼就能看穿、但非常有用的观察：**"$t$ 按 $\pi$ 采样" 和 "$t$ 均匀采样、但给loss乘一个权重 $w(t)=\pi(t)$" 是同一件事**：

$$
\int_0^1 \pi(t)\,\ell(t)\,dt=\mathbb{E}_{t\sim U[0,1]}\big[\pi(t)\,\ell(t)\big]
$$

换句话说，"$t$ 撒在哪" 和 "哪些 $t$ 的loss更重要" 是一体两面。接下来的所有花活，本质都是在设计这个 $\pi(t)$（或者等价地 $w(t)$）。

> 有人管这叫loss weighting，有人管这叫timestep sampling，吵来吵去，其实是同一个东西的两种记号（

### uniform：最朴素的撒法

最直接的选择当然是 $t\sim U[0,1]$，RF原文和早期的一些实现就是这么干的。

它的问题不在"均匀"这个词本身，而在于——**在 $t$ 上均匀，并不等于在真正衡量"噪声强度"的那个量上均匀**。上一篇定义过log-SNR $\lambda_t=\log\frac{\alpha_t}{\sigma_t}$，对RF来说

$$
\lambda_t=\log\frac{1-t}{t}=-\operatorname{logit}(t)
$$

把 $t\sim U[0,1]$ 换元到 $\lambda$，密度是

$$
p_\lambda(\lambda)=\Big|\frac{dt}{d\lambda}\Big|=\frac{e^{\lambda}}{(1+e^{\lambda})^2}
$$

这是一个标准logistic分布：钟形，中心在 $\lambda=0$（也就是 $t=0.5$），往两端 $\lambda\to\pm\infty$ 迅速衰减。

也就是说，均匀撒 $t$ 其实已经隐含地把大量样本压在了中间噪声档（$t\approx0.5$），几乎不管两个端点——这本身是好事，因为 $t\to0$（几乎干净）和 $t\to1$（几乎纯噪声）这两头的回归目标太trivial，网络学不到什么，真正难、真正决定生成质量的是中间那段。

但既然uniform给出的是一个**写死的** logistic（中心、宽度都动不了），那自然的下一个问题就是：能不能让这个"中间钟形"的中心和宽度可调？

### logit-normal：让log-SNR服从高斯

SD3[[1]][r1]给的答案是logit-normal。撒法极简单：先撒一个高斯，再用sigmoid压回 $[0,1]$：

$$
u\sim\mathcal N(m,s^2),\qquad t=\sigma(u)=\frac1{1+e^{-u}}
$$

换元回 $t$，密度是

$$
\pi_{\text{ln}}(t;m,s)=\frac{1}{s\sqrt{2\pi}}\cdot\frac{1}{t(1-t)}\exp\!\Big(-\frac{(\operatorname{logit}(t)-m)^2}{2s^2}\Big)
$$

$m$ 控制峰的位置（峰在 $t=\sigma(m)$，$m=0$ 时正好是 $0.5$），$s$ 控制集中程度。

但它真正优雅的地方，要换到 $\lambda$ 上才看得出来。注意 $u=\operatorname{logit}(t)=-\lambda$，于是

$$
u\sim\mathcal N(m,s^2)\quad\Longleftrightarrow\quad \lambda\sim\mathcal N(-m,\,s^2)
$$

**logit-normal撒 $t$，等价于让log-SNR $\lambda$ 服从一个高斯。** 对比一下：uniform给的是一个固定的logistic$(0,1)$，logit-normal给的是一个可调中心、可调宽度的高斯 $\mathcal N(-m,s^2)$——无非是把那个写死的钟形换成一个能调参的钟形而已。SD3实验里最好的一档是 $m=0,s=1$，比uniform更集中在中间。

> 一旦把"撒 $t$"翻译成"撒 $\lambda$"，很多看着完全不同的采样方案其实都是同一族高斯，差别只在中心和宽度。这正是连续时间的好处——离散扩散那套按step编号撒 $t$ 的做法，是永远看不到这层的（

### timestep shift：高分辨率的麻烦

到目前为止的讨论都默认了一件事：给定 $t$，加进去的噪声"强度"是固定的。可这在高分辨率下不成立。

直觉是这样：一张高分辨率图，相邻像素（latent）高度相关、冗余度极高。在同一个 $t$、同样的 $\sigma_t$ 下，往低分辨率图上加的噪声足以毁掉大半信息，但往高分辨率图上加同一量级的噪声，低频结构几乎纹丝不动——你downsample回去甚至看不出加过噪。换句话说，**分辨率越高，同一个 $t$ 对应的"有效信噪比"越高**，网络在高噪声端根本没被训练够。

补救办法叫timestep shift：把整条schedule往"更多噪声"的方向推。具体地，给一个shift factor $s>1$，把 $t$ 重映射成

$$
t'=\frac{s\,t}{1+(s-1)\,t}
$$

这个变换固定 $0\mapsto0$、$1\mapsto1$，中间往 $1$（高噪声）那头拱。

在 $t$-空间里看，这是个莫名其妙的Möbius变换；但老规矩，换到 $\lambda$ 上就现原形了。RF下

$$
\frac{\sigma'}{\alpha'}=\frac{t'}{1-t'}=\frac{s\,t}{1-t}=s\cdot\frac{\sigma}{\alpha}\quad\Longrightarrow\quad \lambda'=\lambda-\log s
$$

**timestep shift在 $t$ 上是个丑陋的分式变换，在log-SNR上不过是一个常数平移** $\lambda'=\lambda-\log s$——这才是"shift"这个名字的由来。$s>1$ 让 $\lambda$ 整体变小，即整体更noisy，正好补上高分辨率缺的那块高噪声训练。

至于 $s$ 取多少：simple diffusion[[2]][r2]最早指出应该按分辨率平移log-SNR；SD3[[1]][r1]把它写成上面这个RF的timestep形式，并把shift factor跟token数挂钩（$s=\sqrt{m/n}$，$n,m$ 分别是参考、目标分辨率的token数）。而现代模型（Wan、Flux）在推理时具体怎么设这个 $s$，下一节一起讲。

### 一个更狠的统一，和一个例外

"采样=加权"这个观察还能再往前推一步。回想上一篇，x-loss、ε-loss、v-loss 三者只差一个 $t$ 相关的系数：

$$
\|\boldsymbol{\epsilon}_\theta-\boldsymbol{\epsilon}\|^2=e^{2\lambda_t}\|\boldsymbol{x}_\theta-\boldsymbol{x}_0\|^2,\qquad \|\boldsymbol{v}_\theta-\boldsymbol{v}\|^2=\alpha_t^2\dot\lambda_t^2\|\boldsymbol{x}_\theta-\boldsymbol{x}_0\|^2
$$

所以选哪种 loss，无非是给最根本的 denoising loss $\|\boldsymbol{x}_\theta-\boldsymbol{x}_0\|^2$ 再乘一个已知的 $t$ 相关权重。把它和前面的 $\pi(t)$、$w(t)$ 摞在一起，整个训练目标就是

$$
\mathcal{L}=\int \underbrace{\pi(t)}_{\text{采样}}\,\underbrace{w(t)}_{\text{显式权重}}\,\underbrace{g_{\text{loss}}(t)}_{\text{loss 类型}}\,\mathbb{E}\|\boldsymbol{x}_\theta-\boldsymbol{x}_0\|^2\,dt
$$

三个因子全乘进同一个有效权重，换元到 $\lambda$ 上就是一个 $w_{\text{eff}}(\lambda)$ 乘唯一那个 denoising loss。这正是 VDM++[[3]][r3] 的主旨：所有 diffusion 训练目标都是同一个东西在 $\lambda$ 上的加权积分，区别只在权重。

> diffusers 干脆把"采样密度"和"loss 权重"两个函数用了同一个参数名 `weighting_scheme`（`compute_density_for_timestep_sampling` 和 `compute_loss_weighting_for_sd3`）——库自己都懒得区分，因为它俩本就是一回事（

但要小心：能塌进 weighting 的只是 **loss** 那一维。上一篇把 schema 拆成 prediction × loss 两个轴，这里塌缩的是 loss 轴；prediction 轴——网络到底 physically 吐 $\boldsymbol{x}_\theta$、$\boldsymbol{\epsilon}_\theta$ 还是 $\boldsymbol{v}_\theta$——是 **preconditioning**，塞不进任何标量权重。

把裸网络输出记作 $F_\theta$，我们总得用一个仿射映射把它反解成 $\boldsymbol{x}_0$ 的估计：

$$
\boldsymbol{x}_\theta=c_{\text{skip}}(t)\,\boldsymbol{x}_t+c_{\text{out}}(t)\,F_\theta
$$

x/ε/v-prediction 不过是三组写死的 $(c_{\text{skip}},c_{\text{out}})$：x-pred 是 $(0,1)$，ε-pred 是 $(\tfrac1{\alpha_t},-\tfrac{\sigma_t}{\alpha_t})$，RF 的 v-pred 是 $(1,-t)$——于是 $\boldsymbol{x}_\theta=\boldsymbol{x}_t-t\,\boldsymbol{v}_\theta$。这个映射决定了**裸网络在每个 $t$ 上要回归多大尺度的东西**（ε 的方差恒为 1，$\boldsymbol{x}_0$ 是数据方差，$\boldsymbol{v}$ 介于两者之间），也决定了 $c_{\text{skip}}$ 先把 $\boldsymbol{x}_t$ 里现成的部分扣掉、让网络只学残差（低噪声时 $t\boldsymbol{v}_\theta\to0$，几乎啥都不用干）。一个跨所有 $t$ 共享权重的网络，只有当输入/输出在每个 $t$ 都是 $O(1)$、且只需拟合"难的那部分残差"时才训得匀。而 $w(t)$ 缩放的是 loss 本身，动不了网络输出的尺度——这是两件事。

> **这里的 preconditioning 是什么意思？** 不是二阶 solver 里给线性系统左乘一个 $M^{-1}$ 那种——这里压根没有额外的矩阵。它是更一般的"换元"意义上的 preconditioning：挑一个可逆的变量替换，把问题的条件数压小。而这里的"矩阵"是 $M(t)=c(t)\,I$，一个**逐噪声档的标量**。
>
> 被改善条件数的不是像素之间，而是 $t$ 这条连续轴：不同噪声档的目标/梯度天然差好几个数量级，Gauss–Newton 的 Hessian $H=\mathbb{E}\big[w(t)\,c_{\text{out}}(t)^2\,J_F^\top J_F\big]$（$J_F=\partial F_\theta/\partial\theta$）会被少数几个 $t$ 主导、把其余饿死。EDM[[4]][r4] 挑 $c_{\text{in}},c_{\text{out}},c_{\text{skip}}$ 让每个 $t$ 上输入、目标都是单位方差、且 $w(t)c_{\text{out}}^2$ 拉平成常数，$H$ 对 $t$ 的依赖就被压平了。噪声档之间是解耦的回归任务，最优 preconditioner 天然就是对角（标量）的，压根不会冒出耦合坐标的稠密矩阵——这也是它看着不像矩阵的原因。严格说，这是 ML 里"把输入/输出 normalize 一下"那一路的 preconditioning（和你为什么要标准化特征、上 BatchNorm 同源），确实把 GN 系统的 $\kappa$ 压了下去，但不是 Krylov solver 里那种耦合坐标的 $M$。

## solver 超参：推理时的旋钮

有意思的是，训练时我们庆幸摆脱了离散求和，推理时却又不得不把连续ODE离散回有限步——只不过这次离散化是我们**主动、可控**地挑的。上一篇最后停在：RF的反向过程是一个（probability-flow）ODE，把它的解析解

$$
\frac{\boldsymbol{x}_t}{\sigma_t}-\frac{\boldsymbol{x}_s}{\sigma_s}=\int_{\lambda_s}^{\lambda_t}e^{\lambda}\hat{\boldsymbol{x}}_\theta\,d\lambda
$$

用一阶Euler去离散，就得到DDIM。但"用Euler解这个积分"里其实藏了好几个还没定的超参，挨个看。

### 步点放哪：discretization schedule

给定NFE预算 $N$，我们要在 $[0,1]$ 上挑 $N{+}1$ 个节点 $1=t_0>t_1>\dots>t_N\approx0$。这就是推理版的"$t$ 怎么撒"——和训练时如出一辙，只不过训练撒的是连续分布 $\pi(t)$，推理撒的是有限个离散步点。最朴素的两种：uniform-$t$（$t_i=1-i/N$，最省事）和uniform-$\lambda$（在log-SNR上等距；既然上一篇那个积分本来就是对 $\lambda$ 写的，等距 $\lambda$ 才是"配得上积分"的那种均匀）。

而现代模型（Flux、Wan这批）实际怎么排，其实就一句话：**uniform-$t$ 网格 + 一个shift**，也就是上一节那个 $t'=\frac{st}{1+(s-1)t}$。区别只在这个 $s$ 从哪来：

- **Wan**[[5]][r5] 用**静态**shift：按分辨率手动设一个标量（官方推荐480p取 $s\approx3$、720p取 $s\approx5$），整条采样一把梭到底。
- **FLUX**[[6]][r6] 用**动态**shift：把 $\mu=\log s$ 设成图像token数（序列长度 $L$）的仿射函数，在 $256^2$ 到 $1024^2$ 之间线性插值——分辨率一变，shift自动跟着走，不用手调。

两者骨子里是同一件事：$\lambda'=\lambda-\log s$，把有限的NFE整体往高噪声端挪。RF之前，schedule是一整墙花活（上一篇那张图）；RF之后，就压缩成了"一条直线 + 一个shift标量"。现代扩散的采样schedule，基本就剩这一个旋钮。

> 步点撒法对few-step（$N\le8$）影响巨大，对many-step几乎无所谓——步够多时怎么撒都收敛到同一条轨迹。所以只有当你想省NFE时，这个shift才值得较真。

### 用几阶：Euler 一统天下

Euler是一阶，也是现在事实上的唯一答案。理论上阶数是可调的——Heun（2阶）、RK4，以及上一篇提过的DPM-Solver系列，都是更高阶的选项，靠每步多采几个点估计轨迹曲率来压低单步误差。但RF把轨迹拉直之后，一阶截断误差本就小得可怜，高阶那点收益抵不过每步翻倍的NFE，于是高阶solver在RF时代基本退役了。

> 还有一个更隐蔽的原因让高阶失宠：高阶solver假设网络预测的是**真** velocity 场，"多采点估曲率"才成立。可一旦开了CFG，被解的其实是一个被guidance拧过、并不对应任何真实score的场，高阶的收益直接打折——这也是"Euler + CFG"经久不衰、几乎成了现代扩散标配的原因之一。

### 要不要随机：ODE ↔ SDE

到目前为止解的都是deterministic的PF-ODE。但同一组marginal $q_t$，对应的反向过程其实是一整个**单参数族**[[7]][r7]：一端是PF-ODE（全确定），另一端是把噪声原样加回去的reverse SDE（DDPM那种），中间由一个"每步注入多少噪声"的旋钮连续插值——它们共享同一个score，只是越往SDE那端，多出来的score-correction项越强。最常见的记法是DDIM的 $\eta$[[8]][r8]：$\eta=0$ 是确定性ODE，$\eta=1$ 是ancestral（DDPM）采样，中间连续过渡。

要不要随机本是个实打实的trade-off：SDE那端有**自我纠错**的性质——每步重新注入的噪声能冲掉之前累积的离散化/score误差，NFE给够时样本常常更干净；但它需要更多步，也引入额外方差。ODE那端在few-step时更稳，而且轨迹可逆（做inversion/图像编辑时只能用ODE）。而现代RF（Flux、Wan那批）几乎清一色把这个旋钮拧到了0，老实待在确定性ODE这端——实践里绝大多数就是Euler。随机采样如今更多是few-step蒸馏和一些ancestral trick里才会碰的东西了。

> 严格说一句：CFG的guidance scale **不算** solver超参——它改的是被积的那个场本身（把 $\boldsymbol{v}_\theta$ 换成 $\boldsymbol{v}_\theta^{\text{uncond}}+w\,(\boldsymbol{v}_\theta^{\text{cond}}-\boldsymbol{v}_\theta^{\text{uncond}})$），而不是解ODE的方式。它以及它的一票精修（guidance interval、CFG rescale、zero-terminal-SNR……）值得单开一段甚至一篇，这里先按下不表。

## Reference

1. Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. [arXiv:2403.03206][r1]
2. Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In ICML, 2023. [arXiv:2301.11093][r2]
3. Diederik P. Kingma and Ruiqi Gao. Understanding diffusion objectives as the ELBO with simple data augmentation. In NeurIPS, 2023. [arXiv:2303.00848][r3]
4. Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022. [arXiv:2206.00364][r4]
5. Wan Team, Alibaba. Wan: Open and advanced large-scale video generative models. arXiv preprint, 2025. [arXiv:2503.20314][r5]
6. Black Forest Labs. FLUX.1. 2024. [github.com/black-forest-labs/flux][r6]
7. Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021. [arXiv:2011.13456][r7]
8. Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. [arXiv:2010.02502][r8]

[r1]: https://arxiv.org/abs/2403.03206
[r2]: https://arxiv.org/abs/2301.11093
[r3]: https://arxiv.org/abs/2303.00848
[r4]: https://arxiv.org/abs/2206.00364
[r5]: https://arxiv.org/abs/2503.20314
[r6]: https://github.com/black-forest-labs/flux
[r7]: https://arxiv.org/abs/2011.13456
[r8]: https://arxiv.org/abs/2010.02502
