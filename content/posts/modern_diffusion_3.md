# consistency distillation（上）

## Recap: 2023年的(discrete) Consistency Models (dCM)

TL; DR：它是一个 v-pred、x-loss，然后**在同一条 PF-ODE 轨迹的相邻两点上强制自洽**。

具体地，dCM 在 $[\epsilon, T]$ 上取一串离散节点 $\epsilon=t_1<t_2<\dots<t_N=T$，学一个 consistency function $\boldsymbol f_\theta(\boldsymbol x_t, t)$，要求它把同一条 PF-ODE 轨迹上的任意点都映回同一个起点（干净数据），并带边界条件 $\boldsymbol f_\theta(\boldsymbol x_\epsilon, \epsilon)=\boldsymbol x_\epsilon$。训练目标就是让轨迹上相邻两节点的预测对齐：

$$
\mathcal L_{\text{dCM}}=\mathbb E_{n,\,\boldsymbol x_0,\,\boldsymbol\epsilon}\Big[\lambda(t_n)\,d\big(\boldsymbol f_\theta(\boldsymbol x_{t_{n+1}},t_{n+1}),\ \boldsymbol f_{\theta^-}(\hat{\boldsymbol x}_{t_n},t_n)\big)\Big]
$$

其中 $\hat{\boldsymbol x}_{t_n}$ 是从 $\boldsymbol x_{t_{n+1}}$ 出发、拿预训练 teacher 跑一步 ODE solver 得到的相邻（更干净）的点；$\theta^-=\operatorname{stopgrad}(\operatorname{EMA}(\theta))$ 是 target 网络；$d$ 是个距离度量（$\ell_2$ / LPIPS / 后来 iCM 用的 Pseudo-Huber）。所谓 "x-loss"，就是这个 $d$ 直接度量在 clean-$\boldsymbol x$ 空间上——两端各自先把网络的 v 输出换算成 $\boldsymbol x_0$ 预测，再比。轨迹相邻两点一旦处处对齐，$\boldsymbol f_\theta$ 就等于"一步跳到 $\epsilon$"，于是推理时 1 次 NFE 就能出图。

详细的介绍见《家用扩散模型》。

## Continuous-time CMs (sCM)

## Consistenty Trajectory Models (CTM)

## MeanFlow

## Improved MeanFlow (iMF)


## References