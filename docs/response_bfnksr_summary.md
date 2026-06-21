# BF-NKSR Response Attempts Summary

本文档整理这轮 BF-NKSR/显式 Ritz response 开发中做过的主要理论尝试、代表性数值结果和当前问题判断。数值以远端 GPU worker 上保存的 H/He 原子 smoke/diagnostic runs 为主。

## 目标标尺

- H 原子第一 bright 激发能：精确值 `0.375 Ha`。
- H 后续 Rydberg bright 根：约 `0.44444 Ha`, `0.46875 Ha`, `0.48000 Ha`。
- He 第一 bright singlet 激发 `1s^2 1S -> 1s2p 1P`：实验约 `0.780 Ha`。
- 当前最可信的 reported value 是 independent scalar certification，而不是直接 matrix Ritz root：

  ```text
  omega(c) = (c^T H_cert c) / (c^T S_cert c)
  ```

  其中 `c` 在 fit sample 上确定，`H_cert/S_cert` 在独立 sample 上重新评估。

## 主要路线和结果

| 阶段 | 理论重点 | 代表效果 | 暴露的问题 |
| --- | --- | --- | --- |
| 原始 NN response head | 在 FermiNet ground 上加 response heads，解广义 Ritz 问题 `H c = omega S c` | H 有时可接近 `0.37-0.38 Ha`，He 经常漂到 `~0.9-1.0 Ha` 或出现假低根 | NN head 太自由，训练 loss 降不等价于 Ritz 子空间变好 |
| Ritz warmup + residual finetune | 先用 Ritz warmup，再用 residual loss 训练 candidate heads | candidate loss 可以下降 | 下降方向常是弱方向、共线方向或小范数方向，不稳定降低 certified excitation |
| residual candidate + validation gate | 做候选生成、held-out validation、接受/拒绝闭环 | gate 能拒绝 noisy improvement | 解决了误收问题，但没有产生稳定更好的 He carrier |
| source prefactor/direct response | 让 response 更贴近 dipole source，或直接训练响应函数 | H 有改善，He 仍不稳定 | 形式合理，但 NN 优化和 metric 病态仍在 |
| CIS/CAS/CASSCF seed | 用 PySCF/CASSCF 给初始激发态或 teacher | CAS seed 能提供物理参考 | seed 只是初始；蒸馏到 NN 后不能可靠保持 CAS 波函数或改善 certified value |
| SA-CASSCF/CAS external | 直接用 spin-pure CAS teacher 或外部 CAS basis | 有助于理解 dark/bright root 排序 | 有 Gaussian basis/active-space/root-ordering bias；插入 BF-NKSR 后仍未突破 He |
| Krylov teacher | 用 `q, Hq, H^2q, ...` 生成 Krylov carrier | matrix root 有时看起来接近 | independent certification 后很多 improvement 不成立 |
| retained/dressed teacher | 保留 teacher carrier，再加 dressing 或 retained modes | He 仍漂移 | 问题不是缺 seed，而是 carrier 空间、metric 和采样稳定性 |
| sampling 修复 | source sampling、bright-influence sampling、pair resampling、leverage mixture | 方差诊断更可靠 | 对病态 carrier，leverage sampling 会追尖峰；ESS 低，反而更差 |
| cross-fitted scalar certification | fit/cert sample 分离，固定 Ritz vector 后独立评估 Rayleigh quotient | 关键进展：能暴露假 improvement | 证明剩余 He 误差主要是 bias/ansatz 问题，不只是 bootstrap noise |
| 显式 Ritz carrier | 去掉 NN/CAS/Krylov seed，直接用 source-adapted explicit carrier `C_mu = Q0[Phi_mu G_q]` | H 基本成功，He 稳定到 `~0.795 Ha` | 当前最靠谱，但 He 仍高于实验 bright `~0.016 Ha` |
| 扩展 He carrier | spectator-relaxed `s_n p_m + p_m s_n`、扩展 geminal、`(p d + d p)_{L=1}` | 没突破，部分更差 | 简单加基会引入病态低根，不等于更好物理空间 |

## 关键数值

### H 原子

最新 sanity check 使用 22 个显式 p carrier：

```text
carriers = 22
retained = 11
matrix first bright root = 0.3703855607 Ha
independent scalar cert = 0.3779778396 +/- 0.0024862485 Ha
target = 0.375 Ha
first few roots = 0.370386, 0.443161, 0.468326, 0.480022, 0.486068 Ha
```

较小 report run 使用 9 个显式 p carrier：

```text
matrix first bright root = 0.3796452631 Ha
bootstrap mean = 0.3790361351 Ha
bootstrap SE = 0.0044512334 Ha
```

判断：H 单电子情形说明 weak-form Ritz、ground projection、explicit carrier 和 scalar certification 主流程基本成立。

### He 最稳 baseline

135 个 `sp + F12` 显式 carrier，fit 32k/cert 32k：

```text
carriers = 135
retained = 8
raw first root = 0.5240368080 Ha
first bright matrix root = 0.7902714399 Ha
independent scalar cert = 0.7955719218 +/- 0.0029807776 Ha
target bright ~= 0.780 Ha
first few roots = 0.524037, 0.790271, 0.951323, 1.187743, 2.522997 Ha
weights = 1.798e-01, 8.530e+00, 1.529e+01, 1.552e+01, ...
```

判断：这是目前最可信的 He 结果。raw low root 权重小，更像 dark/weak root 或病态低根；bright root scalar-certified 后稳定在 `~0.795-0.796 Ha`，说明还有约 `0.015-0.016 Ha` 的 ansatz bias。

### He spectator-relaxed s 扩展

270 个 carrier，加入 `s_laguerre_orders = [0, 1]`：

```text
carriers = 270
retained = 13
raw first root = 0.5979442220 Ha
first bright matrix root = 0.7938049969 Ha
independent scalar cert = 0.7940335465 +/- 0.0082608216 Ha
```

判断：没有实质改善，只是 certified uncertainty 变大。

### He 扩展 geminal

216 个 carrier，加入更多 geminal，例如 `exp(-gamma r12)` 和 `r12`：

```text
carriers = 216
retained = 8
raw first root = 0.6898447679 Ha
first bright matrix root = 0.8400542286 Ha
independent scalar cert = 0.9014893974 +/- 0.0460531942 Ha
```

判断：明显变差。扩展 geminal 让矩阵根和 independent certification 严重不一致。

### He `(p d + d p)_{L=1}` 扩展

实现方式：用实张量表示 d carrier。令 p 是 vector carrier `P_j`，d 是 symmetric traceless tensor `D_ij`，则 L=1 的 z 分量用

```text
sum_j D_zj(1) P_j(2) + P_j(1) D_zj(2)
```

270 个 carrier，bright gate `0.2`：

```text
carriers = 270
retained = 15
raw first root = 0.4566196476 Ha
first selected bright matrix root = 0.8414420125 Ha
independent scalar cert = 0.8070586029 +/- 0.0350176347 Ha
```

如果 gate `0.05`，低根 `0.4566 Ha` 被选中，但 independent scalar cert 是 `0.1937 +/- 0.587 Ha`，是明显假根。

再加 bright-influence sampling：

```text
carriers = 270
retained = 19
first selected bright matrix root = 0.8179487832 Ha
independent scalar cert = 1.0405856672 +/- 0.0227723425 Ha
proposal ESS fraction = 0.079907
mixture weights: source ~= 0.114, leverage ~= 0.886
```

判断：pd 扩展当前没有解决 He 问题，反而引入强病态和尖峰 leverage influence。

## 当前判断

1. H 的成功说明主流程不是错的。单电子情形下，显式 Ritz carrier 可以复现第一激发和后续 Rydberg 根。
2. He 的 `~0.795 Ha` plateau 更像真实 ansatz/carrier bias，而不是 bootstrap 噪声。cross-fitted scalar certification 后，这个结论更清楚。
3. NN fine tune 不是当前突破口。loss 可以下降，但下降方向不一定是能降低 Rayleigh quotient 的有效独立 carrier。
4. CAS/CASSCF seed 不是根本解。它能提供参考，但有限 Gaussian basis 和 active space 会带来 bias；蒸馏或 dressing 后仍要通过 weak Ritz/scalar cert，之前没有稳定突破。
5. sampling 是必要技术，但不是唯一瓶颈。对 135-carrier baseline，source Sobol 已经能给稳定 certification；对 pd 扩展，leverage/bright-influence sampling 反而暴露出非常尖的影响函数。
6. 不能简单加更多基。spectator s 扩展没改善，geminal 和 pd 扩展更差。后续需要设计条件数可控、物理正确的 He bright P-sector carrier family，而不是继续堆自由度。

## 目前最可信流程

当前最靠谱版本是：

```text
source-adapted explicit Ritz carrier
+ fixed whitening
+ bright root selection
+ independent scalar Rayleigh certification
```

这一流程对 H 基本成功；对 He 给出稳定但偏高的 bright excitation，当前最好约：

```text
He bright = 0.7955719218 +/- 0.0029807776 Ha
experiment ~= 0.780 Ha
error ~= +0.0156 Ha
```

后续如果继续推进，重点应放在 He 显式 carrier 的物理构造和正交化/条件数控制，而不是继续堆 NN fine tune、CAS seed 或 validation gate。
