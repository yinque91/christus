# CORE-READOUT-01 正式报告

**结论先行**：在 WORLD02 接球任务中，原 SM-CTRN 的 `h/c` 可以通过一个 265 参数的
线性读出真正改变动作，并显著提高接球率；把 `h/c` 置零后效应逐位消失，把 `h/c`
每步随机置换后方向性效应大幅缩小、覆盖方向回到随机水平。与此同时，**没有证据**
表明该读出能帮助主体适应 evaluator gain 的变化；相反，A1 在 gain 下降阶段的接球率
下降比 A0 更大。这是一个“状态可被读出并用于行动”的肯定结果，也是一个“读出本身
不构成环境变化适应”的否定结果。

- 预注册：revision 2，SHA256 `3285ad0e0f36f72207755620c20100e8666c8845c9c066f51d8d1776f3940dbc`
- 数据：`results/core_readout_formal01/`，20 种子 × 1000 步 × 4 臂 = 80 次运行
- 自动报告：`results/core_readout_formal01/REPORT.md`、`STATISTICS.json`、`RESULTS.csv`
- 完成度：80/80 运行有 summary；0 失败；A0/A2 20/20 种子的
  episodes/caught/catch_rate/reward/cost/body_mae/action_counts 完全相同，
  且事件级检查 20/20 种子 × 1000 步的
  action/reward/outcome/scent_x/ball_x/paddle 完全相同

## 1. 研究问题与设计

问题：SM-CTRN 内部形成的状态，能否真正指导主体认识环境、选择行动，并在环境变化时
帮助主体调整行为？

四臂（唯一差别是读出模式）：

| 臂 | 读出 | 输入 | 训练 |
|---|---|---|---|
| A0 | 关闭 | — | — |
| A1 | 线性读出 | 当前 `[h;c]`（264 维） | 在线 LMS，目标=当前 `scent_x` |
| A2 | 打开但置零 | 全零 | 不更新 |
| A3 | 打开但打乱 | 每步随机置换 264 维 | 同一 LMS |

共同配置：`profile="safe"`、手写 scent 导航先验关闭（`navigation_gain=0`）、
`action_model=false`、`affect none`、`language off`、观察桥 `scent_x` 四臂相同、
20 种子 31100–31119、每运行 1000 步。详见
[`CORE_READOUT_PREREGISTRATION.md`](CORE_READOUT_PREREGISTRATION.md)。

## 2. 接口审计（为什么需要观察桥）

原核心的 `world` 输入只有 10 维，不含球的水平偏移 `scent_x`。线性探测显示，
桥接前 `h`、`c`、`[h;c]` 对 `scent_x` 的 R² 全为负，符号准确率约 0.56：
**不修观察接口，读出注定是零结果**。加一维 `scent_x` 后，A0（读出关闭）的
`[h;c]` 线性探测 R²=0.293、corr=0.868、sign acc=0.640；上一步 `h` 的 corr 仍约 0.80。

同时确认：Practice 在 `core.step` **之前**选动作。因此读出挂在 core.step 之后、
`world.execute` 之前；默认 `action_hook=None`，SAFE 动作路径不变。

详见 [`CORE_READOUT_INTERFACE_AUDIT.md`](CORE_READOUT_INTERFACE_AUDIT.md)。

## 3. 门控（5 种子 × 300 步，正式前）

| 臂 | readout_corr | alignment | steer_signed | catch_rate | override_rate |
|---|---:|---:|---:|---:|---:|
| A0 | 0.000 | — | 0.015 | 0.171 | 0.000 |
| A1 | 0.436 | 0.708 | 0.381 | 0.943 | 0.646 |
| A2 | 0.000 | — | 0.015 | 0.171 | 0.000 |
| A3 | 0.032 | 0.542 | 0.065 | 0.429 | 0.414 |

9 项门控全部通过；进入正式实验。详见
[`CORE_READOUT_GATE_REPORT.md`](CORE_READOUT_GATE_REPORT.md)。

## 4. 正式结果（20 种子 × 1000 步）

### 4.1 每臂均值 ± 样本标准差

| 臂 | catch_rate | steer_signed | steer_accuracy | override_rate | alignment | readout_corr | reward_sum |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.277 ± 0.071 | -0.001 ± 0.018 | 0.130 ± 0.012 | 0.000 | — | 0.000 | -13.698 ± 3.427 |
| A1 | **0.607 ± 0.069** | **0.395 ± 0.060** | 0.605 ± 0.036 | 0.603 ± 0.034 | **0.646 ± 0.021** | **0.526 ± 0.045** | **-3.332 ± 3.447** |
| A2 | 0.277 ± 0.071 | -0.001 ± 0.018 | 0.130 ± 0.012 | 0.000 | — | 0.000 | -13.698 ± 3.427 |
| A3 | 0.329 ± 0.073 | 0.077 ± 0.031 | 0.372 ± 0.031 | 0.445 ± 0.071 | 0.491 ± 0.026 | 0.131 ± 0.028 | -14.915 ± 3.706 |

### 4.2 主检验（配对 vs A0，Holm 校正 3 对比 × 2 终点）

| 对比 | 终点 | 均值差 | 95% CI | Wilcoxon p | Holm p | dz |
|---|---|---:|---|---:|---:|---:|
| A1-A0 | catch_rate | **+0.330** | [0.284, 0.375] | 8.83e-05 | **3.53e-04** | 3.08 |
| A1-A0 | steer_signed | **+0.396** | [0.371, 0.420] | 1.91e-06 | **1.14e-05** | 6.94 |
| A2-A0 | catch_rate | 0.000 | [0.000, 0.000] | 1.0 | 1.0 | — |
| A2-A0 | steer_signed | 0.000 | [0.000, 0.000] | 1.0 | 1.0 | — |
| A3-A0 | catch_rate | +0.051 | [0.007, 0.094] | 0.046 | 0.139 | 0.50 |
| A3-A0 | steer_signed | +0.078 | [0.061, 0.094] | 3.82e-06 | 1.91e-05 | 2.00 |

### 4.3 方向对齐（A1 覆盖步）

- 合并：7791/12063 = **0.646** 朝球；二项检验 p = 1.09e-228。
- 20/20 个种子的 alignment > 0.5；逐运行 0.597–0.689。
- A1 的读出在 60.3% 的步覆盖了 Practice 的基准动作。
- A3 合并覆盖方向：4358/8901 = **0.490**；单侧“>0.5”二项检验 p=0.976，
  双侧 p=0.051。即 A3 的覆盖方向没有系统朝球，接近随机（略低于 0.5）。

### 4.4 H1/H2 判定

- **H1 支持**：A1 的读出确实改变动作（override_rate=0.603），且改变方向显著朝球
  （alignment=0.646，20/20 种子 >0.5）。A2 完全为零，A3 的覆盖方向回到随机
  （alignment=0.491）。
- **H2 支持**：A1 接球率 +0.330（95% CI 0.284–0.375），Holm p=3.5e-4；
  reward 改善 +10.37（95% CI 8.06–12.65）。A2 与 A0 是事件级相同的动作/结果
  （20/20 种子 × 1000 步的 action/reward/outcome/scent_x/ball_x/paddle 完全一致；
  事件日志的 readout 字段不同，因为 A2 仍记录零输入模块）；
  A3 的接球率 +0.051 在 Holm 校正后不显著（p=0.139）。

## 5. 打乱对照的残余效应（必须如实报告）

A3 并非精确零：

- A3 `steer_signed` = 0.077，A0 = -0.001，差 0.078；Holm p=1.9e-05，统计上可检测。
- 但 A3 的 override 方向 `alignment` = **0.491**，与 0.5 随机水平一致；
  A1 为 0.646。
- A3 的 `readout_corr` = 0.131，约为 A1（0.526）的 25%。

解释：每步随机置换保留了状态的范数/总和等置换不变量，线性读出仍能学到一点
与目标幅值/均值相关的弱信号，因此产生一个小的非特异性偏置；但**覆盖方向不再
跟随球位**。A3 的 0.078 在预注册的“与 A0 差 ≤0.10”等价界内，因此预先规定的
特异性标准通过；但不能宣称“打乱后精确为零”。正确表述是：

> 屏蔽（A2）使效应逐位消失；打乱（A3）移除了绝大部分状态特异性方向效应
> （0.396 → 0.078，约 80%），覆盖方向回到随机；仍有一个小的置换不变量残余。

## 6. 环境变化（H3）：否定结果

WORLD02 的 evaluator gain 在 401–750 步降为 0.45，751 步恢复 1.0。

| 阶段 | A0 catch | A1 catch | A1-A0 | A1 steer_signed | A0 steer_signed |
|---|---:|---:|---:|---:|---:|
| normal 1–400 | 0.303 | 0.734 | +0.432 | 0.332 | 0.004 |
| changed 401–750 | 0.275 | 0.512 | +0.237 | 0.546 | -0.008 |
| return 751–1000 | 0.248 | 0.527 | +0.280 | 0.288 | 0.003 |

A1 从 normal 到 changed 的接球率下降 **0.222**；A0 只下降 **0.027**。配对差
（A1 下降 − A0 下降）= **+0.195**，95% CI [0.087, 0.297]，Wilcoxon p=**0.0045**
（次要配对检验）。
也就是说，A1 对 gain 变化**更敏感**，而不是更鲁棒；H3（A1 下降更小/恢复更快）
**不被支持**。
A1 的优势在三个阶段都仍然存在（+0.24 到 +0.43），但它来自持续的朝球偏置，
不是对 gain 的适应。读出目标是球偏移，不包含 gain/身体增益，因此这个否定结果
与设计一致，不应被主结果掩盖。

## 7. 解释与边界

**支持的 claim**：

- 原 SM-CTRN 的 `h/c`（在接入一维观察后）携带可线性解码的球偏移信息。
- 一个小型线性读出可以使用该信息改变动作，并显著提高任务表现。
- 该效应依赖真实 `h/c`：置零后逐位消失；随机置换后方向性效应大部分消失、
  覆盖方向回到随机。

**不支持的 claim**：

- 不能说核心“自主形成概念”或“从奖励中学会策略”。读出是观察预测的线性解码器
  加固定符号映射，不是策略梯度/actor-critic。
- 不能说读出带来 gain 变化适应；H3 是否定的。
- 不能说 SAFE 的总体性能；实验关闭了手写导航先验，A0 是匹配基线，不是 SAFE 的
  生产性能。
- 不能把 body_mae 当作纯学习指标：A1/A3 更多使用左右动作，身体预测误差天然高于
  很少移动的 A0；body_mae 只作诊断。

## 8. 限制

1. 单一任务（WORLD02 接球）、单一环境、20 种子 × 1000 步；没有跨任务/跨世界迁移。
2. 读出是线性、监督式（目标为可观测 `scent_x`），不是端到端奖励学习。
3. A3 有小的置换不变量残余；A3 不是精确零对照。
4. H3 低功效且结果为负；阶段回合数少（每段 6–10 个 episode/运行）。
5. 未测试真实摄像头、长期运行、断电恢复、社会/语言/情绪整合。
6. 观察桥是一维 `scent_x`；没有证明核心能从原始像素自主发现该变量。
7. 未做读出权重的消融/可视化；未证明读出使用了特定的“概念”维度，只证明了
   线性可解码性。
8. 正式运行后只修改了分析脚本 `analyze_core_readout.py`（增加 H3 配对检验与
   自动判读字段）；运行时代码 `christus/core_readout.py`、`christus/runtime.py`、
   `christus/exp_core_readout.py` 的 SHA256 与 `MANIFEST.json` 一致，数据未受影响。
   分析脚本的 hash 变化记录在此，供复核。

## 9. 复现与文件

```powershell
# 门控（已完成）
.\run_core_readout.ps1 -Out results/core_readout_gate01 -Seeds 31100:31105 -Steps 300 -Gate

# 正式（已完成）
.\run_core_readout.ps1 -Out results/core_readout_formal01 -Seeds 31100:31120 -Steps 1000 -Analyze
```

关键文件：

- 预注册：[`CORE_READOUT_PREREGISTRATION.md`](CORE_READOUT_PREREGISTRATION.md)
- 修订记录：[`results/core_readout_prereg/AMENDMENT_LOG.md`](results/core_readout_prereg/AMENDMENT_LOG.md)
- 接口审计：[`CORE_READOUT_INTERFACE_AUDIT.md`](CORE_READOUT_INTERFACE_AUDIT.md)
- 实现说明：[`CORE_READOUT_IMPLEMENTATION.md`](CORE_READOUT_IMPLEMENTATION.md)
- 门控报告：[`CORE_READOUT_GATE_REPORT.md`](CORE_READOUT_GATE_REPORT.md)
- 正式数据：[`results/core_readout_formal01/`](results/core_readout_formal01/)
- 自动统计：[`results/core_readout_formal01/STATISTICS.json`](results/core_readout_formal01/STATISTICS.json)
- 逐运行表：[`results/core_readout_formal01/RESULTS.csv`](results/core_readout_formal01/RESULTS.csv)
- 失败/负结果：[`CORE_READOUT_FAILURES.md`](CORE_READOUT_FAILURES.md)

## 10. 一句话回答

**能，但有限定**：`h/c` 确实可以被一个小读出用来选择动作，且这种作用依赖真实状态；
但当前读出只提供了“朝球”的方向信息，没有带来环境 gain 变化下的适应。SAFE 仍是
默认配置，实验性读出不是默认，也不宣称核心已经完成自主概念或策略学习。
