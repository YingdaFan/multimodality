# FlowMatching + Diffusion Coupled Model

为 hourly streamflow imputation 设计的 **FM + Diffusion 双向耦合**模型。
设计灵感来自概率流 ODE 的左右两边——FM (velocity field) 和 Diffusion
(score field) 通过数学恒等式 `u(x,t) = f(x,t) − ½g²(t)·s(x,t)` 在训练
期间共享梯度，达到"文明 6 科技-文化协同"式的相互赋能。

> "FM 做得好如何帮助 diffusion?Diffusion 做得好又如何帮助 FM?"

本子包是 LSTM Stage-1 的**可替代实现**，与 `lstm/` 主目录解耦，便于
后续提取为独立方法。

---

## 1. 架构

```
        x (B,L,F)               y_true (B,L,1)
            |                          |
            +-----+---+ NaN-mask -----+
                  |   v
                  +-> y0_filled = where(mask, y_true, 0)
                  |
                  v
   t ~ U[ε,1−ε]  →  perturb:  y_t = α(t)·y0 + σ(t)·ε
                       |        |
                       v        v
              ┌────────────┐  ┌────────────┐
              │ score_net  │  │  vel_net   │   两个独立 DLinear backbone
              │  s_pred    │  │  u_pred    │
              └─────┬──────┘  └─────┬──────┘
                    |                |
   ┌────────────────┼────────────────┼────────────────┐
   |                |                |                |
   v                v                v                v
 score_loss      vel_loss        ode_loss       (NaN-masked, 见 §2)
 (B,L,1) MSE    (B,L,1) MSE   |u−f+½g²s|² 耦合
                                两个网络共享梯度
   \                /
    └─ total = score + vel + λ_ode · ode  →  backward()
```

**组件**:
- `dlinear_ts.py` — DLinear backbone（FiLM 时间编码 + series decomp）
- `coupled_fmdiff.py` — `CoupledFMDiff` wrapper + VP-SDE schedule
- `torch_utils.py` — 训练循环（带 component logging + lambda_ode ramp）
- `predict.py` — 反向 ODE sampling 推断 + npy 输出
- `base.py` — 端到端 runner

---

## 2. 与 LSTM 的接口差异

| 维度 | LSTM | CoupledFMDiff |
|---|---|---|
| 训练 forward | `y_pred = model(x)` | `loss = model.compute_loss(x, y_true)` |
| Loss target | y_true 直接监督 | score / vel / ODE 三分量（生成式训练）|
| 推断 forward | `model(x)` 一步 | 反向 ODE 积分 50 步（在 forward 内部）|
| NaN 处理 | `rmse_masked` 直接 mask | 先 fill 0 不传 NaN，再 mask 三分量 loss |
| 输出格式 | (N,L,1) normalized | 一致 |

`compute_loss` 内 NaN-mask 实现:
```python
mask = ~torch.isnan(y_true)
y0 = torch.where(mask, y_true, torch.zeros_like(y_true))   # NaN→0 占位
n_valid = mask.sum().clamp(min=1)
# … 三个分量都用 (loss_term * mask).sum() / n_valid 聚合
```

---

## 3. 快速测试

### A. Import 烟雾测试（< 5 秒）

```bash
cd /home/yif47/river-dl/temporal/imputation/lstm
python3 -c "
import sys; sys.path.insert(0, '.')
from Model.flowmatchingdiffusion import (
    CoupledFMDiff, train_torch_fmdiff, predict_fmdiff_from_io_data
)
print('all 13 public symbols import OK')
"
```

### B. Loss 正确性 self-test（< 10 秒，跑 10 step 看梯度）

```bash
cd /home/yif47/river-dl/temporal/imputation/lstm
python3 -u << 'PY'
import sys; sys.path.insert(0, '.')
import torch, numpy as np, torch.optim as optim
from Model.flowmatchingdiffusion import CoupledFMDiff

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

# 4-batch, 24-len, 50-feat, 含 ~40% NaN
B, L, F = 4, 24, 50
x = torch.randn(B, L, F, device=device)
y = torch.randn(B, L, 1, device=device)
y[torch.rand_like(y) < 0.4] = float('nan')

model = CoupledFMDiff(input_dim=F, seq_len=L, sample_steps=10, seed=42).to(device)
opt = optim.AdamW(model.parameters(), lr=1e-3)

for step in range(10):
    opt.zero_grad()
    loss = model.compute_loss(x, y)
    assert torch.isfinite(loss), f"step {step}: loss not finite"
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
    opt.step()
    print(f"step {step}: loss={loss.item():.4f}")

# all-NaN edge: 应返回 0
print("all-NaN loss:", model.compute_loss(x, torch.full_like(y, float('nan'))).item())
print("inference output range:", model(x).min().item(), model(x).max().item())
PY
```

### C. 端到端 mini run on basin 03505550（约 5 分钟）

前提：已经跑过 `data_processing/preprocess_perseg_aligntime_camelsh.py 03505550`
生成 `data_processing/data/prepped.npz`。

```bash
cd /home/yif47/river-dl/temporal/imputation/lstm/Model/flowmatchingdiffusion
CUDA_VISIBLE_DEVICES=1 python -u base.py
```

输出位置：`lstm/output_fmdiff/`
- `finetuned_weights.pth` — 最佳 val 时刻的 checkpoint
- `train_log.csv` — 每 epoch 的 4 个 loss 分量 + lambda_ode + 时间
- `preds/{trn,val,tst}.npy` — 反向 sample 出的预测

### D. 调整训练超参

编辑 `base.py` 顶部的 `DEFAULT_CONFIG`，或者放一个 `config_fmdiff.yml`
在该目录下覆盖默认值：

```yaml
# config_fmdiff.yml
ft_epochs: 50
finetune_learning_rate: 5e-4
sample_steps: 100        # 推断更精细
lambda_ode_ramp_frac: 0.1
sample_eval_every: 5     # 每 5 epoch 跑一次 sample-based RMSE
```

---

## 4. 当前实测结果（basin 03505550, seq_len=168, 28 epoch early-stopped）

```
RMSE (normalized)    NSE      vs LSTM 同设置
trn  1.24            -0.55    LSTM RMSE ≈ 0.88
val  1.35            -0.46    LSTM 平台
tst  1.23            -0.52
```

**结论：当前实现的预测质量明显低于 LSTM**。Loss 三分量分析显示 score
和 ode 都正常下降，但 vel_loss 卡在 ~2.5 floor，sample 出的 y_pred
基本是 N(0, 1) 的 noise-like 分布，没建立 X→Y 的有效条件。

---

## 5. 已知结构性限制（root causes，**非超参问题**）

> 重要：这些问题用 sample_steps、log1p、epoch 数等表面修补**解不了**，
> 需要架构层面的改造。

### 问题 1: 模型不知道哪些 y 位置是观测、哪些是缺失 ❌

当前 NaN 处理:
```python
y0 = torch.where(mask, y_true, 0)
y_t = alpha * y0 + sigma * eps    # 加噪后丢失"原本是不是 NaN"信息
score_net(x, y_t, t)              # 网络只看到 y_t, 不知道 mask
```

模型无法区分：
- 观测位置 `y_t = α·y_true + σ·ε` (信号+噪声)
- 缺失位置 `y_t = α·0 + σ·ε = σ·ε` (纯噪声，但模型可能把它当作"真值正好是 0")

正经做法：把 `mask` 作为**额外输入通道**喂给 backbone：
```python
score_net(x, y_t, mask, t)     # 多一个二元通道告诉网络哪里可信
```

参见 CSDI (Tashiro et al. NeurIPS 2021) 的 "conditional mask" 设计。

### 问题 2: VP-SDE 的 σ²-weighted score loss 偏向高噪声区域 ❌

```python
score_loss = ((s_pred - s_target).pow(2) * sigma.pow(2) * mask).sum() / n_valid
                                              ^^^^^^^^^^^^
                                              σ² weighting
```

这个加权可以从噪声预测 MSE 推导得到（DDPM 流派标准做法）。
但意味着:
- t → 1 (远离数据，prior 区域)：σ 大 → loss 大 → 训练充分
- t → 0 (靠近真实数据)：σ 小 → loss 小 → **训练不足**

Sampling 反向积分**结束**在低 t (数据空间)，恰好是模型训练最弱的地方
→ 输出走样。

正经做法：换成 EDM (Karras et al. NeurIPS 2022) 风格的均衡 weighting：
```
weighting(σ) = (σ² + σ_data²) / (σ · σ_data)²
```

### 问题 3: vel target 内禀方差导致 vel_loss 有下限 ❌

```python
u_target = alpha_dot(t) * y0 + sigma_dot(t) * eps    # eps 是每 step 重抽的随机
```

每 step 重新抽 (t, ε)，vel target 本身随机。即使模型完美预测条件期望
`E[u_target | x, y_t, t]`，残差 = `Var[u_target | x, y_t, t] > 0`。

实测 vel_loss 卡在 ~2.5 大概率就是这个 floor。

正经做法：
- (a) 用 `v-prediction`（Salimans & Ho 2022）替代直接 vel：将 target
  re-parameterize 为 `v = α·ε − σ·y₀`，方差小一个量级
- (b) 或 dual-flow + posterior matching：直接 minimize KL divergence
  避免高方差 sampling target

### 问题 4: DLinear backbone 表达力不足 ❌

DLinear 的两个关键操作:
1. **per-channel temporal mixing** (Linear(L, L), 每个 channel 独立)
2. **per-time channel projection** (Linear(F+1, 1))

这能表达：
- "本通道 t1 的值 → 本通道 t2 的值" (per-channel lag)
- "在 t 时刻 多通道线性组合 → 输出" (per-time mix)

**不能表达**：
- "channel A 在 t1 的值 + channel B 在 t2 的值 → output 在 t3 的值"
  这种**跨通道+跨时间的非线性相互作用**。

但 hourly streamflow 恰好需要这种依赖：例如 "12 小时前的 rainfall +
6 小时前的 temperature → 当前的 flow rise"。

正经做法：
- 1D Transformer / 1D U-Net + cross-attention 替代 DLinear
- CSDI 用 Transformer 在 (channel × time) 网格上做 self-attention，
  正是为了捕捉这种 cross dependency

---

## 6. 路线图：如何治本（按优先级）

### Phase 1: 最小可行修改（约 1 周）

1. **加 imputation mask 通道**（解决问题 1）
   - 修改 `DLinearBackboneTS.forward`：增加 `mask` 输入参数
   - 把 `mask` 作为额外通道 concat 到 fused
   - 修改 `_CoupledCore` 和 `CoupledFMDiff.compute_loss` 把 mask 传下去

2. **换 EDM weighting**（解决问题 2）
   - 在 `compute_loss` 里引入 `loss_weighting(σ)` 函数
   - 提供 EDM、VP、SNR-uniform 几个选项

### Phase 2: 模型表达力升级（约 2-3 周）

3. **Transformer backbone 替代 DLinear**（解决问题 4）
   - 新建 `transformer_backbone_ts.py`
   - 1D temporal Transformer + cross-channel attention
   - 保持 (cov_dim, target_dim, seq_len, t) 接口不变

4. **v-prediction 重参数化**（解决问题 3）
   - 在 `_CoupledCore` 加一个 `predict_v(x, y_t, t)` 接口
   - 在 `compute_loss` 里把 vel 目标换成 v-pred

### Phase 3: 高级耦合 + 论文实验

5. **Mixture-of-velocities head + score-gated mixing**
   - vel_net 输出 K 个候选 velocity field + K 个 gate
   - 在 score 范数大的区域（多模态）gate 鼓励发散
   - 这是用户 "civ6 spiral" 设计中 "diffusion 帮 FM 摆脱塌缩" 的核心
     落地

6. **Predictor-Corrector 推断**
   - 在 sample loop 每 step 后加 Langevin corrector
   - `x ← x + ½·g²·s + g·sqrt(dt)·N(0,1)`
   - 进一步降低 sample 方差

7. **多尺度耦合**（hourly + daily）
   - 一个分支 seq_len=168 学 hourly fine
   - 一个分支 seq_len=720 (30 天) 学 daily trend
   - 用 hourly aggregate ≈ daily 作为一致性约束

---

## 7. 输出对比基线

LSTM + 现有 preprocess pipeline 在 basin 03505550 上：
```
seq_len=168, hidden=20, weight_decay=0.001, dropout=0.2
30 epoch:  train_loss plateau ≈ 0.88 (RMSE in normalized space)
           val_loss ≈ 0.95-1.0
           tst_loss ≈ 0.55
NSE in y space: 约 0.4-0.5
```

目标：FMDiff 修完上述 4 个根源问题后，trn RMSE < 0.5、val NSE > 0.6
(单 basin 单实验，不限制泛化)。

---

## 8. 文件状态

| 文件 | 状态 | 备注 |
|---|---|---|
| `dlinear_ts.py` | ✓ 完成 | DLinear 主干，待升级 Transformer |
| `coupled_fmdiff.py` | ✓ 完成（含 NaN-mask）| 待加 imputation mask 输入通道 |
| `torch_utils.py` | ✓ 完成 | 三分量 logging + lambda_ode ramp |
| `predict.py` | ✓ 完成 | 反向 ODE sample + npy 输出 |
| `base.py` | ✓ 完成 | 端到端 runner，可直接 `python base.py` |
| `__init__.py` | ✓ 完成 | 暴露 13 个公开符号 |
| `README.md` | ✓ 本文件 | 设计 + 限制 + 路线图 |

`lstm/` 主目录 (`torch_utils.py`, `model.py`, `predict.py`, `base.py`)
**完全未动**，本子包独立可拆。
