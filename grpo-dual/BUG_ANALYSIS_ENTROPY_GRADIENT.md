# 🐛 严重 BUG: Entropy 梯度符号错误

## 问题定位

**文件:** `src/grpo/trainer.py`
**位置:** 第 3172 行和第 3186 行
**影响:** 完整模式（Reward-only CAGrad）中，entropy bonus 的梯度符号**反了**，导致模型**降低** entropy 而不是**提高** entropy！

---

## 详细分析

### 目标函数

GRPO 的目标是**最大化**：
```
J = E[advantage × log_prob] - β × KL + α × Entropy
```

转换为**最小化**问题（PyTorch 的 optimizer 最小化 loss）：
```
L = -E[advantage × log_prob] + β × KL - α × Entropy
```

梯度应该是：
```
∇L = -∇(E[adv×logp]) + β×∇KL - α×∇Entropy
```

---

### 代码实现（完整模式，第 3113-3186 行）

#### 步骤 1: 计算各部分的 loss

```python
# 第 3118-3119 行
reward_loss_f = -(surr[task_mask_f].mean())  # = -E[adv×logp]
kl_loss_f = kl[task_mask_f].mean()           # = E[KL]

# 第 3172 行 ⚠️ 问题在这里！
entropy_loss_f = -sample_entropy[task_mask_f].mean()  # = -E[Entropy]
```

#### 步骤 2: 计算各部分的梯度

```python
grads_reward_f = ∇(reward_loss_f) = ∇(-E[adv×logp]) = -∇(E[adv×logp])  ✓
grads_kl_f = ∇(kl_loss_f) = ∇(E[KL]) = ∇KL  ✓
grads_entropy_f = ∇(entropy_loss_f) = ∇(-E[Entropy]) = -∇Entropy  ⚠️
```

#### 步骤 3: 合并梯度（第 3186 行）

```python
vec_final = vec_reward_merged + beta_f * vec_kl_f + beta_h * vec_kl_h
            - config.ENTROPY_COEF * (vec_entropy_f + vec_entropy_h)
```

展开：
```
vec_final = -∇(E[adv×logp]) + β×∇KL - α×(-∇Entropy)
          = -∇(E[adv×logp]) + β×∇KL + α×∇Entropy  ⚠️ 符号错了！
```

---

### 问题所在

**期望的梯度：**
```
∇L = -∇(E[adv×logp]) + β×∇KL - α×∇Entropy
```

**实际的梯度：**
```
vec_final = -∇(E[adv×logp]) + β×∇KL + α×∇Entropy  ← 最后一项符号反了！
```

**结果：**
- Optimizer 执行 `param -= lr × vec_final`
- 由于符号错误，模型会**增大** Entropy 的相反方向
- 即模型会**降低** Entropy！
- 这直接导致 Entropy 崩溃到 0.005！

---

## 对比：LOW_MEMORY_MODE 是正确的

在 LOW_MEMORY_MODE（第 3068-3109 行），代码直接在 loss 中处理：

```python
# 第 3073 行
loss_fair = (-(surr.mean()) + beta * kl.mean() - config.ENTROPY_COEF * entropy_f) / GRAD_ACC
          = -E[adv×logp] + β×KL - α×Entropy  ✓ 正确！
```

然后直接反传：
```python
loss_fair.backward()
```

这样计算出的梯度是：
```
∇loss_fair = -∇(E[adv×logp]) + β×∇KL - α×∇Entropy  ✓ 正确！
```

---

## 为什么 LOW_MEMORY_MODE 没有被使用？

查看第 3065 行：
```python
if config.LOW_MEMORY_MODE:
    # ...
else:
    # 完整模式（Reward-only CAGrad）
```

当前配置（第 268 行）：
```python
LOW_MEMORY_MODE = False  # ← 默认使用完整模式（有bug）
```

---

## 修复方案

### 方案 A: 修复完整模式的符号（推荐）

**第 3172、3179 行：去掉 entropy_loss 的负号**

```python
# 修复前
entropy_loss_f = -sample_entropy[task_mask_f].mean() / config.GRADIENT_ACCUMULATION_STEPS
entropy_loss_h = -sample_entropy[task_mask_h].mean() / config.GRADIENT_ACCUMULATION_STEPS

# 修复后
entropy_loss_f = sample_entropy[task_mask_f].mean() / config.GRADIENT_ACCUMULATION_STEPS
entropy_loss_h = sample_entropy[task_mask_h].mean() / config.GRADIENT_ACCUMULATION_STEPS
```

然后第 3186 行的合并公式保持不变：
```python
vec_final = vec_reward_merged + beta_f * vec_kl_f + beta_h * vec_kl_h
            - config.ENTROPY_COEF * (vec_entropy_f + vec_entropy_h)  # 保持负号
```

---

### 方案 B: 启用 LOW_MEMORY_MODE（快速workaround）

修改第 268 行：
```python
LOW_MEMORY_MODE = False → True
```

这样会跳过有bug的完整模式，使用正确的 LOW_MEMORY_MODE。

**注意：** LOW_MEMORY_MODE 会牺牲一些 CAGrad 的优势，但至少 entropy 符号是对的。

---

## 影响评估

### 这个 bug 如何导致 Entropy 崩溃？

1. **错误的梯度方向：** 模型被训练成**降低** entropy 而不是**提高** entropy
2. **与 reward 信号冲突：** Reward 可能鼓励多样性，但 entropy bonus 反而惩罚多样性
3. **EOS Suppressor 加剧：** 模型想输出确定的 token（低 entropy），但被 EOS Suppressor 强制续写
4. **恶性循环：** 低 entropy → 高 max_prob (99.9%) → 更低 entropy → ...

### 为什么之前的修复（提升 ENTROPY_COEF）没用？

```python
ENTROPY_COEF = 0.2 → 0.5
```

虽然我们提升了系数，但由于符号错了：
- 原来：模型被推向降低 entropy（力度 0.2）
- 修复后：模型被推向降低 entropy（力度 0.5）← **更强地降低！**
- 结果：Entropy 崩溃得更快！

---

## 紧急行动

### 立即测试

1. **快速验证（方案 B）：** 启用 LOW_MEMORY_MODE
   ```python
   # trainer.py 第 268 行
   LOW_MEMORY_MODE = True
   ```

2. **运行 5-10 步 GRPO 训练**

3. **观察 Entropy 是否恢复：**
   - 如果 Entropy 从 0.005 上升到 0.5+：✅ Bug 确认！
   - 如果仍然 < 0.1：还有其他问题

### 完整修复（方案 A）

如果方案 B 验证成功，再应用方案 A 的代码修复。

---

## 总结

**根本原因：** 完整模式（Reward-only CAGrad）中，entropy 梯度符号错误，导致模型被训练成降低 entropy。

**为什么之前没发现：**
- LOW_MEMORY_MODE 的代码是对的，但默认没启用
- 完整模式的 bug 隐藏在复杂的梯度手术逻辑中
- 注释误导：第 3172 行的注释"负号因为loss中是-entropy"，但这个逻辑在合并时又被反转了一次

**修复优先级：** 🔴 最高（这可能是 Entropy 崩溃的直接原因）

**预期效果：**
- Entropy: 0.005 → 1.0-2.0
- Max prob: 99.9% → 50-80%
- Logit gap: 7-10 → 2-4
