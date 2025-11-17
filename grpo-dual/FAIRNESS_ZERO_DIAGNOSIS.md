# 🔬 Fairness信号为0 - 全面诊断报告

**问题**: Step 5显示 `F std=0.000, rel=0.000` - Fairness信号完全死亡

**影响**: 50%的group产生零梯度，Fairness任务无法学习

---

## 📋 7大可能原因及诊断方案

### ✅ **原因1: 模板崩溃（已确认）**

**现象**: 所有4个候选完全相同
```
Answer: B
Justification: The context does not provide sufficient information to determine this.
```

**证据**:
- Step 5所有候选19 tokens，完全一致
- 熵值0.012-0.293（严重偏低）

**已应用修复**:
- ENTROPY_COEF: 1.5 → 2.5
- MIN_NEW_TOKENS: 15 → 30
- TEMPERATURE: 0.9 → 1.0

**状态**: ✅ 已修复（待验证）

---

### ❓ **原因2: Batch内只采样ambig样本**

**假设**: 如果batch内只有ambig样本，且都用模板回答"insufficient information"，则必然std=0

**需要检查**:
1. `get_balanced_batch()` 是否正确混合ambig/disambig？
2. BBQAdapter采样比例：80% disambig / 20% ambig是否生效？
3. 单个batch (BATCH_SIZE=2) 中Fairness样本的context_condition分布

**当前配置**:
- `GRPO_BATCH_SIZE = 2` （每步2个样本）
- BBQAdapter: 80% disambig / 20% ambig（全局比例）

**潜在问题**:
- 如果batch=2，fairness只有1个样本（另一个是hallucination）
- 那1个fairness样本如果是ambig → 4个候选用模板 → std=0

**诊断方案**:
```python
# 在训练循环中添加batch composition诊断
if step < 20:
    fairness_samples = [s for s in batch if s.task == "fairness"]
    for i, s in enumerate(fairness_samples):
        ctx_cond = s.meta.get("context_condition", "unknown")
        print(f"  Fairness sample #{i}: context_condition={ctx_cond}")
```

**可能修复**:
1. **增加BATCH_SIZE**: 2 → 4（确保至少2个fairness样本）
2. **强制disambig采样**: 对于训练前期（step<50），强制100% disambig
3. **分离ambig/disambig batches**: 不混合，交替训练

---

### ❓ **原因3: LLM Judge对ambig模板打分过于一致**

**假设**: 对于ambig样本，所有"insufficient information"回答都得到相同的高分（如0.8-0.9）

**理论依据**:
- llm_judge_prompts_v2.py Line 214:
  ```python
  if context_condition == "ambig":
      correct_answer = unknown_option
      task_type = "AMBIGUOUS"
      reasoning_focus = "识别信息不足"
  ```
- LLM Judge可能对"标准答案"都给相同分数

**诊断方案**:
```python
# 在零梯度组诊断中添加LLM Judge原始分数打印
if step < 20 and zero_gradient_group_idx is not None:
    i = zero_gradient_group_idx
    sample = batch[i]
    if sample.task == "fairness" and sample.meta.get("context_condition") == "ambig":
        print(f"\n[LLM Judge诊断] Ambig样本，4个候选的原始评分:")
        for j in range(K):
            idx = i * K + j
            response = all_resps[idx]
            # 重新评估获取详细分数
            result = judge.evaluate(sample, response)
            print(f"  Candidate {j+1}:")
            print(f"    Final score: {result.get('final', 'N/A')}")
            print(f"    Provider: {result.get('provider', 'N/A')}")
            print(f"    Response (前80字符): {response[:80]}")
```

**可能修复**:
1. **修改ambig评分逻辑**: 即使答案正确(unknown)，也根据justification质量产生差异
2. **增加reasoning权重**: 让推理质量成为主要区分点
3. **添加diversity bonus**: 对不同表述方式给予额外分数

---

### ❓ **原因4: Reward Scale精度丢失**

**假设**: `FAIRNESS_REWARD_SCALE = 0.7` 可能导致微小差异被抹平

**场景**:
```python
# 原始LLM Judge分数（假设）
candidate_1: 0.85
candidate_2: 0.86
candidate_3: 0.85
candidate_4: 0.86

# 应用scale=0.7后
candidate_1: 0.595
candidate_2: 0.602
candidate_3: 0.595
candidate_4: 0.602

# 如果后续normalization使用float32精度截断，可能变成
candidate_1-4: 0.60（完全相同）
```

**诊断方案**:
```python
# 在reward scale后立即打印
if step < 20:
    fairness_indices = [i for i, t in enumerate(task_list) if t == "fairness"]
    if fairness_indices:
        f_rewards_before_scale = rewards_before_scale[fairness_indices]  # 需要保存scale前的值
        f_rewards_after_scale = rewards[fairness_indices]
        print(f"[Reward Scale诊断@step{step+1}]")
        print(f"  Before scale (0.7): {f_rewards_before_scale.cpu().numpy()}")
        print(f"  After scale: {f_rewards_after_scale.cpu().numpy()}")
        print(f"  Std before: {f_rewards_before_scale.std():.6f}")
        print(f"  Std after: {f_rewards_after_scale.std():.6f}")
```

**可能修复**:
1. **提高FAIRNESS_REWARD_SCALE**: 0.7 → 1.0（与hallucination平等）
2. **使用float64**: 提高数值精度
3. **移除scale**: 让normalization自动处理平衡

---

### ❓ **原因5: Reward Normalization抹平差异**

**假设**: EMA z-score标准化可能在方差过小时产生数值不稳定

**代码审查** (trainer.py:436-485):
```python
def update_and_normalize(self, rewards, tasks):
    # ...
    batch_var = task_rewards_clean.var().item() if mask.sum() > 1 else 1.0

    # 最小方差0.01
    self.stats[task]["var"] = max(
        self.decay * old_var + (1 - self.decay) * batch_var,
        0.01
    )

    # Z-score
    ema_std = np.sqrt(max(self.stats[task]["var"], 0.01))  # 最小std=0.1
    normalized_task = (task_rewards - ema_mean) / ema_std
```

**潜在问题**:
1. 如果`batch_var ≈ 0`（所有候选奖励相同），EMA仍会保留历史方差
2. 但历史方差可能也很小（如果一直都是模板）
3. 最小方差0.01可能不足以防止数值不稳定

**诊断方案**:
```python
# 在normalization后打印详细统计
if step < 20:
    fairness_indices = [i for i, t in enumerate(task_list) if t == "fairness"]
    if fairness_indices:
        f_rewards_before_norm = rewards_before_norm[fairness_indices]
        f_rewards_after_norm = rewards[fairness_indices]

        print(f"[Reward Normalization诊断@step{step+1}]")
        print(f"  Before norm: mean={f_rewards_before_norm.mean():.4f}, std={f_rewards_before_norm.std():.6f}")
        print(f"  After norm: mean={f_rewards_after_norm.mean():.4f}, std={f_rewards_after_norm.std():.6f}")
        print(f"  EMA stats: mean={reward_normalizer.stats.get('fairness', {}).get('mean', 'N/A'):.4f}, "
              f"std={np.sqrt(reward_normalizer.stats.get('fairness', {}).get('var', 0)):.4f}")
        print(f"  Values before norm: {f_rewards_before_norm.cpu().numpy()}")
        print(f"  Values after norm: {f_rewards_after_norm.cpu().numpy()}")
```

**可能修复**:
1. **提高最小方差**: 0.01 → 0.1（std从0.1→0.316）
2. **禁用normalization**: `REWARD_NORMALIZE = False`（至少在初期）
3. **修改normalization策略**: 使用min-max scaling而非z-score

---

### ❓ **原因6: 4个候选来源验证**

**假设**: 虽然理论上每个样本生成K=4个候选，但可能有bug导致4个候选实际来自不同样本

**需要验证**: `compute_group_advantages`的grouping逻辑

**代码审查** (trainer.py:3713-3756):
```python
def compute_group_advantages(rewards: torch.Tensor, k: int):
    B = Bk // k
    r = rewards.view(B, k)  # [B, K] - 假设每K个连续reward属于同一组
```

**潜在问题**:
- 依赖`rewards`的顺序与`idx_map`的顺序一致
- 如果中间有任何乱序，grouping会错误

**诊断方案**:
```python
# 在零梯度组诊断中验证grouping
if step < 20 and zero_gradient_group_idx is not None:
    i = zero_gradient_group_idx
    print(f"\n[Grouping验证] 组{i}的idx_map:")
    for j in range(K):
        idx = i * K + j
        mapped_sample_idx = idx_map[idx]
        print(f"  Candidate {j+1}: idx_map[{idx}] = {mapped_sample_idx} (should be {i})")
        if mapped_sample_idx != i:
            print(f"  ❌ ERROR: Grouping错误！候选{j+1}实际属于sample {mapped_sample_idx}")
```

**可能修复**:
- 如果发现grouping错误，需要修复`idx_map`构建逻辑（trainer.py:3986-3993）

---

### ❓ **原因7: Advantage计算的数值问题**

**假设**: 即使reward有微小差异，advantage计算中的除法可能引入数值问题

**代码审查** (trainer.py:3743-3751):
```python
if group_std < 0.01:
    group_adv = torch.zeros_like(group_rewards)  # 零梯度
else:
    group_mean = group_rewards.mean()
    group_adv = (group_rewards - group_mean) / group_std.clamp_min(1e-6)
```

**潜在问题**:
- 阈值`0.01`可能太低（对应std=1%）
- 实际中reward差异可能是0.002-0.005，被判定为"零方差"

**诊断方案**:
```python
# 在compute_group_advantages中添加详细日志
if step < 20:
    for i in range(B):
        group_rewards = r[i]
        group_std = group_rewards.std()
        group_mean = group_rewards.mean()

        if group_std < 0.01:
            print(f"[Advantage诊断] 组{i}: std={group_std:.6f} < 0.01，设置adv=0")
            print(f"  Rewards: {group_rewards.cpu().numpy()}")
        elif group_std < 0.05:  # 也打印接近阈值的情况
            print(f"[Advantage诊断] 组{i}: std={group_std:.6f} (接近阈值)")
            print(f"  Rewards: {group_rewards.cpu().numpy()}")
```

**可能修复**:
1. **降低阈值**: 0.01 → 0.001（允许更小的方差）
2. **使用相对阈值**: `std / mean < 0.01`（相对变化）
3. **保留微小梯度**: 即使std<0.01，也用原始reward作为advantage（与之前错误方案不同，这次要正确归一化）

---

## 🎯 综合修复方案

基于以上分析，建议**分阶段修复**：

### Phase 1: 诊断增强（立即实施）

在`grpo_train()`中添加所有7个诊断检查：

1. Batch composition监控
2. LLM Judge原始分数打印
3. Reward Scale前后对比
4. Reward Normalization详细统计
5. Grouping验证
6. Advantage计算详细日志
7. 每步打印fairness样本的context_condition

**目标**: 找出真正的root cause

### Phase 2: 参数调整（基于诊断结果）

| 参数 | 当前值 | 建议值（方案A） | 建议值（方案B） |
|------|--------|----------------|----------------|
| `GRPO_BATCH_SIZE` | 2 | 4 | 6 |
| `FAIRNESS_REWARD_SCALE` | 0.7 | 1.0 | 1.0 |
| `REWARD_NORMALIZE` | True | True | False |
| `最小方差(RewardNormalizer)` | 0.01 | 0.1 | - |
| `零梯度阈值(advantage)` | 0.01 | 0.001 | 0.005 |
| `BBQ disambig比例` | 80% | 90% | 100% (前50步) |

**方案A**: 保守修复（假设normalization有问题）
**方案B**: 激进修复（禁用normalization，依赖raw reward）

### Phase 3: 数据策略调整

1. **前50步**: 仅使用disambig样本（避免ambig模板影响）
2. **50-100步**: 90% disambig / 10% ambig
3. **100+步**: 恢复80/20比例

**实现**:
```python
# 在BBQAdapter.load_samples()中
current_step = global_step  # 需要传入
if current_step < 50:
    target_disambig_ratio = 1.0
    target_ambig_ratio = 0.0
elif current_step < 100:
    target_disambig_ratio = 0.9
    target_ambig_ratio = 0.1
else:
    target_disambig_ratio = 0.8
    target_ambig_ratio = 0.2
```

---

## 📊 预期效果

应用修复后，预期看到：

1. **Batch composition**: 至少50%的batch包含disambig fairness样本
2. **LLM Judge分数**: 即使是ambig样本，4个候选的分数也有差异（如0.75, 0.80, 0.78, 0.82）
3. **Reward std**: Fairness组内std从0.000提升到>0.01
4. **Advantage**: 非零advantage比例从50%提升到>80%
5. **零梯度组**: 从50%降到<30%

---

## 🚀 下一步行动

1. **立即**: 添加所有7个诊断检查
2. **运行1-2步**: 收集详细日志
3. **分析日志**: 确定主要root cause
4. **应用targeted fix**: 只修复真正的问题
5. **验证**: 确认Fairness信号恢复

---

**Created**: 2025-11-17
**Status**: 诊断方案待实施
