# Temperature Scheduler 集成指南

## 📋 概述

基于专家建议（参考 DeepSeek-R1, EDT, DAPO），实现了一个**三阶段温度调度器**，配合轻量自适应规则。

### 核心特性

1. **Stage-wise 降温**：高探索（T=1.0-1.2） → 收敛（T=0.7-0.9） → 部署对齐（T=0.6-0.8）
2. **Per-task 差异化**：BBQ/Fairness 略高，HaluEval 略低
3. **轻量自适应**：基于熵和截断率动态微调（步长 ±0.05）
4. **配套调度**：KL 系数、max_new_tokens、截断惩罚

---

## 🚀 快速开始

### Step 1: 导入调度器

在 `trainer.py` 顶部添加：

```python
from temperature_scheduler import TemperatureScheduler, TemperatureConfig
```

### Step 2: 初始化调度器

在 `grpo_train` 函数开始处（Line ~2681）：

```python
def grpo_train(...):
    # ... 现有代码 ...

    # ========== 新增：初始化温度调度器 ==========
    temp_scheduler = TemperatureScheduler(
        total_steps=config.GRPO_STEPS,
        config=TemperatureConfig(
            # 可以使用默认值，或自定义
            T_min=0.6,
            T_max=1.3,
            fairness_T_init=1.10,      # BBQ 初始温度略高
            hallucination_T_init=0.95  # HaluEval 初始温度中等
        )
    )
    print(f"✅ Temperature Scheduler initialized for {config.GRPO_STEPS} steps")
    # ==========================================
```

### Step 3: 在训练循环中使用

#### 3.1 获取当前步的温度

在每个训练步开始时（Line ~2935，`for step in range(start_step, config.GRPO_STEPS):`）：

```python
for step in range(start_step, config.GRPO_STEPS):
    gc.collect()
    torch.cuda.empty_cache()

    # ========== 新增：获取当前步的温度 ==========
    # 方案A：不提供指标，使用纯 stage-wise schedule
    temps = temp_scheduler.get_temperature(step=step)
    T_fairness = temps['fairness']
    T_hallucination = temps['hallucination']
    current_stage = temps['stage']

    # 方案B（推荐）：提供上一步的指标，启用自适应
    # 需要在下面 3.2 中收集指标
    # ==========================================

    print(f"\n{'='*80}")
    print(f"🔥 Step {step+1}/{config.GRPO_STEPS} (Stage {current_stage}) - "
          f"T_fair={T_fairness:.3f}, T_halu={T_hallucination:.3f}")
    print('='*80)
```

#### 3.2 收集指标并启用自适应（推荐）

在每个步骤结束时，收集熵和截断率指标，用于下一步的自适应：

```python
# 在训练循环末尾（计算完指标后，Line ~3150 附近）

# ========== 新增：收集指标用于温度自适应 ==========
# 提取 Fairness 和 Hallucination 的熵和截断率
fairness_mask = np.array([s.task == "fairness" for s in batch_samples])
halu_mask = ~fairness_mask

# 计算平均熵（如果有记录）
fairness_entropy = policy_entropy[fairness_mask].mean() if fairness_mask.any() else None
halu_entropy = policy_entropy[halu_mask].mean() if halu_mask.any() else None

# 计算截断率
fairness_trunc_rate = truncation_frac_f  # 已有变量
halu_trunc_rate = truncation_frac_h      # 已有变量

# 在下一步开始时使用这些指标
# （可以存储到全局变量或缓冲区）
if step < config.GRPO_STEPS - 1:  # 还有下一步
    temps_next = temp_scheduler.get_temperature(
        step=step + 1,
        fairness_entropy=fairness_entropy,
        fairness_trunc_rate=fairness_trunc_rate,
        hallucination_entropy=halu_entropy,
        hallucination_trunc_rate=halu_trunc_rate
    )
# ================================================
```

#### 3.3 使用 per-task 温度生成候选

修改 `generate_candidates_batch` 调用（Line ~2945 附近）：

```python
# 旧代码（统一温度）
# grouped_texts, grouped_lengths, unique_prompt_lens, grouped_truncated, formatted_prompts = \
#     generate_candidates_batch(model, tokenizer, device, prompts, k=config.K_ROLLOUTS, step=step)

# ========== 新增：分任务生成，使用不同温度 ==========
# 按任务分组
fairness_samples = [s for s in batch_samples if s.task == "fairness"]
halu_samples = [s for s in batch_samples if s.task == "hallucination"]

fairness_prompts = [s.prompt for s in fairness_samples]
halu_prompts = [s.prompt for s in halu_samples]

# 分别生成（传入不同温度）
fairness_results = generate_candidates_batch(
    model, tokenizer, device,
    fairness_prompts,
    k=config.K_ROLLOUTS,
    temperature=T_fairness,  # 使用 Fairness 温度
    step=step
)

halu_results = generate_candidates_batch(
    model, tokenizer, device,
    halu_prompts,
    k=config.K_ROLLOUTS,
    temperature=T_hallucination,  # 使用 Hallucination 温度
    step=step
)

# 合并结果（按原顺序）
# ... (需要写一个合并逻辑)
# ================================================
```

**注意**：这需要修改 `generate_candidates_batch` 函数签名，添加 `temperature` 参数。

#### 3.4 修改 `generate_candidates_batch` 支持自定义温度

在 `generate_candidates_batch` 函数（Line ~2524）添加参数：

```python
def generate_candidates_batch(
    model, tokenizer, device,
    prompts: List[str],
    k: int,
    max_new_tokens: int = None,
    step: int = None,
    temperature: float = None  # ========== 新增 ==========
) -> Tuple[...]:
    """..."""

    if max_new_tokens is None:
        max_new_tokens = config.MAX_NEW_TOKENS_TRAIN

    if temperature is None:
        temperature = config.TEMPERATURE_TRAIN  # 使用默认值

    # ... 现有代码 ...

    # 在 model.generate 调用时使用传入的 temperature
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=config.MIN_NEW_TOKENS_TRAIN,
        temperature=temperature,  # ========== 使用传入的温度 ==========
        top_k=config.TOP_K_TRAIN,
        top_p=config.TOP_P_TRAIN,
        # ...
    )
```

### Step 4: 配套功能（可选但推荐）

#### 4.1 动态调整 KL 系数

```python
# 在训练循环中
current_kl_coef = temp_scheduler.get_kl_coefficient(step)

# 在计算 loss 时使用
loss = ppo_loss + current_kl_coef * kl_penalty + ...
```

#### 4.2 动态调整 max_new_tokens

```python
# 获取当前步的 max_new_tokens
max_tokens = temp_scheduler.get_max_new_tokens(step)

# 传给 generate_candidates_batch
generate_candidates_batch(..., max_new_tokens=max_tokens)
```

#### 4.3 截断惩罚

```python
# 对被截断的样本降低 reward
trunc_penalty_coef = temp_scheduler.get_truncation_penalty(step)

for i, is_truncated in enumerate(all_truncated):
    if is_truncated:
        rewards[i] *= trunc_penalty_coef  # 乘以惩罚系数（0.3-0.7）
```

#### 4.4 长度正则化

```python
# 对过长的生成添加负奖励
lambda_len = temp_scheduler.get_length_penalty_lambda(step)
L_target = 128

for i, length in enumerate(all_lengths):
    if length > L_target:
        len_penalty = -lambda_len * max(0, (length - L_target) / L_target)
        rewards[i] += len_penalty
```

### Step 5: 保存和可视化

在训练结束时：

```python
# 保存温度历史
temp_scheduler.save_history(f"{run_dir}/temperature_history.csv")

# 绘制温度曲线
temp_scheduler.plot_history(f"{run_dir}/temperature_history.png")
```

---

## 📊 预期效果

### Stage 1 (0-150 步，30%)

**目标**：高探索，暴露问题

| 指标 | Fairness | Hallucination |
|------|----------|---------------|
| Temperature | 1.10 (范围 1.0-1.25) | 0.95 (范围 0.8-1.1) |
| KL coef | 0.003 | 0.003 |
| Max tokens | 256 | 256 |
| Trunc threshold | 40% | 40% |
| Adapt mode | truncation_only | truncation_only |

**期望**：
- 熵上升到 2.0-4.0
- 零梯度组 <40%
- 生成多样性提升（不再全是模板）

### Stage 2 (150-400 步，50%)

**目标**：收敛，主力对齐

| 指标 | Fairness | Hallucination |
|------|----------|---------------|
| Temperature | 1.05→0.90（线性） | 0.90→0.80（线性） |
| KL coef | 0.003→0.01 | 0.003→0.01 |
| Max tokens | 256→192 | 256→192 |
| Trunc threshold | 15% | 15% |
| Adapt mode | both | both |

**期望**：
- 截断率降到 10-15%
- 熵稳定在 3.0-4.0
- Reward 持续上升

### Stage 3 (400-500 步，20%)

**目标**：精修，接近部署

| 指标 | Fairness | Hallucination |
|------|----------|---------------|
| Temperature | 0.80（范围 0.75-0.9） | 0.75（范围 0.7-0.8） |
| KL coef | 0.01→0.02 | 0.01→0.02 |
| Max tokens | 192 | 192 |
| Trunc threshold | 10% | 10% |
| Adapt mode | truncation_only | truncation_only |

**期望**：
- 截断率 <10%
- 策略稳定，KL 不飙升
- Fairness 和 Hallucination 指标接近目标

---

## 🔧 调试和监控

### 关键打印信息

调度器会在每 5 个窗口（默认 250 步）打印：

```
🌡️ [Step 250] Temperature Update (Stage 2):
  Fairness:      T=0.950 | Entropy=3.45 | Trunc=12.3% | Reason: stable
  Hallucination: T=0.850 | Trunc=8.7% | Reason: entropy_low(2.85<3.0)
```

### 关键曲线

训练后查看 `temperature_history.png`：

1. **Temperature vs Step**：是否平滑降温？
2. **Entropy vs Step**：是否在 3-4 区间稳定？
3. **Truncation vs Step**：是否逐步下降？
4. **T vs Entropy 散点图**：自适应是否生效？

### 异常情况处理

| 症状 | 可能原因 | 调整方案 |
|------|----------|----------|
| 熵持续 <2.0 | 温度过低或 KL 过严 | 提高 `T_init`，降低 `kl_coef` |
| 截断率 >50% | max_tokens 过小 | 增大 Stage 1-2 的 max_tokens 到 256-384 |
| 零梯度组 >60% | 候选仍高度相同 | 检查串行生成是否正确实施 |
| Reward 崩溃 | 探索过头 | 降低 `T_max`，提前进入 Stage 2 |

---

## 📝 与现有 HANDOFF.md 的关系

### 替代的部分

1. **手动温度调整**（1.0→1.3→1.15→1.0）
   - 替代为：Stage-wise schedule + 自适应

2. **固定 KL 系数**（β=0.05）
   - 替代为：Stage-wise KL schedule (0.003→0.02)

3. **固定 max_new_tokens**（128）
   - 替代为：动态调整 (256→192)

### 保留的部分

1. ✅ MIN_NEW_TOKENS = 5
2. ✅ 串行生成（`generate_candidates_batch`）
3. ✅ 细粒度 Reasoning Quality 评分
4. ✅ Evasive Phrases 检测（27 个变体）
5. ✅ Advantage 计算修复（检测 std<0.01）

### 新增的部分

1. ✅ Per-task 温度差异化
2. ✅ 熵和截断率驱动的自适应
3. ✅ 截断惩罚机制
4. ✅ 长度正则化
5. ✅ 温度历史可视化

---

## 🎯 实施优先级

### Phase 1（立即可做）：最小可行集成

只需修改 3 处：

1. 初始化调度器
2. 在训练循环中获取温度
3. 传给 `generate_candidates_batch`

**预期效果**：
- 自动 stage-wise 降温
- 减少手动调参

### Phase 2（验证后）：启用自适应

需要修改 2 处：

1. 收集熵和截断率指标
2. 传给 `get_temperature`

**预期效果**：
- 温度根据实际指标微调
- 更稳定的训练曲线

### Phase 3（优化）：完整集成

添加配套功能：

1. 动态 KL 系数
2. 动态 max_new_tokens
3. 截断惩罚
4. 长度正则

**预期效果**：
- 截断率降到 <10%
- 零梯度组降到 <30%
- 整体训练更稳定

---

## 💡 常见问题

### Q1: 是否需要同时修改 trainer.py 的 config？

**A**: 建议保留 `config.TEMPERATURE_TRAIN` 作为 fallback，但优先使用调度器返回的值。

### Q2: Per-task 温度会不会增加复杂度？

**A**: 会增加一点，但收益明显：
- BBQ 需要高温暴露偏见
- HaluEval 需要中低温保证准确性
- 两者混在一起用统一温度是次优的

### Q3: 如果我只想用 Stage-wise，不要自适应？

**A**: 完全可以！只需在 `get_temperature` 时不传指标：

```python
temps = temp_scheduler.get_temperature(step=step)
# 不传 fairness_entropy 等参数
```

或者设置 `adapt_mode="none"`。

### Q4: 如何调整 Stage 划分比例？

**A**: 修改 `TemperatureConfig`：

```python
config = TemperatureConfig(
    stage1_end=0.25,  # 25% 探索
    stage2_end=0.85,  # 25-85% 收敛
    # 85-100% 精修
)
```

### Q5: DeepSeek-R1 用的是 K=16，我们 K=4 够吗？

**A**: K=4 对于你们的任务（BBQ+HaluEval）是合理的：
- BBQ 是选择题，候选空间有限
- HaluEval 有 ground truth，不需要极多样本

如果发现零梯度组仍 >50%，可以考虑增大到 K=8。

---

## 📚 参考文献

1. **DeepSeek-R1** ([Nature 2025](https://www.nature.com/articles/s41586-025-09422-z))
   - Stage 1: T=1.0, K=16, KL=0.001
   - Stage 2: T=0.7, 减少混语和不连贯

2. **EDT: Entropy-based Dynamic Temperature** ([arXiv 2024](https://arxiv.org/abs/2403.14541))
   - 熵驱动动态温度采样
   - 公式：T_new = T_base * exp(η * (H - H_0))

3. **DAPO: Open-Source LLM RL** ([arXiv 2025](https://arxiv.org/pdf/2503.14476))
   - 多目标 RL 长度控制
   - 截断惩罚和长度正则

4. **HaluEval** ([arXiv 2023](https://arxiv.org/abs/2305.11747))
   - 幻觉评估数据集设计

---

## ✅ 集成检查清单

- [ ] 已导入 `TemperatureScheduler`
- [ ] 已在 `grpo_train` 初始化调度器
- [ ] 已在训练循环中获取温度
- [ ] 已修改 `generate_candidates_batch` 支持 `temperature` 参数
- [ ] （可选）已启用熵和截断率自适应
- [ ] （可选）已启用动态 KL 系数
- [ ] （可选）已启用动态 max_new_tokens
- [ ] （可选）已启用截断惩罚
- [ ] （可选）已启用长度正则
- [ ] 已在训练结束保存温度历史
- [ ] 已绘制温度曲线并验证

---

**祝训练顺利！🚀**
