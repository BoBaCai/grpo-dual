#!/usr/bin/env python3
"""
深度 Entropy 崩溃诊断脚本

检查 10+ 个可能导致 Entropy 崩溃的根本原因：
1. SFT 过度拟合
2. Base model 本身 Entropy 低
3. Reward 信号退化
4. Advantage 计算 bug
5. LoRA 梯度消失
6. Temperature 配置被覆盖
7. KL penalty 过强
8. Logits 根源问题
9. Repetition penalty 副作用
10. 数据泄露

用法：
  python scripts/deep_entropy_diagnosis.py
"""

import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

print("="*80)
print("🔬 深度 Entropy 崩溃诊断")
print("="*80)

# ============================================================================
# 诊断 1: SFT 过度拟合检查
# ============================================================================
print("\n" + "="*80)
print("📊 诊断 1: SFT 过度拟合检查")
print("="*80)

print("""
检查点:
1. SFT 最终 loss 是否过低（< 0.1）？
2. SFT 训练步数是否过多？
3. 训练数据是否太少导致重复过多？

请查看 SFT 训练日志，回答以下问题：
- SFT 最终 loss: ______
- SFT 训练步数: ______
- 训练样本数: BBQ 1100 + HaluEval 400 = 1500

⚠️ 警告标准:
  - Loss < 0.1: 可能过拟合
  - Loss < 0.05: 严重过拟合（模型"背诵"训练数据）
  - Steps/Samples ratio > 0.2: 过度训练（每个样本平均见 >0.2 次）

💡 如果过拟合：
  - 方案A: 减少 SFT_STEPS (200 → 100)
  - 方案B: 增加数据量 (1500 → 3000+)
  - 方案C: 增加 dropout (0.1 → 0.2)
  - 方案D: 跳过 SFT，直接从 base model 开始 GRPO
""")

# ============================================================================
# 诊断 2: Base Model Entropy 检查
# ============================================================================
print("\n" + "="*80)
print("📊 诊断 2: Base Model (Reference) Entropy 检查")
print("="*80)

print("""
核心假设：如果 base_model 本身生成就极度确定，那么 GRPO 的 KL penalty 会
强制 policy model 靠近 base_model，导致 policy model 也极度确定。

检查方法：用 base_model 单独生成，看 Entropy

请在 Python 中运行以下代码（需要先加载模型）：

```python
# 加载 base model（不要加载 LoRA adapter）
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-3.2-1B-Instruct"  # 替换为你的模型
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model.eval()

# 生成时记录 logits
prompt = "Context: John has 15 years of experience. Question: Who is more experienced? A) John B) Mary C) Unknown"
inputs = tokenizer([prompt], return_tensors="pt")

with torch.no_grad():
    outputs = base_model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.9,
        return_dict_in_generate=True,
        output_scores=True
    )

# 计算每一步的 entropy
import torch.nn.functional as F
entropies = []
for scores in outputs.scores[:10]:  # 前10个token
    probs = F.softmax(scores[0] / 0.9, dim=-1)  # temperature=0.9
    entropy = -(probs * torch.log(probs + 1e-10)).sum()
    entropies.append(entropy.item())
    max_prob = probs.max().item()
    print(f"Token step: entropy={entropy:.3f}, max_prob={max_prob:.4f}")

print(f"\\nBase model average entropy: {np.mean(entropies):.3f}")
```

⚠️ 判断标准:
  - Base model avg entropy < 0.5: 🔴 Base model 本身就有问题！
  - Base model avg entropy 0.5-1.5: 🟡 偏低但可接受
  - Base model avg entropy > 1.5: ✅ Base model 正常

💡 如果 base model entropy 低:
  - 可能需要换一个 base model
  - 或者降低 KL penalty (beta 降低 50%)
  - 或者完全移除 KL penalty（纯 reward 优化）
""")

# ============================================================================
# 诊断 3: Reward 信号退化检查
# ============================================================================
print("\n" + "="*80)
print("📊 诊断 3: Reward 信号退化检查")
print("="*80)

print("""
检查点:
1. Reward 是否总是固定几个值（0.420, 0.700）？
2. Reward std 是否接近 0？
3. Fairness 和 Hallucination 的 reward 是否都一样？

请从训练日志中提取 10-20 步的 Reward 数据，然后运行：

```python
# 从日志中提取的 Reward 数据（示例）
fairness_rewards = [0.420, 0.420, -0.210, 0.420, ...]  # 替换为实际数据
hallucination_rewards = [-0.500, -0.600, -0.400, ...]

import numpy as np
print(f"Fairness Reward:")
print(f"  Mean: {np.mean(fairness_rewards):.3f}")
print(f"  Std: {np.std(fairness_rewards):.3f}")
print(f"  Unique values: {len(set(fairness_rewards))}")

print(f"Hallucination Reward:")
print(f"  Mean: {np.mean(hallucination_rewards):.3f}")
print(f"  Std: {np.std(hallucination_rewards):.3f}")
print(f"  Unique values: {len(set(hallucination_rewards))}")
```

⚠️ 判断标准:
  - Reward std < 0.1: 🔴 严重退化，无区分度
  - Unique values < 3: 🔴 Reward 过于离散，没有连续信号
  - 某个任务的 reward 全一样: 🔴 该任务的 judge 有问题

💡 如果 reward 退化:
  - 检查 judge 评估逻辑（是否总是返回固定值）
  - 检查 reward normalization（是否过度标准化）
  - 尝试移除 reward normalization，用原始 reward
""")

# ============================================================================
# 诊断 4: GRPO Advantage 计算检查
# ============================================================================
print("\n" + "="*80)
print("📊 诊断 4: GRPO Advantage 计算检查")
print("="*80)

print("""
核心问题：如果所有样本的 advantage 都一样，模型无法区分好坏样本。

请在 trainer.py 的 GRPO 训练循环中添加诊断代码：

```python
# 在计算 advantage 之后（trainer.py 约 2900 行附近）
# 找到这段代码：
#   adv = reward - reward.mean()
# 在后面添加：

print(f"\\n[Advantage 诊断 @step{step}]")
print(f"  Reward: mean={reward.mean():.3f}, std={reward.std():.3f}")
print(f"  Advantage: mean={adv.mean():.3f}, std={adv.std():.3f}")
print(f"  Advantage range: [{adv.min():.3f}, {adv.max():.3f}]")
print(f"  Non-zero advantages: {(adv.abs() > 0.01).sum()}/{len(adv)}")
```

⚠️ 判断标准:
  - Advantage std < 0.05: 🔴 所有样本得分几乎一样
  - Advantage std < 0.2: 🟡 区分度较低
  - Non-zero advantages < 50%: 🔴 大部分样本没有梯度信号

💡 如果 advantage 退化:
  - 检查 reward 计算是否正确
  - 检查是否在 advantage 计算前过度标准化
  - 尝试增加 K_ROLLOUTS（4 → 8），提高样本多样性
""")

# ============================================================================
# 诊断 5: LoRA 梯度检查
# ============================================================================
print("\n" + "="*80)
print("📊 诊断 5: LoRA 梯度消失检查")
print("="*80)

print("""
核心问题：如果 LoRA 梯度为 0 或极小，模型根本没在学习。

请在 trainer.py 的优化器步骤后添加：

```python
# 在 optimizer.step() 之后（约 3200 行附近）
if step % 5 == 0:  # 每5步检查一次
    total_norm = 0
    lora_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            if 'lora' in name.lower():
                lora_norm += param_norm ** 2

    total_norm = total_norm ** 0.5
    lora_norm = lora_norm ** 0.5

    print(f"\\n[Gradient 诊断 @step{step}]")
    print(f"  Total grad norm: {total_norm:.6f}")
    print(f"  LoRA grad norm: {lora_norm:.6f}")

    # 检查 LoRA 权重是否在变化
    if hasattr(model, 'base_model'):
        for name, param in model.base_model.named_parameters():
            if 'lora_A' in name and 'q_proj' in name:  # 检查一个代表性的 LoRA 层
                print(f"  Sample LoRA weight mean: {param.data.mean():.6f}")
                print(f"  Sample LoRA weight std: {param.data.std():.6f}")
                break
```

⚠️ 判断标准:
  - Total grad norm < 1e-6: 🔴 梯度消失
  - LoRA grad norm < 1e-7: 🔴 LoRA 没在学习
  - LoRA weight std < 1e-4: 🔴 LoRA 权重几乎不变

💡 如果梯度消失:
  - 检查 gradient_checkpointing 是否导致梯度断裂
  - 检查 loss.backward() 是否正确调用
  - 检查 LoRA 的 scaling factor (lora_alpha / lora_r)
  - 尝试提高学习率（3e-6 → 1e-5）
""")

# ============================================================================
# 诊断 6: Temperature 配置检查
# ============================================================================
print("\n" + "="*80)
print("📊 诊断 6: Temperature 实际生效检查")
print("="*80)

print("""
核心问题：配置了 temperature=0.9，但可能在某个地方被覆盖或没生效。

检查方法：在 generate() 调用时打印实际参数

请在 trainer.py 的 generate_k_rollouts 函数中添加（约 2030 行）：

```python
# 在 model.generate() 调用之前
print(f"\\n[Generate Config Check @step{step}]")
print(f"  temperature: {config.TEMPERATURE_TRAIN}")
print(f"  top_k: {config.TOP_K_TRAIN}")
print(f"  top_p: {config.TOP_P_TRAIN}")
print(f"  do_sample: True")

# 然后检查 generate 的实际调用
out = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    min_new_tokens=config.MIN_NEW_TOKENS_TRAIN,
    do_sample=True,
    temperature=config.TEMPERATURE_TRAIN,  # 确认这里用的是配置值
    ...
)
```

额外检查：在 DebugLogitsProcessor 中验证 temperature 是否生效

```python
# 在 DebugLogitsProcessor.forward() 中（约 1890 行）
# 添加检查
probs_no_temp = F.softmax(scores[0], dim=-1)  # 不应用 temperature
probs_with_temp = F.softmax(scores[0] / self.temperature, dim=-1)

max_no_temp = probs_no_temp.max().item()
max_with_temp = probs_with_temp.max().item()

print(f"  Max prob (no temp): {max_no_temp:.4f}")
print(f"  Max prob (with temp={self.temperature}): {max_with_temp:.4f}")
print(f"  Difference: {max_no_temp - max_with_temp:.4f}")
```

⚠️ 判断标准:
  - max_prob (no temp) ≈ max_prob (with temp): 🔴 Temperature 没生效
  - Difference < 0.05: 🔴 Temperature 作用太弱
  - 日志显示的 temperature 和配置不一致: 🔴 被覆盖

💡 如果 temperature 没生效:
  - 检查是否被 model.generation_config 覆盖
  - 检查是否被 logits_processor 修改
  - 直接在 logits_processor 中手动应用 temperature
""")

# ============================================================================
# 诊断 7: KL Penalty 过强检查
# ============================================================================
print("\n" + "="*80)
print("📊 诊断 7: KL Penalty 过强检查")
print("="*80)

print("""
核心问题：如果 beta 过大，KL term 主导 loss，模型不敢偏离 base model。

从训练日志中提取 KL 数据：

```python
# 从日志提取（示例）
kl_values = [0.02, 0.01, 0.03, ...]  # 实际的 KL 值
beta_values = [0.15, 0.15, 0.17, ...]  # 实际的 beta 值

import numpy as np
print(f"KL Statistics:")
print(f"  Mean: {np.mean(kl_values):.4f}")
print(f"  Std: {np.std(kl_values):.4f}")
print(f"  Range: [{np.min(kl_values):.4f}, {np.max(kl_values):.4f}]")

print(f"\\nBeta Statistics:")
print(f"  Mean: {np.mean(beta_values):.4f}")
print(f"  Range: [{np.min(beta_values):.4f}, {np.max(beta_values):.4f}]")

# 计算 KL penalty 占 loss 的比例
avg_reward = 0.5  # 从日志中获取
avg_kl = np.mean(kl_values)
avg_beta = np.mean(beta_values)

kl_term = avg_beta * avg_kl
reward_term = avg_reward
total = abs(reward_term) + abs(kl_term)

print(f"\\nLoss Composition:")
print(f"  Reward term: {reward_term:.4f} ({abs(reward_term)/total*100:.1f}%)")
print(f"  KL term: {kl_term:.4f} ({abs(kl_term)/total*100:.1f}%)")
```

⚠️ 判断标准:
  - KL mean < 0.01: 🔴 模型被"锁死"，不敢探索
  - KL term 占比 > 70%: 🔴 KL penalty 主导，reward 信号太弱
  - Beta > 0.5: 🟡 可能过强

💡 如果 KL 过强:
  - 降低 beta（0.15 → 0.05）
  - 或完全移除 KL penalty（实验性）
  - 或增大 reward scale（让 reward 信号更强）
""")

# ============================================================================
# 诊断 8: Logits 根源问题检查
# ============================================================================
print("\n" + "="*80)
print("📊 诊断 8: Logits 根源问题（是否来自 base model）")
print("="*80)

print("""
核心问题：如果 base model 输出的 logits 本身就极度尖锐，那么问题在根源。

检查方法：比较 base_model 和 policy_model 的 logits

请在 GRPO 训练循环中添加（约 2800 行）：

```python
# 在计算 log_probs 时，同时记录 base model 的 logits
with torch.no_grad():
    base_outputs = base_model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
    base_logits = base_outputs.logits

policy_outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
policy_logits = policy_outputs.logits

# 对比分析
for i in range(min(2, len(batch_input_ids))):  # 只看前2个样本
    base_scores = base_logits[i, -1, :]  # 最后一个token的logits
    policy_scores = policy_logits[i, -1, :]

    base_top5 = torch.topk(base_scores, 5)
    policy_top5 = torch.topk(policy_scores, 5)

    base_gap = (base_top5.values[0] - base_top5.values[1]).item()
    policy_gap = (policy_top5.values[0] - policy_top5.values[1]).item()

    print(f"\\n[Logits 对比 @sample{i}]")
    print(f"  Base model gap: {base_gap:.3f}")
    print(f"  Policy model gap: {policy_gap:.3f}")
    print(f"  Difference: {policy_gap - base_gap:.3f}")
```

⚠️ 判断标准:
  - Base gap > 7: 🔴 问题来自 base model
  - Policy gap ≈ Base gap: 🔴 Policy model 没学到新东西
  - Policy gap > Base gap: 🔴 训练让问题更严重了

💡 如果是根源问题:
  - 换一个不同的 base model
  - 或在 SFT 阶段就添加 entropy bonus
  - 或在预处理时对 base model 做 temperature scaling
""")

# ============================================================================
# 诊断 9: Repetition Penalty 副作用检查
# ============================================================================
print("\n" + "="*80)
print("📊 诊断 9: Repetition Penalty 副作用检查")
print("="*80)

print("""
核心问题：REP_PENALTY=1.18 可能太强，导致模型不敢用常见词，
只能输出"最安全"的低频词，反而降低多样性。

实验方法：临时禁用 repetition penalty，看 entropy 是否恢复

请在 trainer.py 中临时修改：

```python
# 第 231 行附近
REP_PENALTY_TRAIN = 1.18 → 1.0  # 完全禁用

# 或者在 generate() 调用时
out = model.generate(
    ...
    repetition_penalty=1.0,  # 强制覆盖
    ...
)
```

运行 5-10 步，观察 Entropy 是否有变化。

⚠️ 判断标准:
  - 禁用后 Entropy 上升 > 0.1: 🔴 REP_PENALTY 是罪魁祸首
  - 禁用后无明显变化: ✅ REP_PENALTY 不是主因

💡 如果是 REP_PENALTY 问题:
  - 降低到 1.05（轻微惩罚）
  - 或完全禁用（1.0）
  - 改用 frequency_penalty（更温和）
""")

# ============================================================================
# 诊断 10: 数据泄露检查
# ============================================================================
print("\n" + "="*80)
print("📊 诊断 10: 数据泄露检查（训练/GRPO 是否用同一批数据）")
print("="*80)

print("""
核心问题：如果 GRPO 训练时抽到的样本都是 SFT 见过的，模型可能直接"背诵"。

检查方法：

1. 检查数据采样逻辑（trainer.py 约 2666 行）
   - SFT 和 GRPO 是否从同一个 dataset 采样？
   - 是否有 train/val split？

2. 打印 GRPO 训练时的样本 ID：

```python
# 在 GRPO 训练循环中
for step in range(config.GRPO_STEPS):
    batch = dataset.get_balanced_batch(config.GRPO_BATCH_SIZE)

    # 添加诊断
    sample_ids = [s.id for s in batch]
    print(f"[GRPO Step {step}] Sample IDs: {sample_ids[:5]}...")  # 打印前5个
```

3. 与 SFT 训练时的样本 ID 对比，看是否重复。

⚠️ 判断标准:
  - GRPO 样本 ID 与 SFT 100% 重叠: 🔴 完全泄露
  - 重叠 > 80%: 🟡 严重泄露
  - 重叠 < 20%: ✅ 可接受

💡 如果有数据泄露:
  - 实现 train/val split（8:2）
  - SFT 用 train，GRPO 用 val
  - 或增加数据量，降低重复概率
""")

# ============================================================================
# 最终总结
# ============================================================================
print("\n" + "="*80)
print("💡 诊断流程建议")
print("="*80)

print("""
按优先级依次检查：

1. 🔴 高优先级（最可能）:
   - 诊断 2: Base model entropy（如果 < 0.5，直接换模型）
   - 诊断 3: Reward 退化（如果 std < 0.1，修复 judge）
   - 诊断 4: Advantage 退化（如果 std < 0.2，检查计算逻辑）
   - 诊断 6: Temperature 失效（如果没生效，强制应用）

2. 🟡 中优先级:
   - 诊断 1: SFT 过拟合（如果 loss < 0.1，减少步数）
   - 诊断 5: LoRA 梯度消失（如果 norm < 1e-6，检查梯度流）
   - 诊断 7: KL 过强（如果 KL < 0.01，降低 beta）

3. 🟢 低优先级（可能性较小）:
   - 诊断 8: Logits 根源问题
   - 诊断 9: REP_PENALTY 副作用
   - 诊断 10: 数据泄露

建议：
1. 先运行诊断 2（最快，只需生成一次）
2. 然后运行诊断 3-4（从日志提取数据）
3. 如果还没找到原因，再逐一添加诊断代码到 trainer.py

每完成一个诊断，请把结果发给我，我会帮您分析！
""")

print("\n" + "="*80)
print("✅ 诊断脚本准备完成")
print("="*80)
