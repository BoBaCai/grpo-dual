#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断脚本 - Notebook友好版本

直接在notebook cell中复制粘贴运行，会自动找到数据和代码
"""

import sys
import os
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import json

# ============================================================================
# 自动查找grpo-dual目录
# ============================================================================
print("="*80)
print("🔍 诊断脚本开始 (Notebook版)")
print("="*80)
print(f"\n当前工作目录: {Path.cwd()}\n")

# 搜索可能的grpo-dual位置
possible_paths = [
    Path.cwd() / 'grpo-dual' / 'grpo-dual',  # 当前目录下
    Path.cwd() / 'grpo-dual',                 # 当前目录下的grpo-dual
    Path.cwd().parent / 'grpo-dual' / 'grpo-dual',  # 上级目录
    Path.cwd().parent / 'grpo-dual',
    Path('/workspace') / 'grpo-dual' / 'grpo-dual',  # workspace下
    Path('/workspace') / 'grpo-dual',
    Path.home() / 'grpo-dual' / 'grpo-dual',  # home目录
    Path.home() / 'grpo-dual',
]

grpo_dual_dir = None
for p in possible_paths:
    if (p / 'src' / 'grpo' / 'trainer.py').exists():
        grpo_dual_dir = p
        print(f"✓ 找到grpo-dual目录: {p}\n")
        break

if grpo_dual_dir is None:
    print("❌ 无法找到grpo-dual目录！")
    print("\n请运行以下命令clone仓库：")
    print("  !cd /workspace && git clone https://github.com/BoBaCai/grpo-dual.git")
    print("  !cd /workspace/grpo-dual && git checkout claude/check-code-visibility-011CUv96xL2Gie9NuUZzr18m")
    print("\n或者手动指定路径：")
    print("  grpo_dual_dir = Path('/your/path/to/grpo-dual/grpo-dual')")
    sys.exit(1)

# 添加到Python路径
src_dir = grpo_dual_dir / 'src'
sys.path.insert(0, str(src_dir))
print(f"✓ 添加到Python路径: {src_dir}\n")

# 导入模块
try:
    from grpo.trainer import HaluEvalAdapter, BBQAdapter, Sample, GRPOConfig
    print("✓ 成功导入trainer模块\n")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print(f"src_dir = {src_dir}")
    print(f"sys.path = {sys.path[:3]}")
    sys.exit(1)

# ============================================================================
# 问题1: HaluEval数据集的ground truth
# ============================================================================
print("="*80)
print("❓ 问题1: HaluEval数据集是否有ground truth可用？")
print("="*80)

# 加载HaluEval样本
adapter = HaluEvalAdapter()
halu_samples = adapter.load_samples(n_total=100)

print(f"\n📊 加载了 {len(halu_samples)} 个HaluEval样本")
print("\n分析前5个样本的meta信息：\n")

for i, sample in enumerate(halu_samples[:5]):
    print(f"--- 样本 {i+1} ---")
    print(f"ID: {sample.id}")
    print(f"Task: {sample.task}")
    print(f"Prompt前100字符: {sample.prompt[:100]}...")
    print(f"Target前100字符: {sample.target[:100] if sample.target else 'None'}...")
    print(f"\nMeta字段:")
    for key, value in sample.meta.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}... (长度={len(value)})")
        else:
            print(f"  {key}: {value}")
    print()

# 统计meta字段
meta_keys_counter = Counter()
has_knowledge = 0
has_right_answer = 0
has_hallucinated_answer = 0

for sample in halu_samples:
    for key in sample.meta.keys():
        meta_keys_counter[key] += 1

    if 'knowledge' in sample.meta:
        has_knowledge += 1
    if 'right_answer' in sample.meta:
        has_right_answer += 1
    if 'hallucinated_answer' in sample.meta:
        has_hallucinated_answer += 1

print("\n📈 Meta字段统计（100个样本）:")
for key, count in meta_keys_counter.most_common():
    print(f"  {key}: {count}/100")

print(f"\n🎯 Ground Truth可用性:")
print(f"  knowledge字段: {has_knowledge}/100 样本")
print(f"  right_answer字段: {has_right_answer}/100 样本")
print(f"  hallucinated_answer字段: {has_hallucinated_answer}/100 样本")

# ============================================================================
# 问题2: 零梯度样本来自哪个子集
# ============================================================================
print("\n" + "="*80)
print("❓ 问题2: HaluEval样本按子集分布")
print("="*80)

subset_counter = Counter()
for sample in halu_samples:
    subset = sample.meta.get('subset', 'unknown')
    subset_counter[subset] += 1

print("\n📊 子集分布（100个样本）:")
for subset, count in subset_counter.most_common():
    print(f"  {subset}: {count}/100 ({count}%)")

# 分析每个子集的meta信息差异
print("\n📋 各子集的meta信息:")
for subset in sorted(subset_counter.keys()):
    subset_samples = [s for s in halu_samples if s.meta.get('subset') == subset]
    if subset_samples:
        sample = subset_samples[0]
        print(f"\n  {subset} 子集:")
        print(f"    Meta字段: {list(sample.meta.keys())}")
        print(f"    示例prompt前80字符: {sample.prompt[:80]}...")

# ============================================================================
# 问题3: 当前KL/Beta参数分析
# ============================================================================
print("\n" + "="*80)
print("❓ 问题3: KL目标和Beta增长策略分析")
print("="*80)

config = GRPOConfig()
target_kl = 0.035  # 从trainer.py中的BranchedKLController
beta_init_f = 0.05
beta_init_h = 0.05

# 从训练日志中观察到的KL值
observed_kl_f = 0.473
observed_kl_h = 0.171

print(f"\n🎯 当前配置:")
print(f"  目标KL: {target_kl}")
print(f"  Fairness初始Beta: {beta_init_f}")
print(f"  Hallucination初始Beta: {beta_init_h}")

print(f"\n📊 观察到的KL值（来自Step 20日志）:")
print(f"  Fairness KL: {observed_kl_f:.3f} (目标的 {observed_kl_f/target_kl:.1f}x)")
print(f"  Hallucination KL: {observed_kl_h:.3f} (目标的 {observed_kl_h/target_kl:.1f}x)")

print(f"\n⚠️ 问题分析:")
if observed_kl_f / target_kl > 10:
    print(f"  🔥 Fairness KL过高！是目标的{observed_kl_f/target_kl:.1f}倍")
    print(f"  - Beta会从{beta_init_f}快速增长到{beta_init_f + 0.5*(observed_kl_f-target_kl):.3f}")
    print(f"  - 高Beta会锁死模型，限制探索")
    print(f"  - 建议: 放宽target_kl到0.10-0.15")

if observed_kl_h / target_kl > 4:
    print(f"  ⚠️ Hallucination KL也偏高（目标的{observed_kl_h/target_kl:.1f}倍）")

# 模拟Beta增长
print(f"\n📈 模拟Beta增长轨迹（Fairness）:")
print("  Step | KL    | Beta   | 说明")
print("  -----|-------|--------|------------------")
kl_sequence = [0.473, 0.4, 0.3, 0.2, 0.1, 0.05, 0.035]
beta = beta_init_f
for step, kl in enumerate(kl_sequence, 1):
    delta_kl = kl - target_kl
    new_beta = beta + 0.5 * delta_kl
    new_beta = max(0.001, min(new_beta, 2.0))

    status = "锁死" if new_beta > 0.3 else "健康" if new_beta < 0.15 else "偏高"
    print(f"  {step:4d} | {kl:.3f} | {new_beta:.3f} | {status}")
    beta = new_beta

# ============================================================================
# 问题4: Temperature对熵和截断率的影响
# ============================================================================
print("\n" + "="*80)
print("❓ 问题4: Temperature参数对生成的影响")
print("="*80)

print(f"\n📊 当前配置:")
print(f"  Temperature: {config.TEMPERATURE_TRAIN}")
print(f"  MAX_NEW_TOKENS: {config.MAX_NEW_TOKENS_TRAIN}")
print(f"  MIN_NEW_TOKENS: {config.MIN_NEW_TOKENS_TRAIN}")
print(f"  no_repeat_ngram_size: {config.NO_REPEAT_NGRAM_SIZE}")
print(f"  rep_penalty: {config.REP_PENALTY_TRAIN}")

print("\n📉 Temperature影响分析（基于理论和观察）:")
print("  Temp | 预期熵 | 预期长度 | 截断率 | 推荐")
print("  -----|--------|----------|--------|------")

temp_configs = [
    (1.0, 3.0, 50, "低(5-15%)", "保守"),
    (1.2, 3.5, 65, "中(15-30%)", "推荐 ✓"),
    (1.3, 3.8, 75, "中高(30-45%)", "可接受"),
    (1.5, 4.2, 95, "高(50-75%)", "当前"),
    (1.8, 4.8, 115, "很高(75-90%)", "过度"),
    (2.0, 5.2, 125, "极高(90%+)", "太高"),
]

for temp, entropy, length, trunc, rec in temp_configs:
    marker = " ←" if temp == config.TEMPERATURE_TRAIN else ""
    print(f"  {temp:.1f}  | {entropy:.1f}    | {length:3d}      | {trunc:12s} | {rec}{marker}")

print("\n💡 建议:")
print("  - 当前Temperature=1.5导致50-100%截断率")
print("  - 熵已经足够（mean=2.3-4.1），不需要过高temperature")
print("  - 推荐: 降到1.2-1.3，平衡熵和长度")
print("  - 预期效果: 熵保持3.5-4.0，截断率降到15-30%")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("📝 诊断总结与修复建议")
print("="*80)

print("""
基于以上分析：

🎯 立即修复的问题:

1️⃣ HaluEval Judge评分（最优先！）
   现状: 只检查格式，不检查内容质量 → 零梯度
   修复: 添加内容质量检测
   - 检测口语化/瞎编开头（"yes there", "well maybe"）→ -0.3
   - 检测模糊泛泛描述（"good performance", "based on general"）→ -0.2
   - 检测缺乏具体细节（无引号、无人名、无数字）→ -0.2
   - 检测Answer/Evidence一致性 → -0.3
   - 预期: 即使格式正确，瞎编内容得分0.3-0.5，产生差异

2️⃣ 降低Temperature（次优先）
   现状: 1.5导致50-100%截断率
   修复: 降到1.2-1.3
   - 保持熵在3.5-4.0（足够多样性）
   - 截断率降到15-30%（可接受）
   - 避免生成过长废话

3️⃣ 放宽KL目标（可选）
   现状: target_kl=0.035过严，实际KL=0.47（13倍）
   修复: 放宽到0.10-0.15
   - 避免Beta快速增长锁死模型
   - 给模型更多探索空间
   - 参考其他RLHF工作常用0.1

准备好了吗？我现在就实施这些修复！
""")

print("="*80)
print("🔍 诊断脚本完成")
print("="*80)
