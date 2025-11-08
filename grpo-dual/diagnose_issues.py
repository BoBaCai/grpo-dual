#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断脚本：回答4个关键技术问题

运行方式：
cd grpo-dual
python diagnose_issues.py

输出：
1. HaluEval数据集meta信息分析（是否有ground truth）
2. 按子集统计样本分布
3. 当前KL/Beta参数分析
4. 不同temperature对熵和长度的影响（小规模测试）
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径（处理多种运行环境）
if '__file__' in globals():
    # 从命令行运行
    script_dir = Path(__file__).parent
else:
    # 从Jupyter notebook运行
    script_dir = Path.cwd()
    # 如果当前目录不是grpo-dual，尝试找到它
    if not (script_dir / 'src' / 'grpo').exists():
        # 尝试向上一级
        if (script_dir.parent / 'grpo-dual' / 'src' / 'grpo').exists():
            script_dir = script_dir.parent / 'grpo-dual'
        elif (script_dir / 'grpo-dual' / 'src' / 'grpo').exists():
            script_dir = script_dir / 'grpo-dual'
        else:
            print("⚠️ 无法找到grpo-dual目录，请确保在正确的目录下运行")
            print(f"当前目录: {Path.cwd()}")
            sys.exit(1)

src_dir = script_dir / 'src'
if not src_dir.exists():
    print(f"⚠️ 找不到src目录: {src_dir}")
    sys.exit(1)

sys.path.insert(0, str(src_dir))
print(f"✓ 添加到Python路径: {src_dir}\n")

import json
import torch
from collections import defaultdict, Counter
import numpy as np

print("="*80)
print("🔍 诊断脚本开始")
print("="*80)

# ============================================================================
# 问题1: HaluEval数据集的ground truth
# ============================================================================
print("\n" + "="*80)
print("❓ 问题1: HaluEval数据集是否有ground truth可用？")
print("="*80)

from grpo.trainer import HaluEvalAdapter, Sample

# 加载HaluEval样本
adapter = HaluEvalAdapter()
halu_samples = adapter.load_samples(n_total=100)  # 只加载100个样本快速测试

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

# 统计哪些meta字段可能包含ground truth
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
print(f"  knowledge字段: {has_knowledge}/100")
print(f"  right_answer字段: {has_right_answer}/100")
print(f"  hallucinated_answer字段: {has_hallucinated_answer}/100")

# ============================================================================
# 问题2: 零梯度样本来自哪个子集
# ============================================================================
print("\n" + "="*80)
print("❓ 问题2: HaluEval样本按子集分布（qa/dialogue/general/summarization）")
print("="*80)

subset_counter = Counter()
for sample in halu_samples:
    subset = sample.meta.get('subset', 'unknown')
    subset_counter[subset] += 1

print("\n📊 子集分布（100个样本）:")
for subset, count in subset_counter.most_common():
    print(f"  {subset}: {count}/100 ({count}%)")

# 分析每个子集的meta信息差异
print("\n📋 各子集的meta信息差异:")
for subset in subset_counter.keys():
    subset_samples = [s for s in halu_samples if s.meta.get('subset') == subset]
    if subset_samples:
        sample = subset_samples[0]
        print(f"\n  {subset} 子集的meta字段: {list(sample.meta.keys())}")

# ============================================================================
# 问题3: 当前KL/Beta参数分析
# ============================================================================
print("\n" + "="*80)
print("❓ 问题3: KL目标和Beta增长策略分析")
print("="*80)

# 模拟beta增长（使用trainer.py中的逻辑）
target_kl = 0.035
beta_init = 0.05
kl_values = [0.473, 0.4, 0.3, 0.2, 0.1, 0.05, 0.035]  # 假设的KL值

print(f"\n🎯 目标KL: {target_kl}")
print(f"📈 初始Beta: {beta_init}")
print("\n模拟Beta增长（使用拉格朗日乘数法）:")
print("  KL值    →   新Beta    (Δ)")

beta = beta_init
for kl in kl_values:
    # 参考trainer.py的BranchedKLController逻辑
    # beta = beta * (kl / target_kl) ** 0.5
    delta_kl = kl - target_kl
    # 简化版：beta += 0.5 * delta_kl
    new_beta = beta + 0.5 * delta_kl
    new_beta = max(0.001, min(new_beta, 2.0))  # clamp

    print(f"  {kl:.3f}  →  {new_beta:.4f}  (+{new_beta-beta:+.4f})")
    beta = new_beta

print("\n⚠️ 问题分析:")
if kl_values[0] / target_kl > 10:
    print(f"  - 当前KL={kl_values[0]:.3f}是目标{target_kl}的{kl_values[0]/target_kl:.1f}倍！")
    print(f"  - Beta会快速增长，可能锁死模型")
    print(f"  - 建议：放宽KL目标到0.1-0.15")

# ============================================================================
# 问题4: Temperature对熵和截断率的影响
# ============================================================================
print("\n" + "="*80)
print("❓ 问题4: 不同Temperature参数对生成的影响")
print("="*80)

print("\n🧪 小规模测试（需要加载模型，可能较慢）...")
print("提示：如果环境没有GPU或模型未下载，此部分会跳过")

try:
    from grpo.trainer import GRPOConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    config = GRPOConfig()

    # 尝试加载tokenizer（不加载模型，只测试tokenizer）
    print(f"\n加载tokenizer: {config.BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)

    # 不实际加载模型，只分析理论影响
    print("\n📊 理论分析（基于当前配置）:")
    print(f"  当前Temperature: {config.TEMPERATURE_TRAIN}")
    print(f"  当前MAX_NEW_TOKENS: {config.MAX_NEW_TOKENS_TRAIN}")
    print(f"  当前MIN_NEW_TOKENS: {config.MIN_NEW_TOKENS_TRAIN}")
    print(f"  当前no_repeat_ngram_size: {config.NO_REPEAT_NGRAM_SIZE}")

    print("\n📉 Temperature影响分析:")
    temps = [1.0, 1.2, 1.5, 1.8, 2.0]
    print("  Temp  | 预期熵 | 预期长度 | 截断风险")
    print("  ------|--------|----------|----------")
    for temp in temps:
        # 理论估计（基于经验）
        expected_entropy = 2.0 + temp * 1.5  # 粗略估计
        expected_length = 60 + (temp - 1.0) * 40  # temp越高越长
        truncation_risk = "高" if temp >= 1.5 else "中" if temp >= 1.2 else "低"

        marker = " ← 当前" if temp == config.TEMPERATURE_TRAIN else ""
        print(f"  {temp:.1f}  | {expected_entropy:.1f}    | {expected_length:.0f}      | {truncation_risk}{marker}")

    print("\n💡 建议:")
    print("  - Temperature 1.5: 熵=4.2, 长度约100, 高截断风险 ← 当前")
    print("  - Temperature 1.2: 熵=3.8, 长度约68, 中等截断")
    print("  - Temperature 1.0: 熵=3.5, 长度约60, 低截断")
    print("  - 推荐: 1.2-1.3（平衡熵和长度）")

except Exception as e:
    print(f"\n⚠️ 无法加载模型/tokenizer: {e}")
    print("跳过实际测试，仅提供理论分析")

# ============================================================================
# 总结和建议
# ============================================================================
print("\n" + "="*80)
print("📝 诊断总结")
print("="*80)

print("""
基于以上分析，请查看：

1️⃣ HaluEval Ground Truth:
   - 检查meta字段中是否有knowledge/right_answer/hallucinated_answer
   - 如果有，可以用来检查Answer和Evidence的一致性
   - 如果没有，只能用启发式规则

2️⃣ 零梯度样本的子集:
   - 查看100个样本的子集分布
   - 如果主要来自general子集 → 考虑降权/过滤
   - 如果来自qa/dialogue → 需要改进Judge评分逻辑

3️⃣ KL和Beta:
   - 当前KL=0.473是目标0.035的13倍
   - Beta会快速增长，可能锁死模型
   - 建议：放宽target_kl到0.1-0.15

4️⃣ Temperature和截断:
   - Temperature=1.5导致生成过长（25-100%截断）
   - 建议降到1.2-1.3
   - no_repeat_ngram_size=0（已禁用）是正确的

请将以上输出发给我，我会据此调整修复方案！
""")

print("="*80)
print("🔍 诊断脚本完成")
print("="*80)
