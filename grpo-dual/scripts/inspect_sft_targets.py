#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SFT Target 长度检查脚本

诊断目标：检查 SFT 训练数据的 target 长度是否与 MIN_NEW_TOKENS=5 匹配
核心问题：如果 target 平均长度远大于 5 tokens，会导致：
  1. 模型在 SFT 学习生成更长的内容
  2. GRPO 时模型想生成长内容，但被 MIN_NEW_TOKENS=5 约束
  3. EOS Suppressor 强制禁止过早结束
  4. 模型不知道说什么 → 生成"最确定"的token → Entropy崩溃

用法：
  python scripts/inspect_sft_targets.py
"""

import sys
import os
from pathlib import Path
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 导入配置
from grpo.trainer import config, BBQAdapter, HaluEvalAdapter, Sample

def tokenize_rough(text: str) -> int:
    """
    粗略估算token数量（实际会用真实tokenizer）
    经验公式：英文约4字符/token，中文约1.5字符/token
    这里用保守估算：3.5字符/token
    """
    return len(text) // 4  # 粗略估算

def analyze_targets(samples: List[Sample], name: str):
    """分析target长度分布"""
    print(f"\n{'='*80}")
    print(f"📊 {name} Target 长度分析")
    print(f"{'='*80}")

    if not samples:
        print("❌ 无样本数据")
        return

    # 统计
    char_lengths = []
    token_lengths_rough = []

    for s in samples:
        if s.target:
            char_len = len(s.target)
            token_len = tokenize_rough(s.target)
            char_lengths.append(char_len)
            token_lengths_rough.append(token_len)

    if not char_lengths:
        print("❌ 所有样本的 target 都为空")
        return

    # 统计指标
    char_mean = sum(char_lengths) / len(char_lengths)
    char_min = min(char_lengths)
    char_max = max(char_lengths)
    char_median = sorted(char_lengths)[len(char_lengths)//2]

    token_mean = sum(token_lengths_rough) / len(token_lengths_rough)
    token_min = min(token_lengths_rough)
    token_max = max(token_lengths_rough)
    token_median = sorted(token_lengths_rough)[len(token_lengths_rough)//2]

    print(f"\n样本数: {len(samples)}")
    print(f"\n字符长度统计:")
    print(f"  平均: {char_mean:.1f} 字符")
    print(f"  中位数: {char_median} 字符")
    print(f"  范围: {char_min} - {char_max} 字符")

    print(f"\nToken 长度估算 (粗略):")
    print(f"  平均: {token_mean:.1f} tokens")
    print(f"  中位数: {token_median} tokens")
    print(f"  范围: {token_min} - {token_max} tokens")

    # 🔥 关键诊断
    print(f"\n🔥 关键诊断:")
    print(f"  当前配置: MIN_NEW_TOKENS_TRAIN = {config.MIN_NEW_TOKENS_TRAIN}")

    if token_mean > config.MIN_NEW_TOKENS_TRAIN * 2:
        print(f"  ⚠️ 警告: Target 平均长度 ({token_mean:.1f}) 是 MIN_NEW_TOKENS ({config.MIN_NEW_TOKENS_TRAIN}) 的 {token_mean/config.MIN_NEW_TOKENS_TRAIN:.1f}x")
        print(f"  → SFT 训练模型生成 {token_mean:.1f} tokens，但 GRPO 时只允许最少 {config.MIN_NEW_TOKENS_TRAIN} tokens")
        print(f"  → 这可能导致模型想生成更长内容但被强制截断 → EOS Suppressor 触发 → Entropy 崩溃")
        print(f"  建议: MIN_NEW_TOKENS_TRAIN 应至少设为 {int(token_mean * 0.7)}-{int(token_mean)}")
    elif token_mean > config.MIN_NEW_TOKENS_TRAIN:
        print(f"  ✅ Target 平均长度 ({token_mean:.1f}) 略高于 MIN_NEW_TOKENS ({config.MIN_NEW_TOKENS_TRAIN})")
        print(f"  建议: 考虑提升 MIN_NEW_TOKENS_TRAIN 到 {int(token_mean * 0.8)}-{int(token_mean)} 以更好匹配 SFT 训练")
    else:
        print(f"  ✅ Target 平均长度 ({token_mean:.1f}) 与 MIN_NEW_TOKENS ({config.MIN_NEW_TOKENS_TRAIN}) 基本匹配")

    # 分布统计
    bins = [0, 5, 10, 20, 50, 100, 200, float('inf')]
    bin_labels = ["0-5", "5-10", "10-20", "20-50", "50-100", "100-200", "200+"]
    bin_counts = [0] * len(bin_labels)

    for tl in token_lengths_rough:
        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            if low <= tl < high:
                bin_counts[i] += 1
                break

    print(f"\nToken 长度分布:")
    for label, count in zip(bin_labels, bin_counts):
        pct = count / len(token_lengths_rough) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:>10} tokens: {count:>4} ({pct:>5.1f}%) {bar}")

    # 展示几个样本
    print(f"\n📝 样本展示 (随机抽取3个):")
    sample_indices = random.sample(range(len(samples)), min(3, len(samples)))
    for idx in sample_indices:
        s = samples[idx]
        target_preview = s.target[:100] + "..." if len(s.target) > 100 else s.target
        print(f"\n样本 #{idx} ({s.task}):")
        print(f"  ID: {s.id}")
        print(f"  Target ({len(s.target)} 字符, ~{tokenize_rough(s.target)} tokens):")
        print(f"    {target_preview}")

def main():
    print("="*80)
    print("🔍 SFT Target 长度检查脚本")
    print("="*80)
    print(f"\n当前配置:")
    print(f"  BBQ_DIR: {config.BBQ_DIR}")
    print(f"  HALUEVAL_DIR: {config.HALUEVAL_DIR}")
    print(f"  N_BBQ_TRAIN: {config.N_BBQ_TRAIN}")
    print(f"  N_HALU_TRAIN: {config.N_HALU_TRAIN}")
    print(f"  MIN_NEW_TOKENS_TRAIN: {config.MIN_NEW_TOKENS_TRAIN}")
    print(f"  MAX_NEW_TOKENS_TRAIN: {config.MAX_NEW_TOKENS_TRAIN}")

    # 检查数据目录是否存在
    if not config.BBQ_DIR.exists():
        print(f"\n❌ BBQ 目录不存在: {config.BBQ_DIR}")
        print("   请确保数据目录正确")
        return

    if not config.HALUEVAL_DIR.exists():
        print(f"\n❌ HaluEval 目录不存在: {config.HALUEVAL_DIR}")
        print("   请确保数据目录正确")
        return

    # 加载数据
    print(f"\n{'='*80}")
    print("📦 加载数据...")
    print(f"{'='*80}")

    bbq = BBQAdapter()
    bbq_samples = bbq.load_samples(config.N_BBQ_TRAIN)

    halu = HaluEvalAdapter()
    halu_samples = halu.load_samples(config.N_HALU_TRAIN)

    # 分析
    analyze_targets(bbq_samples, "BBQ (Fairness)")
    analyze_targets(halu_samples, "HaluEval (Hallucination)")

    # 总体分析
    all_samples = bbq_samples + halu_samples
    analyze_targets(all_samples, "总体 (BBQ + HaluEval)")

    # 最终建议
    print(f"\n{'='*80}")
    print("💡 最终建议")
    print(f"{'='*80}")

    all_token_lengths = [tokenize_rough(s.target) for s in all_samples if s.target]
    if all_token_lengths:
        avg_len = sum(all_token_lengths) / len(all_token_lengths)

        print(f"\n当前配置:")
        print(f"  MIN_NEW_TOKENS_TRAIN = {config.MIN_NEW_TOKENS_TRAIN}")
        print(f"  SFT Target 平均长度 ≈ {avg_len:.1f} tokens")

        if avg_len > config.MIN_NEW_TOKENS_TRAIN * 2:
            recommended_min = int(avg_len * 0.7)
            print(f"\n🔴 严重不匹配！建议修改:")
            print(f"  MIN_NEW_TOKENS_TRAIN = 5 → {recommended_min}")
            print(f"\n原因:")
            print(f"  - SFT 训练模型生成 {avg_len:.1f} tokens 的内容")
            print(f"  - GRPO 时 MIN_NEW_TOKENS=5 过短，导致模型想生成长内容但被限制")
            print(f"  - EOS Suppressor 强制禁止前5个token的EOS → 模型被迫续写")
            print(f"  - 模型不知道说什么 → 生成最确定的token → Entropy 崩溃到 0.005")
            print(f"\n修复步骤:")
            print(f"  1. 修改 trainer.py 第 {214} 行左右:")
            print(f"     MIN_NEW_TOKENS_TRAIN = 5 → {recommended_min}")
            print(f"  2. 重新开始 GRPO 训练（SFT 不需要重训）")
        elif avg_len > config.MIN_NEW_TOKENS_TRAIN * 1.5:
            recommended_min = int(avg_len * 0.8)
            print(f"\n🟡 中等不匹配，建议优化:")
            print(f"  MIN_NEW_TOKENS_TRAIN = 5 → {recommended_min}")
        else:
            print(f"\n✅ Target 长度与 MIN_NEW_TOKENS 基本匹配")
            print(f"   Entropy 崩溃可能由其他原因引起")

if __name__ == "__main__":
    main()
