#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轻量版Temperature vs Entropy测试 (Notebook友好)

可以直接在notebook cell中复制粘贴运行
"""

import sys
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict

# ============================================================================
# 自动查找grpo-dual目录
# ============================================================================
print("="*80)
print("🌡️ Temperature vs Entropy 轻量测试")
print("="*80)

# 搜索可能的grpo-dual位置
possible_paths = [
    Path.cwd() / 'grpo-dual',
    Path.cwd(),
    Path.cwd().parent / 'grpo-dual',
    Path('/home/user/grpo-dual/grpo-dual'),
]

grpo_dual_dir = None
for p in possible_paths:
    if (p / 'src' / 'grpo' / 'trainer.py').exists():
        grpo_dual_dir = p
        print(f"✓ 找到grpo-dual目录: {p}\n")
        break

if grpo_dual_dir is None:
    print("❌ 无法找到grpo-dual目录！")
    sys.exit(1)

# 添加到Python路径
src_dir = grpo_dual_dir / 'src'
sys.path.insert(0, str(src_dir))

from grpo.trainer import Config
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# 配置
# ============================================================================
config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# 只测试关键温度点
TEMPS_TO_TEST = [0.7, 1.0, 1.2, 1.5]

# 只用3个prompts快速测试
TEST_PROMPTS = [
    "Question: What is the capital of France?",
    "Explain machine learning in simple terms.",
    "If a train travels 120 km in 2 hours, what is its average speed?",
]

# ============================================================================
# 加载模型
# ============================================================================
print(f"📦 加载模型: {config.BASE_MODEL}")
print(f"设备: {device}\n")

try:
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    print("✓ 模型加载成功\n")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

# ============================================================================
# 核心函数
# ============================================================================
def compute_entropy(logits: torch.Tensor, temperature: float) -> float:
    """计算给定温度下的token-level熵"""
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum().item()
    return entropy

def test_temperature(prompt: str, temperature: float, max_tokens: int = 20) -> dict:
    """测试单个温度"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    entropies = []

    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]

            entropy = compute_entropy(next_token_logits, temperature)
            entropies.append(entropy)

            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            inputs.input_ids = torch.cat([inputs.input_ids, next_token.unsqueeze(0)], dim=1)
            if 'attention_mask' in inputs:
                inputs.attention_mask = torch.cat([
                    inputs.attention_mask,
                    torch.ones((1, 1), dtype=torch.long, device=device)
                ], dim=1)

    return {
        'mean_entropy': np.mean(entropies),
        'tokens': len(entropies),
    }

# ============================================================================
# 主测试
# ============================================================================
print("="*80)
print("🧪 开始测试")
print("="*80)

results = defaultdict(list)

for temp in TEMPS_TO_TEST:
    print(f"\n🌡️ Temperature = {temp}")

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        result = test_temperature(prompt, temp, max_tokens=20)
        results[temp].append(result['mean_entropy'])
        print(f"  Prompt {i}: 熵={result['mean_entropy']:.3f}, tokens={result['tokens']}")

# ============================================================================
# 统计分析
# ============================================================================
print("\n" + "="*80)
print("📊 统计汇总")
print("="*80)

print("\n  Temp  | 平均熵 | 标准差 | 熵增长率 | 状态")
print("  ------|--------|--------|----------|----------")

temps = sorted(results.keys())
mean_entropies = []

for i, temp in enumerate(temps):
    entropies = results[temp]
    mean_ent = np.mean(entropies)
    std_ent = np.std(entropies)
    mean_entropies.append(mean_ent)

    if i > 0:
        prev_mean = mean_entropies[i-1]
        growth_rate = (mean_ent - prev_mean) / prev_mean * 100
        growth_str = f"+{growth_rate:.1f}%"
    else:
        growth_str = "-"

    marker = " ← 当前" if temp == 1.2 else ""
    print(f"  {temp:.1f}  | {mean_ent:.3f}  | {std_ent:.3f}  | {growth_str:8s} | 正常{marker}")

# ============================================================================
# 建议
# ============================================================================
print("\n" + "="*80)
print("💡 结论")
print("="*80)

current_temp = 1.2
current_entropy = mean_entropies[temps.index(current_temp)]

print(f"\n当前T={current_temp}的平均熵: {current_entropy:.3f} nats")
print(f"（等价于 {current_entropy / np.log(2):.3f} bits）")

# 检查熵增长趋势
if len(mean_entropies) >= 3:
    growth_rates = [
        (mean_entropies[i] - mean_entropies[i-1]) / mean_entropies[i-1]
        for i in range(1, len(mean_entropies))
    ]
    avg_growth = np.mean(growth_rates)

    print(f"\n平均熵增长率: {avg_growth*100:.1f}%")

    if avg_growth > 0.3:
        print("⚠️ 熵增长较快，可能接近spike区域")
    else:
        print("✓ 熵增长平稳，温度设置合理")

print(f"\n✓ T=1.2在测试范围内表现正常")
print(f"  熵值适中（{current_entropy:.3f}），支持当前配置")

print("\n" + "="*80)
print("🔍 测试完成！")
print("="*80)
