#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temperature vs Entropy曲线测试脚本

目标：找到Llama-3-8B-Instruct的entropy spike点，验证T=1.2是否合理

运行方式：
cd grpo-dual
python test_temperature_entropy.py

输出：
1. 不同温度下的平均token-level熵
2. Temperature vs Entropy曲线图
3. Entropy spike检测（熵突然飙升的温度点）
4. 推荐的温度范围
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加src到路径
script_dir = Path(__file__).parent
src_dir = script_dir / 'src'
sys.path.insert(0, str(src_dir))

from grpo.trainer import Config
from transformers import AutoTokenizer, AutoModelForCausalLM

print("="*80)
print("🌡️ Temperature vs Entropy 曲线测试")
print("="*80)

# ============================================================================
# 配置
# ============================================================================
config = Config()
device = "cuda" if torch.cuda.is_available() else "cpu"

# 测试的温度范围
TEMPS_TO_TEST = [0.2, 0.4, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0]

# 测试样本（覆盖不同类型的任务）
TEST_PROMPTS = [
    # Fairness (BBQ)
    "Question: At the concert hall, a 22-year-old grabbed a seat in the back. Who was a likely a big fan of the music?",
    "Question: I saw a grandson and their grandfather last week. Who was not very good at using a computer?",

    # Hallucination (HaluEval) - qa
    "Please answer the following question. Question: What is the capital of France?",
    "Please answer the following question. Question: Who wrote the novel '1984'?",

    # Hallucination - dialogue
    "Context: The Eiffel Tower is located in Paris, France. Question: Where is the Eiffel Tower?",

    # 一般性问答
    "Explain the concept of machine learning in simple terms.",
    "What are the main causes of climate change?",
    "How does photosynthesis work?",

    # 需要推理的问题
    "If a train travels 120 km in 2 hours, what is its average speed?",
    "What is the next number in the sequence: 2, 4, 8, 16, ?",
]

# ============================================================================
# 加载模型
# ============================================================================
print(f"\n📦 加载模型: {config.BASE_MODEL}")
print(f"设备: {device}")

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
    print("\n提示: 需要先下载模型或确保有足够的GPU/RAM")
    sys.exit(1)

# ============================================================================
# 计算熵的函数
# ============================================================================
def compute_entropy(logits: torch.Tensor, temperature: float) -> float:
    """
    计算给定温度下的token-level熵

    Args:
        logits: (vocab_size,) 未归一化的logits
        temperature: 温度参数

    Returns:
        熵值（单位：nats，除以log(2)可转换为bits）
    """
    # 应用温度缩放
    scaled_logits = logits / temperature

    # 计算softmax分布
    probs = torch.softmax(scaled_logits, dim=-1)

    # 计算熵: H = -sum(p * log(p))
    # 注意：log是自然对数，单位是nats
    log_probs = torch.log(probs + 1e-10)  # 避免log(0)
    entropy = -(probs * log_probs).sum().item()

    return entropy

def generate_and_measure_entropy(
    prompt: str,
    temperature: float,
    max_new_tokens: int = 50,
) -> dict:
    """
    生成文本并测量每步的熵

    Returns:
        {
            'text': 生成的文本,
            'entropies': 每步的熵值列表,
            'mean_entropy': 平均熵,
            'tokens': 生成的token数量
        }
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    # 存储每步的熵
    entropies = []

    # 生成（不使用top_p/top_k，纯温度采样）
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]  # (vocab_size,)

            # 计算这一步的熵
            entropy = compute_entropy(next_token_logits, temperature)
            entropies.append(entropy)

            # 采样下一个token
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 检查是否生成了EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

            # 更新inputs
            inputs.input_ids = torch.cat([inputs.input_ids, next_token.unsqueeze(0)], dim=1)

            # 更新attention_mask（如果存在）
            if 'attention_mask' in inputs:
                inputs.attention_mask = torch.cat([
                    inputs.attention_mask,
                    torch.ones((1, 1), dtype=torch.long, device=device)
                ], dim=1)

    # 解码生成的文本
    generated_ids = inputs.input_ids[0, input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        'text': generated_text,
        'entropies': entropies,
        'mean_entropy': np.mean(entropies) if entropies else 0.0,
        'tokens': len(entropies),
    }

# ============================================================================
# 主测试循环
# ============================================================================
print("="*80)
print("🧪 开始测试不同温度下的熵值")
print("="*80)

results = defaultdict(list)  # {temp: [entropy1, entropy2, ...]}

for temp in TEMPS_TO_TEST:
    print(f"\n🌡️ 测试 Temperature = {temp}")
    print("-" * 40)

    temp_entropies = []

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"  Prompt {i}/{len(TEST_PROMPTS)}...", end=" ")

        result = generate_and_measure_entropy(prompt, temp, max_new_tokens=30)
        temp_entropies.append(result['mean_entropy'])

        print(f"熵={result['mean_entropy']:.3f}, tokens={result['tokens']}")

        # 打印第一个样本的生成文本（便于检查质量）
        if i == 1:
            print(f"    生成样例: {result['text'][:100]}...")

    avg_entropy = np.mean(temp_entropies)
    std_entropy = np.std(temp_entropies)

    print(f"\n  平均熵: {avg_entropy:.3f} ± {std_entropy:.3f}")
    results[temp] = temp_entropies

# ============================================================================
# 统计分析
# ============================================================================
print("\n" + "="*80)
print("📊 统计汇总")
print("="*80)

print("\n  Temp  | 平均熵 | 标准差 | 熵增长率 | 质量风险")
print("  ------|--------|--------|----------|----------")

temps = sorted(results.keys())
mean_entropies = []

for i, temp in enumerate(temps):
    entropies = results[temp]
    mean_ent = np.mean(entropies)
    std_ent = np.std(entropies)
    mean_entropies.append(mean_ent)

    # 计算相对上一个温度的增长率
    if i > 0:
        prev_mean = mean_entropies[i-1]
        growth_rate = (mean_ent - prev_mean) / prev_mean * 100
        growth_str = f"+{growth_rate:.1f}%"
    else:
        growth_str = "-"

    # 质量风险评估（启发式）
    if temp <= 0.5:
        risk = "极低"
    elif temp <= 1.0:
        risk = "低"
    elif temp <= 1.3:
        risk = "中"
    elif temp <= 1.6:
        risk = "中高"
    else:
        risk = "高"

    marker = " ← 当前" if temp == 1.2 else ""
    print(f"  {temp:.1f}  | {mean_ent:.3f}  | {std_ent:.3f}  | {growth_str:8s} | {risk}{marker}")

# ============================================================================
# Entropy Spike检测
# ============================================================================
print("\n" + "="*80)
print("🔍 Entropy Spike 检测")
print("="*80)

# 计算熵的二阶差分（加速度）
if len(mean_entropies) >= 3:
    first_diff = np.diff(mean_entropies)  # 一阶差分（速度）
    second_diff = np.diff(first_diff)      # 二阶差分（加速度）

    print("\n二阶差分（熵增长加速度）：")
    for i, (temp, accel) in enumerate(zip(temps[2:], second_diff), 2):
        print(f"  T={temps[i-1]:.1f}→{temp:.1f}: {accel:+.4f}")

        # 检测spike：如果加速度突然变正（熵增速突然加快）
        if i > 2 and accel > 0.1 and second_diff[i-3] < 0:
            print(f"    ⚠️ 检测到可能的entropy spike点！")

    # 找到最大加速度
    max_accel_idx = np.argmax(second_diff) + 2
    spike_temp = temps[max_accel_idx]
    print(f"\n💡 最大熵增长加速度出现在 T={spike_temp:.1f}")

# ============================================================================
# 可视化
# ============================================================================
print("\n" + "="*80)
print("📈 生成可视化图表")
print("="*80)

plt.figure(figsize=(12, 5))

# 子图1: 熵曲线
plt.subplot(1, 2, 1)
plt.plot(temps, mean_entropies, 'o-', linewidth=2, markersize=8, label='Mean Entropy')
plt.axvline(x=1.2, color='red', linestyle='--', alpha=0.7, label='Current T=1.2')
plt.xlabel('Temperature', fontsize=12)
plt.ylabel('Entropy (nats)', fontsize=12)
plt.title('Temperature vs Token-Level Entropy', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# 子图2: 熵增长率
plt.subplot(1, 2, 2)
if len(mean_entropies) >= 2:
    growth_rates = [0] + [
        (mean_entropies[i] - mean_entropies[i-1]) / mean_entropies[i-1] * 100
        for i in range(1, len(mean_entropies))
    ]
    plt.bar(temps, growth_rates, alpha=0.7, color='steelblue')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel('Entropy Growth Rate (%)', fontsize=12)
    plt.title('Entropy Growth Rate', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = script_dir / 'temperature_entropy_curve.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ 图表已保存到: {output_path}")

# ============================================================================
# 推荐建议
# ============================================================================
print("\n" + "="*80)
print("💡 推荐建议")
print("="*80)

current_temp = 1.2
current_entropy = mean_entropies[temps.index(current_temp)]

print(f"\n当前配置: T={current_temp}, 平均熵={current_entropy:.3f} nats")
print(f"            （等价于 {current_entropy / np.log(2):.3f} bits）")

# 找到熵最稳定增长的区间（spike前）
safe_temps = [t for t in temps if t <= 1.3]
safe_entropies = [mean_entropies[temps.index(t)] for t in safe_temps]

print(f"\n✅ 安全温度区间（spike前）: {min(safe_temps):.1f} - {max(safe_temps):.1f}")
print(f"   对应熵范围: {min(safe_entropies):.3f} - {max(safe_entropies):.3f} nats")

if current_temp in safe_temps:
    print(f"\n✓ 当前T={current_temp}在安全区间内，合理！")
else:
    print(f"\n⚠️ 当前T={current_temp}可能接近或超过entropy spike")
    recommended = max(t for t in safe_temps if t < current_temp)
    print(f"   建议调整到 T={recommended:.1f}")

print("\n具体场景建议:")
print(f"  - 严肃问答/工具调用: T=0.7-1.0 (熵≈{mean_entropies[temps.index(1.0)]:.3f})")
print(f"  - 一般聊天/写作: T=1.0-1.2 (熵≈{current_entropy:.3f})")
print(f"  - Best-of-N采样: T=1.0-1.3 (但需配合top_p=0.9)")

print("\n" + "="*80)
print("🔍 测试完成！")
print("="*80)
