#!/usr/bin/env python3
"""
自动化 Entropy 崩溃诊断脚本

这个脚本会实际执行诊断代码，而不只是打印指导信息。

功能：
1. 自动解析训练日志
2. 提取关键指标（Entropy, Reward, KL, Logits）
3. 自动判断问题所在
4. 生成诊断报告

用法：
  # 诊断训练日志
  python scripts/auto_diagnose_entropy.py --log train.log

  # 测试 base model entropy（需要模型路径）
  python scripts/auto_diagnose_entropy.py --test-base-model meta-llama/Llama-3.2-1B-Instruct

  # 完整诊断（日志 + 模型）
  python scripts/auto_diagnose_entropy.py --log train.log --test-base-model MODEL_PATH
"""

import re
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

print("="*80)
print("🔬 自动化 Entropy 崩溃诊断")
print("="*80)

# ============================================================================
# 日志解析器
# ============================================================================

class LogParser:
    """解析训练日志，提取关键指标"""

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.steps = defaultdict(dict)

    def parse(self):
        """解析日志文件"""
        if not self.log_path.exists():
            print(f"❌ 日志文件不存在: {self.log_path}")
            return False

        print(f"\n📂 解析日志: {self.log_path}")

        with open(self.log_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取 Logits 信息
        logits_pattern = r'\[Step (\d+) Logits\].*?Max logit=([\d.]+).*?Gap=([\d.]+).*?Top5=\[([\d.e\-,\s]+)\]'
        for match in re.finditer(logits_pattern, content, re.DOTALL):
            step = int(match.group(1))
            max_logit = float(match.group(2))
            gap = float(match.group(3))
            top5_str = match.group(4)
            top5 = [float(x.strip()) for x in top5_str.split(',')]

            self.steps[step]['max_logit'] = max_logit
            self.steps[step]['gap'] = gap
            self.steps[step]['top5'] = top5
            self.steps[step]['max_prob'] = top5[0] if top5 else None

        # 提取 Entropy 信息
        entropy_pattern = r'\[Fairness诊断@step(\d+)\].*?Entropy.*?mean.*?=([\d.]+)'
        for match in re.finditer(entropy_pattern, content, re.DOTALL):
            step = int(match.group(1)) - 1  # 日志中是 step+1
            entropy_mean = float(match.group(2))
            self.steps[step]['entropy_mean'] = entropy_mean

        # 提取 Reward 信息
        reward_pattern = r'\[Fairness诊断@step(\d+)\].*?Reward.*?\(F\).*?=([\d.\-+]+).*?\(H\).*?=([\d.\-+]+)'
        for match in re.finditer(reward_pattern, content, re.DOTALL):
            step = int(match.group(1)) - 1
            reward_f = float(match.group(2))
            reward_h = float(match.group(3))
            self.steps[step]['reward_f'] = reward_f
            self.steps[step]['reward_h'] = reward_h

        # 提取 EOS Suppressor 信息
        eos_pattern = r'Call#(\d+).*?EOS.*?阻止.*?\((\d+)/(\d+)\)'
        eos_blocks = []
        for match in re.finditer(eos_pattern, content):
            call = int(match.group(1))
            blocked = int(match.group(2))
            total = int(match.group(3))
            eos_blocks.append((call, blocked, total))

        if eos_blocks:
            # 计算平均阻止率
            avg_block_rate = np.mean([b/t for _, b, t in eos_blocks])
            for step in self.steps:
                self.steps[step]['eos_block_rate'] = avg_block_rate

        print(f"✅ 成功解析 {len(self.steps)} 个训练步")
        return len(self.steps) > 0

    def get_stats(self):
        """获取统计信息"""
        if not self.steps:
            return None

        stats = {
            'entropy_mean': [],
            'max_prob': [],
            'gap': [],
            'reward_f': [],
            'reward_h': [],
            'eos_block_rate': []
        }

        for step_data in self.steps.values():
            for key in stats.keys():
                if key in step_data and step_data[key] is not None:
                    stats[key].append(step_data[key])

        # 计算统计量
        result = {}
        for key, values in stats.items():
            if values:
                result[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }

        return result

# ============================================================================
# Base Model Entropy 测试器
# ============================================================================

class BaseModelTester:
    """测试 base model 的 entropy"""

    def __init__(self, model_name):
        self.model_name = model_name

    def test(self):
        """测试 base model entropy"""
        print(f"\n🧪 测试 Base Model: {self.model_name}")

        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print("  加载模型...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            model.eval()

            # 测试 prompt
            prompts = [
                "Context: John has 15 years of experience. Mary has 3 years. Question: Who is more experienced? A) John B) Mary C) Unknown",
                "Answer the following question: What is 2+2?",
                "Complete this sentence: The capital of France is"
            ]

            all_entropies = []
            all_max_probs = []

            for prompt in prompts:
                print(f"\n  测试 prompt: {prompt[:50]}...")

                inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=True,
                        temperature=0.9,
                        return_dict_in_generate=True,
                        output_scores=True
                    )

                # 计算每步的 entropy
                entropies = []
                max_probs = []
                for scores in outputs.scores[:10]:  # 前10个token
                    probs = F.softmax(scores[0] / 0.9, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum()
                    max_prob = probs.max().item()

                    entropies.append(entropy.item())
                    max_probs.append(max_prob)

                avg_entropy = np.mean(entropies)
                avg_max_prob = np.mean(max_probs)

                all_entropies.extend(entropies)
                all_max_probs.extend(max_probs)

                print(f"    Avg entropy: {avg_entropy:.3f}")
                print(f"    Avg max_prob: {avg_max_prob:.4f}")

            overall_entropy = np.mean(all_entropies)
            overall_max_prob = np.mean(all_max_probs)

            return {
                'mean_entropy': overall_entropy,
                'mean_max_prob': overall_max_prob,
                'all_entropies': all_entropies,
                'all_max_probs': all_max_probs
            }

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return None

# ============================================================================
# 诊断引擎
# ============================================================================

class DiagnosticEngine:
    """诊断引擎，分析数据并给出结论"""

    def __init__(self):
        self.issues = []
        self.suggestions = []

    def analyze_training_log(self, stats):
        """分析训练日志统计"""
        print("\n" + "="*80)
        print("📊 训练日志分析")
        print("="*80)

        if not stats:
            print("❌ 无有效数据")
            return

        # 分析 Entropy
        if 'entropy_mean' in stats:
            ent = stats['entropy_mean']
            print(f"\n🔍 Entropy 分析:")
            print(f"  平均: {ent['mean']:.4f}")
            print(f"  中位数: {ent['median']:.4f}")
            print(f"  范围: [{ent['min']:.4f}, {ent['max']:.4f}]")

            if ent['mean'] < 0.05:
                self.issues.append("🔴 Entropy 严重崩溃 (< 0.05)")
                self.suggestions.append("立即应用 LOW_MEMORY_MODE=True 快速修复")
            elif ent['mean'] < 0.3:
                self.issues.append("🟡 Entropy 偏低 (< 0.3)")
                self.suggestions.append("提升 ENTROPY_COEF 或检查梯度符号")
            else:
                print("  ✅ Entropy 正常")

        # 分析 Max Prob
        if 'max_prob' in stats:
            mp = stats['max_prob']
            print(f"\n🔍 Max Probability 分析:")
            print(f"  平均: {mp['mean']:.4f}")
            print(f"  中位数: {mp['median']:.4f}")

            if mp['mean'] > 0.99:
                self.issues.append("🔴 Max Prob 过高 (> 99%)")
                self.suggestions.append("模型输出极度确定，配合 Entropy 崩溃")
            elif mp['mean'] > 0.90:
                self.issues.append("🟡 Max Prob 偏高 (> 90%)")

        # 分析 Logit Gap
        if 'gap' in stats:
            gap = stats['gap']
            print(f"\n🔍 Logit Gap 分析:")
            print(f"  平均: {gap['mean']:.3f}")
            print(f"  中位数: {gap['median']:.3f}")

            if gap['mean'] > 7:
                self.issues.append("🔴 Logit Gap 过大 (> 7)")
                self.suggestions.append("Logits 极度尖锐，可能是 base model 问题")
            elif gap['mean'] > 5:
                self.issues.append("🟡 Logit Gap 偏大 (> 5)")

        # 分析 Reward
        if 'reward_f' in stats and 'reward_h' in stats:
            rf = stats['reward_f']
            rh = stats['reward_h']
            print(f"\n🔍 Reward 分析:")
            print(f"  Fairness: mean={rf['mean']:.3f}, std={rf['std']:.3f}")
            print(f"  Hallucination: mean={rh['mean']:.3f}, std={rh['std']:.3f}")

            if rf['std'] < 0.1:
                self.issues.append("🔴 Fairness Reward 无变化 (std < 0.1)")
                self.suggestions.append("Reward 信号退化，检查 judge 评估逻辑")

            if rh['std'] < 0.1:
                self.issues.append("🔴 Hallucination Reward 无变化 (std < 0.1)")

        # 分析 EOS Suppressor
        if 'eos_block_rate' in stats:
            eos = stats['eos_block_rate']
            print(f"\n🔍 EOS Suppressor 分析:")
            print(f"  平均阻止率: {eos['mean']*100:.1f}%")

            if eos['mean'] > 0.8:
                self.issues.append("🔴 EOS Suppressor 触发率过高 (> 80%)")
                self.suggestions.append("MIN_NEW_TOKENS 与 SFT target 不匹配")
            elif eos['mean'] > 0.5:
                self.issues.append("🟡 EOS Suppressor 触发率偏高 (> 50%)")

    def analyze_base_model(self, result):
        """分析 base model 测试结果"""
        print("\n" + "="*80)
        print("📊 Base Model 分析")
        print("="*80)

        if not result:
            print("❌ 无测试数据")
            return

        mean_ent = result['mean_entropy']
        mean_mp = result['mean_max_prob']

        print(f"\n🔍 Base Model Entropy:")
        print(f"  平均: {mean_ent:.3f}")
        print(f"  Max prob: {mean_mp:.4f}")

        if mean_ent < 0.5:
            self.issues.append("🔴 Base Model Entropy 过低 (< 0.5)")
            self.suggestions.append("Base model 本身就有问题，考虑换模型或降低 KL penalty")
        elif mean_ent < 1.5:
            self.issues.append("🟡 Base Model Entropy 偏低 (< 1.5)")
        else:
            print("  ✅ Base Model Entropy 正常")

    def generate_report(self):
        """生成诊断报告"""
        print("\n" + "="*80)
        print("📋 诊断报告")
        print("="*80)

        if not self.issues:
            print("\n✅ 未发现明显问题")
            return

        print(f"\n发现 {len(self.issues)} 个问题:")
        for i, issue in enumerate(self.issues, 1):
            print(f"  {i}. {issue}")

        if self.suggestions:
            print(f"\n💡 修复建议 ({len(self.suggestions)} 条):")
            for i, suggestion in enumerate(self.suggestions, 1):
                print(f"  {i}. {suggestion}")

        # 综合判断
        print("\n" + "="*80)
        print("🎯 综合诊断结论")
        print("="*80)

        # 检查是否是 Entropy 梯度符号问题
        has_entropy_collapse = any("Entropy 严重崩溃" in issue for issue in self.issues)
        has_high_max_prob = any("Max Prob 过高" in issue for issue in self.issues)
        has_high_eos = any("EOS Suppressor 触发率过高" in issue for issue in self.issues)

        if has_entropy_collapse and has_high_max_prob:
            print("\n🔴 高度怀疑：Entropy 梯度符号错误！")
            print("\n原因：")
            print("  1. Entropy 崩溃到 < 0.05")
            print("  2. Max Prob > 99%")
            print("  3. 这符合'梯度符号反了'的特征")
            print("\n🚀 立即修复:")
            print("  trainer.py 第 286 行: LOW_MEMORY_MODE = False → True")
            print("\n预期效果:")
            print("  - Entropy: 0.005 → 0.5-1.5")
            print("  - Max Prob: 99.9% → 60-85%")

        if has_high_eos:
            print("\n🟡 发现：MIN_NEW_TOKENS 不匹配")
            print("\n原因：")
            print("  EOS Suppressor 触发率 > 80%")
            print("  说明 MIN_NEW_TOKENS 远小于实际生成需求")
            print("\n🚀 修复:")
            print("  trainer.py 第 226 行: MIN_NEW_TOKENS_TRAIN = 5 → 30")

# ============================================================================
# 主函数
# ============================================================================

def diagnose(log_path=None, base_model=None):
    """
    诊断函数（可在 notebook 中直接调用）

    Args:
        log_path: 训练日志文件路径
        base_model: Base model 名称/路径
    """
    if not log_path and not base_model:
        print("❌ 请指定 log_path 或 base_model")
        print("\n用法:")
        print("  diagnose(log_path='train.log')")
        print("  diagnose(base_model='meta-llama/Llama-3.2-1B-Instruct')")
        print("  diagnose(log_path='train.log', base_model='MODEL_NAME')")
        return None

    engine = DiagnosticEngine()

    # 分析训练日志
    if log_path:
        parser = LogParser(log_path)
        if parser.parse():
            stats = parser.get_stats()
            engine.analyze_training_log(stats)

    # 测试 base model
    if base_model:
        tester = BaseModelTester(base_model)
        result = tester.test()
        if result:
            engine.analyze_base_model(result)

    # 生成报告
    engine.generate_report()

    print("\n" + "="*80)
    print("✅ 诊断完成")
    print("="*80)

    return engine

def main():
    """命令行入口（检测 notebook 环境）"""
    # 检测是否在 notebook 环境
    try:
        get_ipython()
        in_notebook = True
    except NameError:
        in_notebook = False

    if in_notebook:
        print("🔔 检测到 Jupyter notebook 环境")
        print("\n在 notebook 中使用函数式接口：")
        print("  from auto_diagnose_entropy import diagnose")
        print("  diagnose(log_path='train.log')")
        print("  diagnose(base_model='MODEL_NAME')")
        return

    # 命令行模式
    parser = argparse.ArgumentParser(description='自动化 Entropy 崩溃诊断')
    parser.add_argument('--log', type=str, help='训练日志文件路径')
    parser.add_argument('--test-base-model', type=str, help='Base model 名称/路径')

    args = parser.parse_args()

    diagnose(log_path=args.log, base_model=args.test_base_model)

if __name__ == "__main__":
    main()
