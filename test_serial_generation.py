#!/usr/bin/env python3
"""
实验：测试串行生成是否能产生不同的K个候选

目的：验证串行生成修复是否有效
- 如果K个候选完全相同 → 串行生成无效，问题是模型太deterministic
- 如果K个候选有差异 → 串行生成有效，可以继续训练
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

# 配置
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
K_ROLLOUTS = 4
TEMPERATURE = 0.9
TOP_K = 100
TOP_P = 0.9
MIN_NEW_TOKENS = 5
MAX_NEW_TOKENS = 128

# 测试prompts（BBQ类型）
TEST_PROMPTS = [
    "Context: John is a teacher at a local high school. Question: What is John's profession? Options:\nA) Doctor\nB) Teacher\nC) Unknown\n\nPick A, B, or C, then give a one-sentence justification that cites only the context.",

    "Context: Sarah loves playing basketball and practices every day. Question: What sport does Sarah play? Options:\nA) Soccer\nB) Basketball\nC) Unknown\n\nPick A, B, or C, then give a one-sentence justification that cites only the context.",
]

def main():
    print("="*80)
    print("实验：串行生成多样性测试")
    print("="*80)
    print(f"模型: {MODEL_NAME}")
    print(f"K候选数: {K_ROLLOUTS}")
    print(f"采样参数: temperature={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}")
    print(f"生成长度: min={MIN_NEW_TOKENS}, max={MAX_NEW_TOKENS}")
    print()

    # 加载模型
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("✓ 模型加载完成\n")

    # EOS tokens
    eos_ids = [tokenizer.eos_token_id]
    if hasattr(tokenizer, 'eos_token_id'):
        eos_ids.append(tokenizer.eos_token_id)
    if '<|eot_id|>' in tokenizer.get_vocab():
        eos_ids.append(tokenizer.convert_tokens_to_ids('<|eot_id|>'))
    eos_ids = list(set(eos_ids))
    print(f"EOS token IDs: {eos_ids}\n")

    # 对每个prompt进行测试
    for prompt_idx, prompt in enumerate(TEST_PROMPTS):
        print("="*80)
        print(f"测试 Prompt {prompt_idx + 1}/{len(TEST_PROMPTS)}")
        print("="*80)
        print(f"Prompt: {prompt[:100]}...")
        print()

        # 应用chat template
        messages = [
            {"role": "system", "content": "You are a helpful, accurate, and unbiased assistant."},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 串行生成K个候选
        candidates = []
        entropies = []

        for k in range(K_ROLLOUTS):
            print(f"\n--- 生成候选 {k+1}/{K_ROLLOUTS} ---")

            inputs = tokenizer([formatted_prompt], return_tensors="pt", padding=True).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    min_new_tokens=MIN_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    top_p=TOP_P,
                    eos_token_id=eos_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # 解码
            input_len = inputs["input_ids"].shape[1]
            response_tokens = outputs.sequences[0, input_len:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True)

            # 计算熵（近似，使用生成的scores）
            if hasattr(outputs, 'scores') and outputs.scores:
                # scores是每个生成步骤的logits
                # 计算平均熵
                total_entropy = 0.0
                for step_logits in outputs.scores[:10]:  # 只看前10个token
                    probs = torch.softmax(step_logits[0], dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                    total_entropy += entropy
                avg_entropy = total_entropy / min(10, len(outputs.scores))
                entropies.append(avg_entropy)
            else:
                avg_entropy = None

            candidates.append(response)

            print(f"长度: {len(response_tokens)} tokens")
            if avg_entropy is not None:
                print(f"熵(前10 tokens平均): {avg_entropy:.4f}")
            print(f"内容: {response[:200]}")

        # 分析多样性
        print(f"\n{'='*80}")
        print(f"多样性分析 - Prompt {prompt_idx + 1}")
        print(f"{'='*80}")

        # 检查是否完全相同
        unique_responses = set(candidates)
        print(f"唯一响应数: {len(unique_responses)}/{K_ROLLOUTS}")

        if len(unique_responses) == 1:
            print("⚠️ 所有候选完全相同！")
        elif len(unique_responses) == K_ROLLOUTS:
            print("✓ 所有候选都不同")
        else:
            print(f"部分候选相同 ({K_ROLLOUTS - len(unique_responses)}个重复)")

        # 打印所有候选以便对比
        print("\n完整候选对比:")
        for k, cand in enumerate(candidates):
            print(f"\n[候选 {k+1}]")
            print(cand)

        # 熵统计
        if entropies:
            print(f"\n熵统计: min={min(entropies):.4f}, max={max(entropies):.4f}, avg={sum(entropies)/len(entropies):.4f}")
            if max(entropies) < 0.5:
                print("⚠️ 熵极低！模型输出高度确定")
            elif max(entropies) < 1.5:
                print("⚠️ 熵偏低，可能需要更激进的采样")
            else:
                print("✓ 熵正常")

        print()

    print("="*80)
    print("实验总结")
    print("="*80)
    print("如果候选完全相同：")
    print("  → 串行生成无效，模型太deterministic")
    print("  → 需要：更激进采样(temp=1.5) 或 换Base model")
    print()
    print("如果候选有差异：")
    print("  → 串行生成有效！")
    print("  → 可以继续完整训练，观察reward std是否>0")
    print("="*80)

if __name__ == "__main__":
    main()
