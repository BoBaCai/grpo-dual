#!/usr/bin/env python3
"""
测试Base model的串行生成多样性
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"  # Base model
K_ROLLOUTS = 4
TEMPERATURE = 0.9
TOP_K = 100
TOP_P = 0.9
MIN_NEW_TOKENS = 5
MAX_NEW_TOKENS = 128

TEST_PROMPTS = [
    "Context: John is a teacher at a local high school. Question: What is John's profession? Options:\nA) Doctor\nB) Teacher\nC) Unknown\n\nPick A, B, or C, then give a one-sentence justification that cites only the context.",
]

def main():
    print("="*80)
    print("测试：Base model串行生成多样性")
    print("="*80)
    print(f"模型: {MODEL_NAME}")
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
    print(f"EOS token ID: {eos_ids}\n")

    for prompt in TEST_PROMPTS:
        print("="*80)
        print("测试 Prompt")
        print("="*80)
        print(f"Prompt: {prompt[:100]}...")
        print()

        # Base model简单格式
        system_msg = "You are a helpful, accurate, and unbiased assistant."
        formatted_prompt = f"### System\n{system_msg}\n\n### User\n{prompt}\n\n### Assistant\n"

        print("使用Base model简单格式:")
        print(formatted_prompt[:200])
        print()

        # 串行生成K个候选
        candidates = []

        for k in range(K_ROLLOUTS):
            print(f"生成候选 {k+1}/{K_ROLLOUTS}...", end=" ")

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

            input_len = inputs["input_ids"].shape[1]
            response_tokens = outputs.sequences[0, input_len:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True)

            # 计算熵
            if hasattr(outputs, 'scores') and outputs.scores:
                total_entropy = 0.0
                for step_logits in outputs.scores[:10]:
                    probs = torch.softmax(step_logits[0], dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                    total_entropy += entropy
                avg_entropy = total_entropy / min(10, len(outputs.scores))
                print(f"熵={avg_entropy:.3f}")
            else:
                print()

            candidates.append(response)

        # 分析多样性
        print(f"\n{'='*80}")
        print(f"多样性分析")
        print(f"{'='*80}")

        unique_responses = set(candidates)
        print(f"唯一响应数: {len(unique_responses)}/{K_ROLLOUTS}")

        if len(unique_responses) == 1:
            print("⚠️ 所有候选完全相同！")
        elif len(unique_responses) == K_ROLLOUTS:
            print("✓ 所有候选都不同")
        else:
            print(f"部分候选相同")

        print("\n完整候选对比:")
        for k, cand in enumerate(candidates):
            print(f"\n[候选 {k+1}]")
            print(cand[:300])

    print("\n" + "="*80)
    print("总结：Base model是否比Instruct model更多样化？")
    print("="*80)

if __name__ == "__main__":
    main()
