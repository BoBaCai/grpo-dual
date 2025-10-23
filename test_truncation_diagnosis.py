#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
截断率问题诊断脚本
测试聊天模板、多终止符、生成配置是否正确工作
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

# 设置环境变量（静音日志）
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

print("="*80)
print("截断率问题诊断工具 v1.0")
print("="*80)

# ============================================================================
# 配置
# ============================================================================
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.5
TOP_K = 30
TOP_P = 0.85
REP_PENALTY = 1.15

# 测试用例
TEST_PROMPTS = [
    "What is 2+2?",
    "Explain quantum computing in one sentence.",
    "List three colors.",
    "Context: John is 25 years old and lives in Paris.\nQuestion: Where does John live?\nOptions:\nA) London\nB) Paris\nC) Unknown\n\nPick A, B, or C, then give a one-sentence justification.",
]

# ============================================================================
# 辅助函数（从grpo_train.py复制）
# ============================================================================
def apply_chat_template(tokenizer, prompt: str, system_message: str = None) -> str:
    """应用聊天模板"""
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted
    except Exception as e:
        print(f"⚠️ 聊天模板应用失败: {e}")
        return prompt

def get_eos_token_ids(tokenizer) -> List[int]:
    """获取所有终止符ID"""
    eos_ids = []

    # 1. 标准EOS
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)

    # 2. EOT token（LLaMA-3特有）
    if hasattr(tokenizer, 'eot_token_id') and tokenizer.eot_token_id is not None:
        eos_ids.append(tokenizer.eot_token_id)
    elif '<|eot_id|>' in tokenizer.get_vocab():
        eot_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')
        eos_ids.append(eot_id)

    eos_ids = list(set(eos_ids))
    return eos_ids

def analyze_generation(tokenizer, output_ids, eos_ids, max_new_tokens):
    """分析生成结果"""
    # 解码
    text = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_clean = tokenizer.decode(output_ids, skip_special_tokens=True)

    # 统计
    length = len(output_ids)
    last_token = int(output_ids[-1].item()) if length > 0 else -1

    # 检查截断
    hit_eos = last_token in eos_ids
    is_truncated = (length >= max_new_tokens) and not hit_eos

    # 查找EOS/EOT位置
    eos_positions = []
    for i, token_id in enumerate(output_ids):
        if int(token_id.item()) in eos_ids:
            eos_positions.append((i, int(token_id.item())))

    return {
        "text": text_clean,
        "text_with_special": text,
        "length": length,
        "last_token": last_token,
        "hit_eos": hit_eos,
        "is_truncated": is_truncated,
        "eos_positions": eos_positions,
        "max_new_tokens": max_new_tokens,
    }

# ============================================================================
# 主诊断流程
# ============================================================================
def main():
    print("\n[步骤1] 加载模型和tokenizer")
    print("-" * 80)

    extra = {}
    if HF_TOKEN:
        extra["token"] = HF_TOKEN

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            **extra
        )
        print(f"✅ Tokenizer加载成功: {BASE_MODEL}")
    except Exception as e:
        print(f"❌ Tokenizer加载失败: {e}")
        return

    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 检查特殊token
    print(f"\n[步骤2] 检查特殊token")
    print("-" * 80)
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    # 检查EOT
    eot_id = None
    if hasattr(tokenizer, 'eot_token_id'):
        eot_id = tokenizer.eot_token_id
        print(f"EOT token (attribute): ID {eot_id}")

    if '<|eot_id|>' in tokenizer.get_vocab():
        eot_id_from_vocab = tokenizer.convert_tokens_to_ids('<|eot_id|>')
        print(f"EOT token (vocab): <|eot_id|> (ID: {eot_id_from_vocab})")
        if eot_id is None:
            eot_id = eot_id_from_vocab

    eos_ids = get_eos_token_ids(tokenizer)
    print(f"\n所有终止符IDs: {eos_ids}")

    # 检查是否支持聊天模板
    print(f"\n[步骤3] 测试聊天模板")
    print("-" * 80)
    test_msg = [{"role": "user", "content": "Hello"}]
    try:
        formatted = tokenizer.apply_chat_template(
            test_msg,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"✅ 聊天模板支持正常")
        print(f"示例格式:\n{formatted[:200]}...")
    except Exception as e:
        print(f"❌ 聊天模板不支持: {e}")
        print("⚠️ 将使用原始prompt")

    # 加载模型（仅CPU或GPU的一半精度）
    print(f"\n[步骤4] 加载模型（这可能需要几分钟）")
    print("-" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        print("⚠️ 使用CPU，生成会比较慢")
        dtype = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if device.type == "cuda" else None,
            **extra
        )
        print(f"✅ 模型加载成功")
        model.eval()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 运行测试
    print(f"\n[步骤5] 运行生成测试")
    print("=" * 80)

    system_msg = "You are a helpful, accurate, and unbiased assistant."

    results = []
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n测试 {i+1}/{len(TEST_PROMPTS)}")
        print("-" * 80)
        print(f"原始Prompt: {prompt[:100]}...")

        # 应用聊天模板
        formatted_prompt = apply_chat_template(tokenizer, prompt, system_msg)
        print(f"格式化Prompt长度: {len(formatted_prompt)} 字符")

        # Tokenize
        inputs = tokenizer(
            [formatted_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)

        src_len = inputs["input_ids"].shape[1]
        print(f"输入token数: {src_len}")

        # 生成（使用与训练相同的配置）
        print(f"\n生成配置:")
        print(f"  max_new_tokens: {MAX_NEW_TOKENS}")
        print(f"  temperature: {TEMPERATURE}")
        print(f"  top_k: {TOP_K}")
        print(f"  top_p: {TOP_P}")
        print(f"  repetition_penalty: {REP_PENALTY}")
        print(f"  eos_token_id: {eos_ids}")

        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    min_new_tokens=3,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    top_p=TOP_P,
                    repetition_penalty=REP_PENALTY,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_ids,  # 多终止符
                    use_cache=True,
                )

                # 提取生成部分
                response_ids = outputs[0, src_len:]

                # 分析结果
                result = analyze_generation(tokenizer, response_ids, eos_ids, MAX_NEW_TOKENS)
                results.append(result)

                # 打印结果
                print(f"\n生成结果:")
                print(f"  长度: {result['length']} tokens")
                print(f"  最后token ID: {result['last_token']}")
                print(f"  命中EOS/EOT: {'✅ 是' if result['hit_eos'] else '❌ 否'}")
                print(f"  是否截断: {'⚠️ 是' if result['is_truncated'] else '✅ 否'}")

                if result['eos_positions']:
                    print(f"  EOS/EOT位置: {result['eos_positions']}")

                print(f"\n生成文本:")
                print(f"  {result['text'][:200]}...")

                if result['is_truncated']:
                    print(f"\n⚠️ 警告：此次生成被截断！")
                    print(f"  - 长度达到上限但未命中EOS/EOT")
                    print(f"  - 最后token: {result['last_token']} (不在 {eos_ids} 中)")

            except Exception as e:
                print(f"❌ 生成失败: {e}")
                import traceback
                traceback.print_exc()

    # 汇总统计
    print("\n" + "=" * 80)
    print("[步骤6] 测试结果汇总")
    print("=" * 80)

    total = len(results)
    truncated_count = sum(1 for r in results if r['is_truncated'])
    hit_eos_count = sum(1 for r in results if r['hit_eos'])
    avg_length = sum(r['length'] for r in results) / max(1, total)

    print(f"总测试数: {total}")
    print(f"截断数: {truncated_count} ({truncated_count/max(1,total)*100:.1f}%)")
    print(f"命中EOS/EOT数: {hit_eos_count} ({hit_eos_count/max(1,total)*100:.1f}%)")
    print(f"平均长度: {avg_length:.1f} tokens")

    print(f"\n诊断结论:")
    if truncated_count == 0:
        print("✅ 生成正常！所有测试都正确终止，未发生截断。")
    elif truncated_count < total * 0.2:
        print("⚠️ 轻微问题：少数生成被截断（<20%）。")
        print("   建议：调整采样参数或增加max_new_tokens。")
    else:
        print("❌ 严重问题：大量生成被截断（≥20%）！")
        print("\n可能原因：")
        if hit_eos_count == 0:
            print("  1. ❌ 模型从未生成EOS/EOT token")
            print("     → 检查聊天模板是否正确")
            print("     → 检查eos_token_id是否正确传递")
        else:
            print("  1. ⚠️ 模型偶尔生成EOS/EOT，但不够频繁")
            print("     → 调整采样参数（降低temperature，增大repetition_penalty）")

        print("  2. ⚠️ max_new_tokens设置过小")
        print(f"     → 当前: {MAX_NEW_TOKENS}, 平均长度: {avg_length:.1f}")

        print("\n建议修复方案：")
        print("  - 确认聊天模板已正确应用（步骤3检查）")
        print("  - 确认eos_token_id包含所有终止符（步骤2检查）")
        print("  - 降低temperature（当前{:.1f} → 0.3-0.4）".format(TEMPERATURE))
        print("  - 增大repetition_penalty（当前{:.2f} → 1.2-1.3）".format(REP_PENALTY))

    print("\n" + "=" * 80)
    print("诊断完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
