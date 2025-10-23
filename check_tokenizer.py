#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速检查 LLaMA-3 tokenizer 的终止符配置
"""

import os
from transformers import AutoTokenizer

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

print("="*80)
print("LLaMA-3 Tokenizer 终止符检查")
print("="*80)

extra = {}
if HF_TOKEN:
    extra["token"] = HF_TOKEN

try:
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        **extra
    )
    print(f"✅ Tokenizer加载成功\n")
except Exception as e:
    print(f"❌ Tokenizer加载失败: {e}")
    exit(1)

# 检查所有可能的终止符
print("[1] 检查 tokenizer 属性")
print("-" * 80)
print(f"tokenizer.eos_token: {tokenizer.eos_token}")
print(f"tokenizer.eos_token_id: {tokenizer.eos_token_id}")
print(f"tokenizer.pad_token: {tokenizer.pad_token}")
print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")

if hasattr(tokenizer, 'eot_token'):
    print(f"tokenizer.eot_token: {tokenizer.eot_token}")
if hasattr(tokenizer, 'eot_token_id'):
    print(f"tokenizer.eot_token_id: {tokenizer.eot_token_id}")

print(f"\n[2] 检查 vocab 中的特殊token")
print("-" * 80)

vocab = tokenizer.get_vocab()

# 查找所有包含 'end' 或 'eot' 的token
special_tokens = []
for token, token_id in vocab.items():
    if 'end' in token.lower() or 'eot' in token.lower():
        special_tokens.append((token, token_id))

special_tokens.sort(key=lambda x: x[1])

for token, token_id in special_tokens:
    print(f"  {token_id:6d}: {token}")

print(f"\n[3] 检查关键终止符")
print("-" * 80)

critical_tokens = [
    '<|end_of_text|>',
    '<|eot_id|>',
    '<|endoftext|>',
    '</s>',
]

found_ids = []
for token in critical_tokens:
    if token in vocab:
        token_id = tokenizer.convert_tokens_to_ids(token)
        found_ids.append(token_id)
        print(f"✅ {token:20s} -> ID: {token_id}")
    else:
        print(f"❌ {token:20s} -> 不存在")

print(f"\n[4] 推荐配置")
print("-" * 80)

if found_ids:
    print(f"建议使用的终止符ID列表: {found_ids}")

    # 验证解码
    print(f"\n解码验证:")
    for tid in found_ids:
        decoded = tokenizer.decode([tid])
        print(f"  ID {tid} -> '{decoded}'")
else:
    print(f"⚠️ 未找到任何关键终止符，使用默认: [tokenizer.eos_token_id]")

print(f"\n[5] 测试聊天模板")
print("-" * 80)

try:
    messages = [{"role": "user", "content": "Hello"}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("✅ 聊天模板支持正常")
    print(f"示例输出:\n{formatted}")

    # 检查模板中是否包含终止符
    for token in critical_tokens:
        if token in formatted:
            print(f"\n✅ 模板中包含 {token}")

except Exception as e:
    print(f"❌ 聊天模板不支持: {e}")

print("\n" + "="*80)
print("检查完成！")
print("="*80)
