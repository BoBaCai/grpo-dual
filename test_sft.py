#!/usr/bin/env python3
"""测试SFT模型质量"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载SFT checkpoint
checkpoint_path = "checkpoints/sft_model"  # 修改为实际路径
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.float16)
model.eval()

# 测试prompt
test_prompts = [
    "What is the capital of France?",
    "Explain photosynthesis in simple terms.",
    "Is the sky blue? Answer yes or no."
]

print("=== SFT Model Quality Test ===\n")

for prompt in test_prompts:
    # 应用chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 生成
    inputs = tokenizer(formatted, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Contains gibberish: {'Yes' if any(c in response for c in ['uang', 'CDF', 'SeiteNr']) else 'No'}")
    print("-" * 70)
