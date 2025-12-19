# =============================================================================
# API Keys 设置脚本（轻量级，适合 kernel 重启后快速加载）
# =============================================================================
# 用途：Jupyter kernel 重启后，不需要重新运行整个安装脚本
#      只需运行这个脚本就能重新加载 API keys
#
# 使用方法：
#   1. 将下面的占位符替换为你的真实 API keys
#   2. 运行这个 cell：%run src/grpo/setup_api_keys.py
#   3. 然后就可以运行训练了：%run src/grpo/trainer.py
# =============================================================================

import os

# ============================================================
# 👉 在这里填写你的 API Keys
# ============================================================

# OpenAI API Key（必需）
# 用于：LLM Judge (gpt-4o-mini) 评估模型生成的回答
# 获取地址：https://platform.openai.com/api-keys
OPENAI_API_KEY = "sk-..."  # 👈 替换为你的 OpenAI API key

# Hugging Face Token（必需）
# 用于：下载 Llama-3-8B-Instruct 模型
# 获取地址：https://huggingface.co/settings/tokens
HF_TOKEN = "hf_..."  # 👈 替换为你的 HF token

# Anthropic API Key（可选）
# 用于：如果你想使用 Claude 作为 LLM Judge
# 获取地址：https://console.anthropic.com/settings/keys
ANTHROPIC_API_KEY = ""  # 可选，留空即可

# ============================================================
# 自动设置环境变量
# ============================================================

if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-..."):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    pass  # 保持未设置

if HF_TOKEN and not HF_TOKEN.startswith("hf_..."):
    os.environ["HF_TOKEN"] = HF_TOKEN
else:
    pass  # 保持未设置

if ANTHROPIC_API_KEY and not ANTHROPIC_API_KEY.startswith("sk-ant-"):
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

# ============================================================
# 验证设置
# ============================================================

print("="*80)
print("API Keys 设置状态")
print("="*80)

def check_key(key_name, required=True):
    value = os.environ.get(key_name, "")
    status = "✓" if value and not value.startswith("sk-...") and not value.startswith("hf_...") else "✗"
    masked = value[:10] + "..." if len(value) > 10 else value
    req_label = "【必需】" if required else "【可选】"
    print(f"{status} {key_name:25s} {req_label:10s} {masked if value else '未设置'}")
    return bool(value and not value.startswith("sk-...") and not value.startswith("hf_..."))

openai_ok = check_key("OPENAI_API_KEY", required=True)
hf_ok = check_key("HF_TOKEN", required=True)
anthropic_ok = check_key("ANTHROPIC_API_KEY", required=False)

print("="*80)

if openai_ok and hf_ok:
    print("✓ 必需的 API Keys 已正确设置，可以开始训练！")
    print("\n下一步：运行训练脚本")
    print("  %run src/grpo/trainer.py")
else:
    print("✗ 请检查并设置必需的 API Keys")
    if not openai_ok:
        print("  - OPENAI_API_KEY 未设置或仍是占位符")
    if not hf_ok:
        print("  - HF_TOKEN 未设置或仍是占位符")
    print("\n提示：请修改上面的代码，将占位符替换为真实的 API keys")

print("="*80)
