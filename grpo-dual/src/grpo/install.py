# 适用于 Jupyter Notebook Cell 的安装脚本
# 可以直接复制粘贴到 notebook cell 中运行

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# =============================================================================
# RunPod 路径配置（解决磁盘空间不足问题）
# =============================================================================
# 检测是否在 RunPod 环境（/workspace 存在）
workspace_path = Path("/workspace")
if workspace_path.exists() and workspace_path.is_dir():
    # 设置 HuggingFace 缓存到 /workspace（大容量磁盘）
    cache_dir = workspace_path / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    print(f"✓ RunPod 环境检测到，HuggingFace 缓存目录设置为: {cache_dir}")
else:
    print("ℹ️ 非 RunPod 环境，使用默认缓存目录")

print("")

# =============================================================================
# 👉 第一步：在这里设置你的 API Keys（先填写，再运行安装）
# =============================================================================
# 请将下面的占位符替换为你的真实 API keys，然后运行这个 cell

# OpenAI API Key（必需）- 用于 GPT-4o-mini Judge
# 获取地址：https://platform.openai.com/api-keys
# ⚠️ 重要：API key 必须是一行，不能有空格或换行符
# 示例格式：sk-proj-... 或 sk-...
OPENAI_API_KEY = "sk-..."  # 👈 替换为你的 OpenAI API key（确保没有多余空格）

# Hugging Face Token（必需）- 用于下载 Llama 模型
# 获取地址：https://huggingface.co/settings/tokens
HF_TOKEN = "hf_..."  # 👈 替换为你的 Hugging Face token

# Anthropic API Key（必需）- 用于 Claude 3.5 Haiku Judge
# 获取地址：https://console.anthropic.com/settings/keys
# 示例格式：sk-ant-...
ANTHROPIC_API_KEY = ""  # 👈 替换为你的 Anthropic API key

# Gemini API Key（必需）- 用于 Gemini 2.5 Flash Judge
# 获取地址：https://aistudio.google.com/app/apikey
# 示例格式：AI...
GEMINI_API_KEY = ""  # 👈 替换为你的 Gemini API key

# 设置环境变量（自动清理多余空格）
if OPENAI_API_KEY and OPENAI_API_KEY != "sk-...":
    cleaned_key = OPENAI_API_KEY.strip()  # 清理首尾空格
    os.environ["OPENAI_API_KEY"] = cleaned_key
    print("✓ OPENAI_API_KEY 已设置")
    print(f"  格式: {cleaned_key[:10]}...{cleaned_key[-4:]}")  # 显示前10位和后4位以验证
else:
    print("⚠️ OPENAI_API_KEY 未设置或仍是占位符")

if HF_TOKEN and HF_TOKEN != "hf_...":
    cleaned_token = HF_TOKEN.strip()  # 清理首尾空格
    os.environ["HF_TOKEN"] = cleaned_token
    print("✓ HF_TOKEN 已设置")
    print(f"  格式: {cleaned_token[:10]}...{cleaned_token[-4:]}")
else:
    print("⚠️ HF_TOKEN 未设置或仍是占位符")

if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "" and ANTHROPIC_API_KEY != "sk-ant-...":
    cleaned_anthropic = ANTHROPIC_API_KEY.strip()  # 清理首尾空格
    os.environ["ANTHROPIC_API_KEY"] = cleaned_anthropic
    print("✓ ANTHROPIC_API_KEY 已设置")
    print(f"  格式: {cleaned_anthropic[:10]}...{cleaned_anthropic[-4:]}")
else:
    print("⚠️ ANTHROPIC_API_KEY 未设置（多 LLM Judge 需要此 key）")

if GEMINI_API_KEY and GEMINI_API_KEY != "":
    cleaned_gemini = GEMINI_API_KEY.strip()  # 清理首尾空格
    os.environ["GEMINI_API_KEY"] = cleaned_gemini
    print("✓ GEMINI_API_KEY 已设置")
    print(f"  格式: {cleaned_gemini[:10]}...{cleaned_gemini[-4:]}")
else:
    print("⚠️ GEMINI_API_KEY 未设置（多 LLM Judge 需要此 key）")

print("")

# =============================================================================
# 安装开始
# =============================================================================

def run(cmd, desc=""):
    if desc:
        print(f"\n{'='*80}\n{desc}\n{'='*80}")
    print(f"$ {cmd}\n")
    return subprocess.run(cmd, shell=True, capture_output=False, text=True)

print("="*80)
print("Multi-Objective GRPO 完整安装 - 最终版")
print("="*80)
print("\n包含:")
print("- PyTorch 2.x (最新可用版本)")
print("- Transformers 4.44.2 (稳定版，兼容 peft)")
print("- PEFT 0.9.0")
print("- Jinja2 >= 3.1.0 (支持 chat template)")
print("- Flash Attention 2 (可选加速)")
print("- 所有必要依赖")
print("\n预计时间: 10-15分钟 (包含 Flash Attention 2 编译)")
print("="*80)

# 步骤 1: 完全卸载
run(
    f"{sys.executable} -m pip uninstall -y "
    f"torch torchvision torchaudio "
    f"transformers peft accelerate bitsandbytes datasets flash-attn",
    "步骤 1/7: 卸载现有包"
)

# 步骤 2: 清理
run(f"{sys.executable} -m pip cache purge", "步骤 2/7: 清理缓存")

# 步骤 3: 安装 PyTorch (使用最新稳定版)
run(
    f"{sys.executable} -m pip install "
    f"torch torchvision torchaudio "
    f"--index-url https://download.pytorch.org/whl/cu121",
    "步骤 3/7: 安装 PyTorch (CUDA 12.1)"
)

# 步骤 4: 安装 Transformers (固定版本，确保与 peft 0.9.0 兼容)
run(
    f"{sys.executable} -m pip install 'transformers==4.44.2'",
    "步骤 4/7: 安装 Transformers (4.44.2，兼容 peft 0.9.0)"
)

# 步骤 5: 安装其他核心包
run(
    f"{sys.executable} -m pip install "
    f"peft==0.9.0 "
    f"accelerate==0.27.0 "
    f"'bitsandbytes>=0.43.0' "
    f"datasets "
    f"openai "
    f"anthropic "
    f"google-generativeai "
    f"hf_transfer "
    f"sentencepiece "
    f"protobuf "
    f"'requests>=2.32.2' "
    f"'jinja2>=3.1.0' "
    f"scipy "
    f"tqdm",
    "步骤 5/7: 安装其他依赖（含 jinja2>=3.1.0 和多 LLM Judge APIs）"
)

# 步骤 6: 安装 Flash Attention 2（可选，加速训练）
run(
    f"{sys.executable} -m pip install flash-attn --no-build-isolation",
    "步骤 6/7: 安装 Flash Attention 2（可选加速，需要编译，可能需要几分钟）"
)

# 步骤 7: 验证
print("\n" + "="*80)
print("步骤 7/7: 验证安装")
print("="*80)

# 验证代码
verify_code = "import sys\n\npackages = [\n    ('torch', 'PyTorch'),\n    ('transformers', 'Transformers'),\n    ('peft', 'PEFT'),\n    ('accelerate', 'Accelerate'),\n    ('openai', 'OpenAI'),\n    ('anthropic', 'Anthropic'),\n    ('google.generativeai', 'Google GenAI'),\n    ('hf_transfer', 'HF Transfer')\n]\n\nprint('\\nPackage Verification:')\nprint('-' * 60)\nall_ok = True\nfor module_name, display_name in packages:\n    try:\n        mod = __import__(module_name)\n        ver = getattr(mod, '__version__', 'OK')\n        print(f'OK {display_name:20s} {ver}')\n    except Exception as e:\n        print(f'FAIL {display_name:20s} {e}')\n        all_ok = False\n\ntry:\n    import bitsandbytes\n    print(f'OK {\"BitsAndBytes\":20s} (optional)')\nexcept:\n    print(f'WARN {\"BitsAndBytes\":20s} not installed (optional)')\n\ntry:\n    import flash_attn\n    print(f'OK {\"Flash Attention 2\":20s} {flash_attn.__version__} (可选加速)')\nexcept:\n    print(f'WARN {\"Flash Attention 2\":20s} not installed (optional)')\n\ntry:\n    import torch\n    print(f'\\nCUDA Available: {torch.cuda.is_available()}')\n    if torch.cuda.is_available():\n        print(f'GPU: {torch.cuda.get_device_name(0)}')\n        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')\nexcept:\n    pass\n\ntry:\n    import transformers\n    from packaging import version\n    if version.parse(transformers.__version__) >= version.parse('4.43.0'):\n        print(f'\\nTransformers version supports Llama 3.1')\n    else:\n        print(f'\\nTransformers version may not support Llama 3.1')\nexcept:\n    pass\n\nsys.exit(0 if all_ok else 1)\n"

# 保存到临时文件并执行
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(verify_code)
    temp_file = f.name

run(f"{sys.executable} {temp_file}")

# 清理临时文件
try:
    os.remove(temp_file)
except:
    pass

# 总结
print("\n" + "="*80)
print("安装完成")
print("="*80)

print("\n下一步操作:")
print("\n1. 立即重启 Jupyter Kernel")
print("   菜单: Kernel -> Restart Kernel")
print("   重启后才能使用新安装的包")
print("\n2. 重启后，重新运行这个安装脚本")
print("   这样 API Keys 会重新加载到环境变量中")
print("\n3. 然后就可以运行训练了:")
print("   %run src/grpo/trainer.py")
print("\n注意:")
print("- 已安装 Transformers 4.44.2，确保与 peft 0.9.0 兼容")
print("- 如果 PyTorch 显示 2.8.0，这是正常的最新版本")
print("- 训练脚本配置为使用 Llama 3 Instruct")
print("- 多 LLM Judge 集成：Claude 3.5 Haiku + Gemini 2.5 Flash + GPT-4o-mini")
print("  (需要 OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY)")
print("- 数据集会自动从 GitHub 下载")
print("\n⚠️ 重要：kernel 重启后，需要重新运行这个脚本来加载 API Keys")

print("\n" + "="*80)
