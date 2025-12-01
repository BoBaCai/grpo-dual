import os
# 注意：请在 Jupyter notebook 中设置这些环境变量，不要硬编码在代码中
# os.environ["OPENAI_API_KEY"] = "your-key"
# os.environ["HF_TOKEN"] = "your-hf-token"
# os.environ["ANTHROPIC_API_KEY"] = "your-key"
import subprocess
import sys

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
print("- 所有必要依赖")
print("\n预计时间: 5-10分钟")
print("="*80)

# ============================================================================
# 步骤 1: 完全卸载
# ============================================================================
run(
    f"{sys.executable} -m pip uninstall -y "
    f"torch torchvision torchaudio "
    f"transformers peft accelerate bitsandbytes datasets",
    "步骤 1/6: 卸载现有包"
)

# ============================================================================
# 步骤 2: 清理
# ============================================================================
run(f"{sys.executable} -m pip cache purge", "步骤 2/6: 清理缓存")

# ============================================================================
# 步骤 3: 安装 PyTorch (使用最新稳定版)
# ============================================================================
run(
    f"{sys.executable} -m pip install "
    f"torch torchvision torchaudio "
    f"--index-url https://download.pytorch.org/whl/cu121",
    "步骤 3/6: 安装 PyTorch (CUDA 12.1)"
)

# ============================================================================
# 步骤 4: 安装 Transformers (固定版本，确保与 peft 0.9.0 兼容)
# ============================================================================
run(
    f"{sys.executable} -m pip install 'transformers==4.44.2'",
    "步骤 4/6: 安装 Transformers (4.44.2，兼容 peft 0.9.0)"
)

# ============================================================================
# 步骤 5: 安装其他核心包
# ============================================================================
run(
    f"{sys.executable} -m pip install "
    f"peft==0.9.0 "
    f"accelerate==0.27.0 "
    f"'bitsandbytes>=0.43.0' "
    f"datasets "
    f"google-generativeai "
    f"hf_transfer "
    f"sentencepiece "
    f"protobuf "
    f"'requests>=2.32.2' "
    f"scipy "
    f"tqdm",
    "步骤 5/6: 安装其他依赖"
)

# ============================================================================
# 步骤 6: 验证
# ============================================================================
print("\n" + "="*80)
print("步骤 6/6: 验证安装")
print("="*80)

# 将验证脚本保存到临时文件
verify_code = """
import sys

packages = [
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('peft', 'PEFT'),
    ('accelerate', 'Accelerate'),
    ('google.generativeai', 'Google GenAI'),
    ('hf_transfer', 'HF Transfer')
]

print("\\nPackage Verification:")
print("-" * 60)
all_ok = True
for module_name, display_name in packages:
    try:
        mod = __import__(module_name)
        ver = getattr(mod, '__version__', 'OK')
        print(f"OK {display_name:20s} {ver}")
    except Exception as e:
        print(f"FAIL {display_name:20s} {e}")
        all_ok = False

try:
    import bitsandbytes
    print(f"OK {'BitsAndBytes':20s} (optional)")
except:
    print(f"WARN {'BitsAndBytes':20s} not installed (optional)")

try:
    import torch
    print(f"\\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except:
    pass

try:
    import transformers
    from packaging import version
    if version.parse(transformers.__version__) >= version.parse("4.43.0"):
        print(f"\\nTransformers version supports Llama 3.1")
    else:
        print(f"\\nTransformers version may not support Llama 3.1")
except:
    pass

sys.exit(0 if all_ok else 1)
"""

# 保存到临时文件并执行
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(verify_code)
    temp_file = f.name

run(f"{sys.executable} {temp_file}")

# 清理临时文件
import os
try:
    os.remove(temp_file)
except:
    pass

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("安装完成")
print("="*80)

print("""
重要步骤:

1. 立即重启 Jupyter Kernel
   菜单: Kernel -> Restart Kernel

2. 重启后验证:
   import torch
   import transformers
   print(f"PyTorch: {torch.__version__}")
   print(f"Transformers: {transformers.__version__}")
   print(f"CUDA: {torch.cuda.is_available()}")

3. 设置环境变量:
   import os
   os.environ["GEMINI_API_KEY"] = "your-key"
   os.environ["HF_TOKEN"] = "your-token"

4. 运行训练:
   !python multi_objective_lora_grpo_llama.py

注意:
- 已安装 Transformers 4.44.2，确保与 peft 0.9.0 兼容
- 如果 PyTorch 显示 2.8.0，这是正常的最新版本
- 训练脚本配置为使用 Llama 3 Instruct
- 版本已固定以避免依赖冲突
""")

print("="*80)
