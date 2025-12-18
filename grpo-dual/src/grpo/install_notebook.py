# 适用于 Jupyter Notebook Cell 的安装脚本
# 可以直接复制粘贴到 notebook cell 中运行

import os
import subprocess
import sys
import tempfile

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
print("- 所有必要依赖")
print("\n预计时间: 5-10分钟")
print("="*80)

# 步骤 1: 完全卸载
run(
    f"{sys.executable} -m pip uninstall -y "
    f"torch torchvision torchaudio "
    f"transformers peft accelerate bitsandbytes datasets",
    "步骤 1/6: 卸载现有包"
)

# 步骤 2: 清理
run(f"{sys.executable} -m pip cache purge", "步骤 2/6: 清理缓存")

# 步骤 3: 安装 PyTorch (使用最新稳定版)
run(
    f"{sys.executable} -m pip install "
    f"torch torchvision torchaudio "
    f"--index-url https://download.pytorch.org/whl/cu121",
    "步骤 3/6: 安装 PyTorch (CUDA 12.1)"
)

# 步骤 4: 安装 Transformers (固定版本，确保与 peft 0.9.0 兼容)
run(
    f"{sys.executable} -m pip install 'transformers==4.44.2'",
    "步骤 4/6: 安装 Transformers (4.44.2，兼容 peft 0.9.0)"
)

# 步骤 5: 安装其他核心包
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
    f"'jinja2>=3.1.0' "
    f"scipy "
    f"tqdm",
    "步骤 5/6: 安装其他依赖（含 jinja2>=3.1.0 支持 chat template）"
)

# 步骤 6: 验证
print("\n" + "="*80)
print("步骤 6/6: 验证安装")
print("="*80)

# 验证代码
verify_code = "import sys\n\npackages = [\n    ('torch', 'PyTorch'),\n    ('transformers', 'Transformers'),\n    ('peft', 'PEFT'),\n    ('accelerate', 'Accelerate'),\n    ('google.generativeai', 'Google GenAI'),\n    ('hf_transfer', 'HF Transfer')\n]\n\nprint('\\nPackage Verification:')\nprint('-' * 60)\nall_ok = True\nfor module_name, display_name in packages:\n    try:\n        mod = __import__(module_name)\n        ver = getattr(mod, '__version__', 'OK')\n        print(f'OK {display_name:20s} {ver}')\n    except Exception as e:\n        print(f'FAIL {display_name:20s} {e}')\n        all_ok = False\n\ntry:\n    import bitsandbytes\n    print(f'OK {\"BitsAndBytes\":20s} (optional)')\nexcept:\n    print(f'WARN {\"BitsAndBytes\":20s} not installed (optional)')\n\ntry:\n    import torch\n    print(f'\\nCUDA Available: {torch.cuda.is_available()}')\n    if torch.cuda.is_available():\n        print(f'GPU: {torch.cuda.get_device_name(0)}')\n        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')\nexcept:\n    pass\n\ntry:\n    import transformers\n    from packaging import version\n    if version.parse(transformers.__version__) >= version.parse('4.43.0'):\n        print(f'\\nTransformers version supports Llama 3.1')\n    else:\n        print(f'\\nTransformers version may not support Llama 3.1')\nexcept:\n    pass\n\nsys.exit(0 if all_ok else 1)\n"

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

print("\n重要步骤:")
print("\n1. 立即重启 Jupyter Kernel")
print("   菜单: Kernel -> Restart Kernel")
print("\n2. 重启后验证:")
print("   import torch")
print("   import transformers")
print("   print(f'PyTorch: {torch.__version__}')")
print("   print(f'Transformers: {transformers.__version__}')")
print("   print(f'CUDA: {torch.cuda.is_available()}')")
print("\n3. 设置环境变量:")
print("   import os")
print("   os.environ['GEMINI_API_KEY'] = 'your-key'")
print("   os.environ['HF_TOKEN'] = 'your-token'")
print("\n4. 运行训练:")
print("   !python multi_objective_lora_grpo_llama.py")
print("\n注意:")
print("- 已安装 Transformers 4.44.2，确保与 peft 0.9.0 兼容")
print("- 如果 PyTorch 显示 2.8.0，这是正常的最新版本")
print("- 训练脚本配置为使用 Llama 3 Instruct")
print("- 版本已固定以避免依赖冲突")

print("\n" + "="*80)
