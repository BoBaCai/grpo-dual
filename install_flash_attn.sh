#!/bin/bash
#
# Flash Attention 2 安装脚本 (A100 优化)
#
# 使用方法:
#   bash install_flash_attn.sh
#
# 或者如果需要指定 CUDA 版本:
#   CUDA_VERSION=12.1 bash install_flash_attn.sh
#

set -e  # 遇到错误立即退出

echo "========================================"
echo "Flash Attention 2 安装脚本 (A100)"
echo "========================================"

# 1. 检查 CUDA
echo -e "\n[1/5] 检查 CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
    echo "✅ NVIDIA GPU 检测成功"
else
    echo "❌ 未检测到 nvidia-smi，请确认："
    echo "   1. 是否在 GPU 节点上运行"
    echo "   2. NVIDIA 驱动是否正确安装"
    exit 1
fi

if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "✅ NVCC 版本: $NVCC_VERSION"
else
    echo "⚠️ 未检测到 nvcc，Flash Attention 可能无法从源码编译"
    echo "   尝试使用预编译包..."
fi

# 2. 检查 Python 环境
echo -e "\n[2/5] 检查 Python 环境..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python 版本: $PYTHON_VERSION"

if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'CPU')")
    echo "✅ PyTorch 版本: $TORCH_VERSION"
    echo "✅ PyTorch CUDA 版本: $CUDA_VERSION"
else
    echo "❌ PyTorch 未安装，请先安装 PyTorch"
    exit 1
fi

# 3. 安装依赖
echo -e "\n[3/5] 安装编译依赖..."
pip install packaging ninja -q
echo "✅ 依赖安装完成"

# 4. 尝试安装 Flash Attention 2
echo -e "\n[4/5] 安装 Flash Attention 2..."
echo "⏳ 正在编译（预计需要 5-15 分钟）..."
echo ""

# A100 对应 compute capability 8.0
export TORCH_CUDA_ARCH_LIST="8.0"

# 尝试安装最新稳定版
if pip install flash-attn --no-build-isolation; then
    echo "✅ Flash Attention 2 安装成功！"
else
    echo "❌ 安装失败，尝试降级到稳定版本..."
    if pip install flash-attn==2.5.8 --no-build-isolation; then
        echo "✅ Flash Attention 2 (v2.5.8) 安装成功！"
    else
        echo "❌ 安装失败，可能的原因："
        echo "   1. CUDA 版本不兼容（需要 CUDA 11.6+）"
        echo "   2. 编译工具链缺失"
        echo "   3. 显存不足"
        echo ""
        echo "💡 解决方案："
        echo "   - 检查 CUDA 版本: nvcc --version"
        echo "   - 更新 PyTorch 到支持的版本"
        echo "   - 查看详细错误日志重新安装"
        exit 1
    fi
fi

# 5. 验证安装
echo -e "\n[5/5] 验证安装..."
if python -c "import flash_attn; print(f'✅ flash-attn 版本: {flash_attn.__version__}')" 2>/dev/null; then
    echo ""
    echo "========================================"
    echo "🎉 Flash Attention 2 安装成功！"
    echo "========================================"
    echo ""
    echo "下次运行训练时，你应该看到："
    echo '  ✅ Flash Attention 2 可用，已启用'
    echo ""
    echo "预期加速效果："
    echo "  - 生成阶段: 1.5-2x 加速"
    echo "  - 显存占用: 减少 20-30%"
else
    echo "❌ 验证失败，请检查安装日志"
    exit 1
fi
