#!/usr/bin/env python
"""
GPU 显存监控脚本

使用方法:
    python check_gpu_memory.py

功能:
    - 显示 GPU 基本信息
    - 显示当前显存使用情况
    - 显示 PyTorch 缓存使用情况
    - 提供优化建议
"""

import torch

def format_bytes(bytes_val):
    """将字节转换为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"

def check_gpu_memory():
    """检查 GPU 显存使用情况"""
    print("=" * 70)
    print("GPU 显存监控")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，请检查：")
        print("   1. GPU 驱动是否正确安装")
        print("   2. PyTorch 是否安装了 CUDA 版本")
        print("   3. 是否在 GPU 节点上运行")
        return

    # GPU 基本信息
    num_gpus = torch.cuda.device_count()
    print(f"\n📊 GPU 基本信息")
    print(f"   可用 GPU 数量: {num_gpus}")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"\n   GPU {i}: {props.name}")
        print(f"   计算能力: {props.major}.{props.minor}")
        print(f"   总显存: {format_bytes(props.total_memory)}")

        # 当前显存使用
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        free = props.total_memory - reserved

        print(f"\n   💾 显存使用情况:")
        print(f"   已分配: {format_bytes(allocated)} ({allocated/props.total_memory*100:.1f}%)")
        print(f"   已预留: {format_bytes(reserved)} ({reserved/props.total_memory*100:.1f}%)")
        print(f"   可用: {format_bytes(free)} ({free/props.total_memory*100:.1f}%)")

        # PyTorch 缓存统计
        if hasattr(torch.cuda, 'memory_stats'):
            stats = torch.cuda.memory_stats(i)
            num_alloc_retries = stats.get('num_alloc_retries', 0)
            num_ooms = stats.get('num_ooms', 0)

            print(f"\n   📈 缓存统计:")
            print(f"   分配重试次数: {num_alloc_retries}")
            print(f"   OOM 次数: {num_ooms}")

            if num_ooms > 0:
                print(f"\n   ⚠️ 检测到 {num_ooms} 次 OOM，建议优化！")

    # 优化建议
    print(f"\n" + "=" * 70)
    print("💡 显存优化建议")
    print("=" * 70)

    total_mem_gb = props.total_memory / 1024**3
    used_percent = reserved / props.total_memory * 100

    if used_percent > 90:
        print("\n⚠️ 显存使用率 >90%，强烈建议优化：")
        print("   1. 减小 GRPO_BATCH_SIZE (当前推荐: 1-2)")
        print("   2. 减小 MAX_NEW_TOKENS (当前推荐: 64-96)")
        print("   3. 减小 LORA_R (当前推荐: 4-8)")
        print("   4. 启用梯度累积 (GRADIENT_ACCUMULATION_STEPS=2-4)")
    elif used_percent > 75:
        print("\n⚠️ 显存使用率 >75%，建议优化：")
        print("   1. 减小 MAX_NEW_TOKENS (当前推荐: 96)")
        print("   2. 启用梯度累积 (GRADIENT_ACCUMULATION_STEPS=2)")
        print("   3. 减小 LORA_R 到 8")
    elif used_percent < 50:
        print("\n✅ 显存使用率良好，可以考虑：")
        print("   1. 增大 GRPO_BATCH_SIZE 以加速训练")
        print("   2. 增大 K_ROLLOUTS 以提高采样多样性")
        print("   3. 增大 MAX_NEW_TOKENS 以支持更长回答")
    else:
        print("\n✅ 显存使用率正常")

    # 显存清理提示
    print(f"\n🔧 如需清理显存，运行：")
    print("   torch.cuda.empty_cache()")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_gpu_memory()
