#!/usr/bin/env python
"""
路径查找脚本 - 找到grpo-dual的实际位置
"""
from pathlib import Path

print("="*80)
print("🔍 查找grpo-dual目录")
print("="*80)

# 所有可能的路径
possible_paths = [
    Path('/workspace/data/halueval/grpo-dual/grpo-dual'),  # 最可能的路径
    Path('/workspace/data/halueval/grpo-dual'),
    Path('/workspace/grpo-dual/grpo-dual'),
    Path('/workspace/grpo-dual'),
    Path('/home/user/grpo-dual/grpo-dual'),
    Path('/home/user/grpo-dual'),
    Path.cwd(),
    Path.cwd() / 'grpo-dual',
    Path.cwd().parent,
    Path.cwd().parent / 'grpo-dual',
]

print("\n搜索结果：\n")
found = []

for p in possible_paths:
    exists = p.exists()
    has_src = (p / 'src').exists() if exists else False
    has_trainer = (p / 'src' / 'grpo' / 'trainer.py').exists() if exists else False

    status = "✓" if has_trainer else "✗"
    print(f"{status} {p}")
    print(f"   目录存在: {exists}, src/存在: {has_src}, trainer.py存在: {has_trainer}")

    if has_trainer:
        found.append(p)
        print(f"   ✅ 这是正确路径！")
    print()

print("="*80)
if found:
    print(f"✓ 找到 {len(found)} 个有效路径：")
    for p in found:
        print(f"  {p}")
    print(f"\n推荐使用: {found[0]}")
    print(f"\n在test_temp_manual.py中修改第16行为：")
    print(f"  GRPO_DUAL_DIR = Path('{found[0]}')")
else:
    print("❌ 未找到有效路径")
print("="*80)
