#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断脚本：查找 llm_judge_prompts_v2.py 的位置并测试导入
"""

import sys
from pathlib import Path

print("=" * 80)
print("诊断：查找 llm_judge_prompts_v2.py")
print("=" * 80)

# 1. 显示当前工作目录
cwd = Path.cwd()
print(f"\n当前工作目录: {cwd}")

# 2. 显示当前 sys.path
print(f"\n当前 sys.path:")
for i, p in enumerate(sys.path):
    print(f"  [{i}] {p}")

# 3. 搜索 llm_judge_prompts_v2.py
print(f"\n搜索 llm_judge_prompts_v2.py...")

search_paths = [
    cwd,                          # workspace/
    cwd / "judges",               # workspace/judges/
    cwd / "src" / "judges",       # workspace/src/judges/
    cwd / "llm_judge_prompts_v2", # workspace/llm_judge_prompts_v2/
    cwd.parent,                   # 父目录
    cwd.parent / "judges",        # 父目录/judges/
]

found_files = []

for search_path in search_paths:
    target_file = search_path / "llm_judge_prompts_v2.py"
    if target_file.exists():
        found_files.append(target_file)
        print(f"  ✅ 找到: {target_file}")
        print(f"     - 是文件: {target_file.is_file()}")
        print(f"     - 大小: {target_file.stat().st_size} bytes")
    else:
        print(f"  ❌ 不存在: {target_file}")

# 4. 使用 glob 递归搜索
print(f"\n使用 glob 递归搜索 (最多搜索3层)...")
for pattern in ["**/llm_judge_prompts_v2.py", "**/*llm_judge*.py"]:
    matches = list(cwd.glob(pattern))
    if matches:
        for match in matches[:5]:  # 最多显示5个结果
            print(f"  ✅ glob找到: {match}")
            found_files.append(match)

# 5. 去重找到的文件
found_files = list(set(found_files))

if not found_files:
    print("\n❌ 没有找到 llm_judge_prompts_v2.py！")
    print("\n请确认文件已上传到 workspace。")
    print("可以运行以下命令查看 workspace 内容:")
    print("  !ls -la /workspace/")
else:
    print(f"\n总共找到 {len(found_files)} 个文件:")
    for i, f in enumerate(found_files):
        print(f"  [{i}] {f}")
        print(f"      相对路径: {f.relative_to(cwd) if f.is_relative_to(cwd) else 'N/A'}")

    # 6. 测试导入
    print("\n" + "=" * 80)
    print("测试导入")
    print("=" * 80)

    for i, file_path in enumerate(found_files):
        print(f"\n尝试从 {file_path.parent} 导入...")

        # 临时添加到 sys.path
        parent_dir = str(file_path.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            print(f"  添加到 sys.path: {parent_dir}")

        # 尝试导入
        try:
            # 清除之前的导入缓存
            if 'llm_judge_prompts_v2' in sys.modules:
                del sys.modules['llm_judge_prompts_v2']

            import llm_judge_prompts_v2
            print(f"  ✅ 导入成功!")
            print(f"  模块位置: {llm_judge_prompts_v2.__file__}")

            # 检查函数是否存在
            if hasattr(llm_judge_prompts_v2, 'get_adaptive_bbq_prompt'):
                print(f"  ✅ get_adaptive_bbq_prompt 存在")
            else:
                print(f"  ❌ get_adaptive_bbq_prompt 不存在")

            if hasattr(llm_judge_prompts_v2, 'get_adaptive_halueval_prompt'):
                print(f"  ✅ get_adaptive_halueval_prompt 存在")
            else:
                print(f"  ❌ get_adaptive_halueval_prompt 不存在")

            break  # 成功导入后退出

        except Exception as e:
            print(f"  ❌ 导入失败: {e}")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)

# 7. 推荐的修复方案
if found_files:
    best_path = found_files[0].parent
    print(f"\n推荐操作：")
    print(f"在 notebook 中运行以下代码：")
    print(f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('{best_path}')))

# 然后运行 trainer
import trainer
trainer.main()
""")
