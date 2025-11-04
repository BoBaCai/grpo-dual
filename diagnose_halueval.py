#!/usr/bin/env python3
"""诊断HaluEval文件格式"""

from pathlib import Path

HALUEVAL_DIR = Path("/workspace/data/halueval")

files = ["qa_data.json", "dialogue_data.json", "summarization_data.json", "general_data.json"]

for filename in files:
    fp = HALUEVAL_DIR / filename
    if not fp.exists():
        print(f"❌ {filename} 不存在")
        continue

    print(f"\n{'='*80}")
    print(f"{filename}")
    print(f"{'='*80}")

    # 读取前5行查看格式
    with open(fp, 'r') as f:
        for i in range(5):
            line = f.readline()
            if not line:
                break
            print(f"Line {i+1} (前100字符): {line[:100]}")

    print("\n文件大小:", fp.stat().st_size / 1024 / 1024, "MB")

    # 统计行数
    with open(fp, 'r') as f:
        line_count = sum(1 for _ in f)
    print(f"总行数: {line_count}")
