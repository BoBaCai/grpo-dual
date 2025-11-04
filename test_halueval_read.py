#!/usr/bin/env python3
"""测试HaluEval读取和显示完整样本"""

import json
from pathlib import Path

HALUEVAL_DIR = Path("/workspace/data/halueval")

files = {
    "qa": "qa_data.json",
    "dialogue": "dialogue_data.json",
    "summarization": "summarization_data.json",
    "general": "general_data.json"
}

for subset, filename in files.items():
    fp = HALUEVAL_DIR / filename
    if not fp.exists():
        print(f"❌ {filename} 不存在\n")
        continue

    print(f"\n{'='*80}")
    print(f"{subset.upper()} 子集")
    print(f"{'='*80}")

    # 读取第一个样本
    with open(fp, 'r') as f:
        first_line = f.readline()
        try:
            sample = json.loads(first_line)

            # 显示字段名
            print(f"\n字段列表: {list(sample.keys())}")

            # 显示完整样本（截断长字段）
            print(f"\n完整样本：")
            for key, value in sample.items():
                if isinstance(value, str):
                    if len(value) > 200:
                        print(f"  {key}: {value[:200]}... (共{len(value)}字符)")
                    else:
                        print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")

        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            print(f"原始内容（前500字符）: {first_line[:500]}")

    # 统计总样本数
    with open(fp, 'r') as f:
        count = sum(1 for _ in f)
    print(f"\n总样本数: {count}")

print("\n" + "="*80)
print("读取完成！")
