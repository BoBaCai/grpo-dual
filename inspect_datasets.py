#!/usr/bin/env python3
"""
数据集检查脚本 - 可以直接在Jupyter notebook运行
复制整个脚本到notebook cell执行即可
"""

import json
from pathlib import Path
from collections import defaultdict

# ============================================================================
# 配置路径
# ============================================================================
BBQ_DIR = Path("/workspace/data/bbq")
HALUEVAL_DIR = Path("/workspace/data/halueval")

# 如果/workspace不存在，尝试其他路径
if not BBQ_DIR.exists():
    BBQ_DIR = Path("data/bbq")
if not HALUEVAL_DIR.exists():
    HALUEVAL_DIR = Path("data/halueval")

print("="*80)
print("数据集路径检查")
print("="*80)
print(f"BBQ: {BBQ_DIR} {'✅ 存在' if BBQ_DIR.exists() else '❌ 不存在'}")
print(f"HaluEval: {HALUEVAL_DIR} {'✅ 存在' if HALUEVAL_DIR.exists() else '❌ 不存在'}")
print()

# ============================================================================
# Part 1: BBQ数据集检查
# ============================================================================
print("\n" + "="*80)
print("Part 1: BBQ数据集统计")
print("="*80)

BBQ_FILES = [
    "Age.jsonl",
    "Disability_status.jsonl",
    "Gender_identity.jsonl",
    "Nationality.jsonl",
    "Physical_appearance.jsonl",
    "Race_ethnicity.jsonl",
    "Race_x_gender.jsonl",
    "Race_x_SES.jsonl",
    "Religion.jsonl",
    "SES.jsonl",
    "Sexual_orientation.jsonl",
]

if BBQ_DIR.exists():
    bbq_stats = []

    for filename in BBQ_FILES:
        fp = BBQ_DIR / filename
        if not fp.exists():
            print(f"⚠️  {filename}: 文件不存在")
            continue

        ambig = 0
        disambig = 0
        context_lengths = []

        with open(fp, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item.get("context_condition") == "ambig":
                        ambig += 1
                    else:
                        disambig += 1

                    # 收集context长度
                    context = item.get("context", "")
                    context_lengths.append(len(context))
                except:
                    continue

        total = ambig + disambig
        if total > 0:
            pct_disambig = disambig / total * 100
            avg_context_len = sum(context_lengths) / len(context_lengths) if context_lengths else 0

            stat = {
                "file": filename.replace(".jsonl", ""),
                "total": total,
                "ambig": ambig,
                "disambig": disambig,
                "pct_disambig": pct_disambig,
                "avg_context_len": avg_context_len
            }
            bbq_stats.append(stat)

            print(f"\n{filename.replace('.jsonl', '')}:")
            print(f"  Total: {total}")
            print(f"  Disambig: {disambig} ({pct_disambig:.1f}%)")
            print(f"  Ambig: {ambig} ({100-pct_disambig:.1f}%)")
            print(f"  Avg context length: {avg_context_len:.0f} chars")

            # 检查是否满足70% disambig目标
            if pct_disambig < 70:
                print(f"  ⚠️  Disambig比例<70%，自适应采样会调整为60%或更低")

    # 总计
    if bbq_stats:
        total_all = sum(s["total"] for s in bbq_stats)
        total_disambig = sum(s["disambig"] for s in bbq_stats)
        print(f"\n{'='*80}")
        print(f"BBQ总计: {total_all} 样本")
        print(f"  Disambig: {total_disambig} ({total_disambig/total_all*100:.1f}%)")
        print(f"  Ambig: {total_all-total_disambig} ({(total_all-total_disambig)/total_all*100:.1f}%)")
else:
    print("❌ BBQ数据集路径不存在，跳过")

# ============================================================================
# Part 2: BBQ样本示例
# ============================================================================
print("\n" + "="*80)
print("Part 2: BBQ样本示例（Age子集的第一个样本）")
print("="*80)

if BBQ_DIR.exists():
    age_file = BBQ_DIR / "Age.jsonl"
    if age_file.exists():
        with open(age_file, 'r') as f:
            first_line = f.readline()
            sample = json.loads(first_line)

            print("\n完整样本内容：")
            print(json.dumps(sample, indent=2, ensure_ascii=False))

            print("\n\n关键字段提取：")
            print(f"Context: {sample.get('context', '')[:200]}...")
            print(f"Question: {sample.get('question', '')}")
            print(f"Options:")
            print(f"  A) {sample.get('ans0', '')}")
            print(f"  B) {sample.get('ans1', '')}")
            print(f"  C) {sample.get('ans2', '')}")
            print(f"Context condition: {sample.get('context_condition', '')}")
            print(f"Label: {sample.get('label', '')}")
            print(f"Answer info: {sample.get('answer_info', {})}")
    else:
        print("❌ Age.jsonl不存在")
else:
    print("❌ BBQ数据集路径不存在，跳过")

# ============================================================================
# Part 3: HaluEval数据集检查
# ============================================================================
print("\n" + "="*80)
print("Part 3: HaluEval数据集统计")
print("="*80)

HALUEVAL_FILES = {
    "qa": "qa_data.json",
    "dialogue": "dialogue_data.json",
    "summarization": "summarization_data.json",
    "general": "general_data.json"
}

if HALUEVAL_DIR.exists():
    for subset, filename in HALUEVAL_FILES.items():
        fp = HALUEVAL_DIR / filename
        if not fp.exists():
            print(f"⚠️  {filename}: 文件不存在")
            continue

        try:
            # HaluEval是JSONL格式（每行一个JSON对象）
            # 只读前100行用于统计，避免大文件读取过慢
            data = []
            with open(fp, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 100:  # 只读前100行
                        break
                    if line.strip():
                        data.append(json.loads(line))

            # 统计总行数
            with open(fp, 'r') as f:
                total = sum(1 for _ in f)
            file_size_mb = fp.stat().st_size / 1024 / 1024

            print(f"\n{subset}:")
            print(f"  Total samples: {total}")
            print(f"  File size: {file_size_mb:.1f} MB")

            # 检查knowledge/document长度
            if total > 0:
                if subset == "qa" and "knowledge" in data[0]:
                    know_lens = [len(str(item.get("knowledge", ""))) for item in data[:100]]
                    avg_len = sum(know_lens) / len(know_lens)
                    print(f"  Avg knowledge length: {avg_len:.0f} chars")

                elif subset == "summarization" and "document" in data[0]:
                    doc_lens = [len(str(item.get("document", ""))) for item in data[:100]]
                    avg_len = sum(doc_lens) / len(doc_lens)
                    max_len = max(doc_lens)
                    print(f"  Avg document length: {avg_len:.0f} chars")
                    print(f"  Max document length (first 100): {max_len} chars")
        except Exception as e:
            print(f"❌ 读取{filename}失败: {e}")
else:
    print("❌ HaluEval数据集路径不存在，跳过")

# ============================================================================
# Part 4: HaluEval样本示例（最关键！）
# ============================================================================
print("\n" + "="*80)
print("Part 4: HaluEval样本示例（每个子集的第一个样本）")
print("="*80)

if HALUEVAL_DIR.exists():
    for subset, filename in HALUEVAL_FILES.items():
        fp = HALUEVAL_DIR / filename
        if not fp.exists():
            continue

        try:
            # HaluEval是JSONL格式（每行一个JSON对象）
            # 只读第一行获取样本
            with open(fp, 'r') as f:
                first_line = f.readline()
                sample = json.loads(first_line) if first_line.strip() else {}

            if sample:
                print(f"\n{'='*80}")
                print(f"{subset.upper()} 子集第一个样本：")
                print(f"{'='*80}")

                # 打印所有字段名
                print(f"\n可用字段: {list(sample.keys())}")

                # 打印完整样本（截断长字段）
                print(f"\n完整样本内容：")
                display_sample = {}
                for key, value in sample.items():
                    if isinstance(value, str) and len(value) > 200:
                        display_sample[key] = value[:200] + "... (truncated)"
                    else:
                        display_sample[key] = value

                print(json.dumps(display_sample, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"❌ 读取{filename}失败: {e}")
else:
    print("❌ HaluEval数据集路径不存在，跳过")

# ============================================================================
# Part 5: 关键问题总结
# ============================================================================
print("\n" + "="*80)
print("Part 5: 关键发现总结")
print("="*80)

print("""
请检查以上输出，重点关注：

1. BBQ数据集：
   - 每个子集的disambig比例是否都>=60%？
   - Context平均长度？50字符的snippet是否足够？
   - answer_info的格式如何？

2. HaluEval数据集：
   - General子集是否有knowledge字段？
   - right_answer的格式（短答案 vs 完整句子）？
   - Summarization的document是否需要截断？

3. 采样策略验证：
   - 当前N_BBQ_TRAIN=1100（每个子集100），是否足够？
   - 当前N_HALU_TRAIN=400（每个子集100），是否合理？
""")

print("\n脚本执行完成！✅")
