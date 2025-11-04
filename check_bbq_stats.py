#!/usr/bin/env python3
"""检查BBQ数据集的实际统计信息"""

import json
from pathlib import Path
from collections import defaultdict

BBQ_DIR = Path("/workspace/data/bbq")

# 如果/workspace不存在，尝试本地路径
if not BBQ_DIR.exists():
    BBQ_DIR = Path("data/bbq")
    if not BBQ_DIR.exists():
        print("❌ BBQ数据集路径不存在")
        exit(1)

BBQ_FILES = {
    "Age": "Age.jsonl",
    "Disability_status": "Disability_status.jsonl",
    "Gender_identity": "Gender_identity.jsonl",
    "Nationality": "Nationality.jsonl",
    "Physical_appearance": "Physical_appearance.jsonl",
    "Race_ethnicity": "Race_ethnicity.jsonl",
    "Race_x_gender": "Race_x_gender.jsonl",
    "Race_x_SES": "Race_x_SES.jsonl",
    "Religion": "Religion.jsonl",
    "SES": "SES.jsonl",
    "Sexual_orientation": "Sexual_orientation.jsonl",
}

print("="*80)
print("BBQ数据集统计信息")
print("="*80)

total_ambig = 0
total_disambig = 0
category_stats = []

for cat, filename in BBQ_FILES.items():
    fp = BBQ_DIR / filename
    if not fp.exists():
        print(f"⚠️  {cat}: 文件不存在 - {filename}")
        continue

    ambig = 0
    disambig = 0

    with open(fp, 'r') as f:
        for line in f:
            try:
                it = json.loads(line)
                if it.get("context_condition") == "ambig":
                    ambig += 1
                else:
                    disambig += 1
            except:
                continue

    total = ambig + disambig
    if total > 0:
        pct_ambig = ambig / total * 100
        pct_disambig = disambig / total * 100

        category_stats.append({
            "category": cat,
            "ambig": ambig,
            "disambig": disambig,
            "total": total,
            "pct_ambig": pct_ambig
        })

        print(f"\n{cat}:")
        print(f"  Total: {total}")
        print(f"  Ambiguous: {ambig} ({pct_ambig:.1f}%)")
        print(f"  Disambiguated: {disambig} ({pct_disambig:.1f}%)")

        total_ambig += ambig
        total_disambig += disambig

print("\n" + "="*80)
print("总计:")
print(f"  Total: {total_ambig + total_disambig}")
print(f"  Ambiguous: {total_ambig} ({total_ambig/(total_ambig+total_disambig)*100:.1f}%)")
print(f"  Disambiguated: {total_disambig} ({total_disambig/(total_ambig+total_disambig)*100:.1f}%)")
print("="*80)

# 检查是否有子集ambig比例过高或过低
print("\n⚠️  潜在问题:")
for stat in category_stats:
    if stat["pct_ambig"] < 30:
        print(f"  {stat['category']}: ambig比例过低 ({stat['pct_ambig']:.1f}%) - 强制20%可能会重复采样")
    elif stat["pct_ambig"] > 70:
        print(f"  {stat['category']}: ambig比例过高 ({stat['pct_ambig']:.1f}%) - 强制80% disambig可能不够")

# 模拟采样
print("\n" + "="*80)
print("模拟采样 N_BBQ_TRAIN=1100 (每个类别100样本)")
print("="*80)

per_cat = 1100 // 11  # 100
print(f"每个类别目标: {per_cat} 样本\n")

for stat in category_stats:
    cat = stat["category"]
    total = stat["total"]
    ambig = stat["ambig"]
    disambig = stat["disambig"]

    # 强制80/20
    need_disambig = int(per_cat * 0.8)  # 80
    need_ambig = per_cat - need_disambig  # 20

    status = "✅"
    warning = ""

    if disambig < need_disambig:
        status = "❌"
        warning = f"disambig不足 (需要{need_disambig}，只有{disambig})"
    elif ambig < need_ambig:
        status = "⚠️ "
        warning = f"ambig不足 (需要{need_ambig}，只有{ambig})"

    print(f"{status} {cat}:")
    print(f"    需要: {need_disambig} disambig + {need_ambig} ambig")
    print(f"    实际: {disambig} disambig + {ambig} ambig")
    if warning:
        print(f"    {warning}")
