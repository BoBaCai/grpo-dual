#!/usr/bin/env python3
"""
SFT Target 长度检查脚本 - Jupyter notebook友好版
可以直接复制整个脚本到notebook cell执行

诊断目标：检查 SFT 训练数据的 target 长度是否与 MIN_NEW_TOKENS=5 匹配
核心问题：如果 target 平均长度远大于 5 tokens，会导致：
  1. 模型在 SFT 学习生成更长的内容
  2. GRPO 时模型想生成长内容，但被 MIN_NEW_TOKENS=5 约束
  3. EOS Suppressor 强制禁止过早结束
  4. 模型不知道说什么 → 生成"最确定"的token → Entropy崩溃

用法：
  # 方式1：命令行运行
  python scripts/inspect_sft_targets.py

  # 方式2：Jupyter notebook
  %run scripts/inspect_sft_targets.py

  # 方式3：直接复制整个脚本到notebook cell运行
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# ============================================================================
# 配置路径（自动检测）
# ============================================================================
print("="*80)
print("🔍 SFT Target 长度检查脚本")
print("="*80)

# 尝试多个可能的数据路径
BBQ_PATHS = [
    Path("/workspace/data/bbq"),
    Path("data/bbq"),
    Path("../data/bbq"),
    Path("grpo-dual/data/bbq"),
]

HALUEVAL_PATHS = [
    Path("/workspace/data/halueval"),
    Path("data/halueval"),
    Path("../data/halueval"),
    Path("grpo-dual/data/halueval"),
]

BBQ_DIR = None
HALUEVAL_DIR = None

for path in BBQ_PATHS:
    if path.exists():
        BBQ_DIR = path
        break

for path in HALUEVAL_PATHS:
    if path.exists():
        HALUEVAL_DIR = path
        break

print(f"\n数据路径检测:")
print(f"  BBQ: {BBQ_DIR if BBQ_DIR else '❌ 未找到'}")
print(f"  HaluEval: {HALUEVAL_DIR if HALUEVAL_DIR else '❌ 未找到'}")

if not BBQ_DIR or not HALUEVAL_DIR:
    print(f"\n⚠️  数据集路径未找到，请手动设置：")
    print(f"  BBQ_DIR = Path('/your/path/to/bbq')")
    print(f"  HALUEVAL_DIR = Path('/your/path/to/halueval')")
    # 不退出，继续尝试

# ============================================================================
# 配置参数（与 trainer.py 保持一致）
# ============================================================================
MIN_NEW_TOKENS_TRAIN = 5  # 当前配置
MAX_NEW_TOKENS_TRAIN = 128
N_BBQ_TRAIN = 1100
N_HALU_TRAIN = 400
SUMM_MAX_DOC_CHARS = 1000

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

HALUEVAL_FILES = {
    "qa": "qa_data.json",
    "dialogue": "dialogue_data.json",
    "summarization": "summarization_data.json",
    "general": "general_data.json",
}

# ============================================================================
# 辅助函数
# ============================================================================
def tokenize_rough(text: str) -> int:
    """粗略估算token数量（英文约4字符/token）"""
    return max(1, len(text) // 4)

def read_jsonl(fp: Path, max_lines: int = None) -> list:
    """读取JSONL文件"""
    data = []
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        print(f"❌ 读取 {fp.name} 失败: {e}")
    return data

# ============================================================================
# BBQ Target 构建（与 trainer.py 一致）
# ============================================================================
def build_bbq_target(item: dict) -> str:
    """构建BBQ的SFT target"""
    context_condition = item.get("context_condition", "")

    if context_condition == "ambig":
        # Ambiguous: 选Unknown
        unk = find_unknown_option(item)
        return f"Answer: {unk}\nJustification: The context does not provide sufficient information to determine this."
    else:
        # Disambiguous: 选正确答案并引用context
        label = item.get("label", -1)
        if 0 <= label <= 2:
            answer_key = f"ans{label}"
            correct_answer = chr(65 + label)  # A, B, C
            answer_text = item.get(answer_key, "")

            # 提取context片段（120字符）
            context = item.get("context", "")
            context_snippet = context[:120] + "..." if len(context) > 120 else context

            return f"Answer: {correct_answer}\nJustification: Based on the context: \"{context_snippet}\", the answer is {answer_text}."
        else:
            # 没有label，回退到ambiguous
            unk = find_unknown_option(item)
            return f"Answer: {unk}\nJustification: The context does not provide sufficient information to determine this."

def find_unknown_option(item: dict) -> str:
    """查找Unknown选项"""
    info = item.get("answer_info", {})
    for idx, key in enumerate(["ans0", "ans1", "ans2"]):
        if key in info:
            val = info[key]
            if isinstance(val, list) and "unknown" in [str(x).lower() for x in val]:
                return chr(65 + idx)
    return "C"

# ============================================================================
# HaluEval Target 构建（与 trainer.py 一致）
# ============================================================================
def build_halueval_target(item: dict, subset: str) -> str:
    """构建HaluEval的SFT target"""

    if subset == "qa":
        answer = item.get("right_answer", "")
        know = item.get("knowledge", "")
        # QA: 150字符snippet
        know_snippet = know[:150] + "..." if len(know) > 150 else know
        return f"Answer: {answer}\nEvidence: \"{know_snippet}\""

    elif subset == "dialogue":
        response = item.get("right_response", "")
        know = item.get("knowledge", "")
        # Dialogue: 150字符snippet
        know_snippet = know[:150] + "..." if len(know) > 150 else know
        return f"Answer: {response}\nEvidence: \"{know_snippet}\""

    elif subset == "summarization":
        gold = item.get("right_summary", "") or item.get("summary", "")
        doc = item.get("document", "") or item.get("article", "")
        # 截断document
        if len(doc) > SUMM_MAX_DOC_CHARS:
            doc = doc[:SUMM_MAX_DOC_CHARS] + "..."
        # 200字符evidence
        doc_snippet = doc[:200] + "..." if len(doc) > 200 else doc
        return f"Summary: {gold}\nEvidence: \"{doc_snippet}\""

    else:  # general
        chatgpt_resp = item.get("chatgpt_response", "")
        hallucination = item.get("hallucination", "no")

        if hallucination == "no":
            # 无hallucination
            resp_truncated = chatgpt_resp[:200] + "..." if len(chatgpt_resp) > 200 else chatgpt_resp
            return f"Answer: {resp_truncated}\nEvidence: \"Based on general knowledge\""
        else:
            # 有hallucination
            return "Answer: I need more information to provide an accurate answer.\nEvidence: \"insufficient\""

# ============================================================================
# 分析函数
# ============================================================================
def analyze_targets(targets: list, name: str):
    """分析target长度分布"""
    print(f"\n{'='*80}")
    print(f"📊 {name} Target 长度分析")
    print(f"{'='*80}")

    if not targets:
        print("❌ 无样本数据")
        return None

    char_lengths = [len(t) for t in targets]
    token_lengths = [tokenize_rough(t) for t in targets]

    char_mean = sum(char_lengths) / len(char_lengths)
    char_median = sorted(char_lengths)[len(char_lengths)//2]
    char_min, char_max = min(char_lengths), max(char_lengths)

    token_mean = sum(token_lengths) / len(token_lengths)
    token_median = sorted(token_lengths)[len(token_lengths)//2]
    token_min, token_max = min(token_lengths), max(token_lengths)

    print(f"\n样本数: {len(targets)}")
    print(f"\n字符长度:")
    print(f"  平均: {char_mean:.1f}, 中位数: {char_median}, 范围: [{char_min}, {char_max}]")
    print(f"\nToken 估算:")
    print(f"  平均: {token_mean:.1f}, 中位数: {token_median}, 范围: [{token_min}, {token_max}]")

    # 🔥 关键诊断
    print(f"\n🔥 关键诊断:")
    print(f"  当前配置: MIN_NEW_TOKENS_TRAIN = {MIN_NEW_TOKENS_TRAIN}")
    ratio = token_mean / MIN_NEW_TOKENS_TRAIN

    if ratio > 3:
        print(f"  🔴 严重不匹配: Target平均 {token_mean:.1f} tokens 是 MIN_NEW_TOKENS {MIN_NEW_TOKENS_TRAIN} 的 {ratio:.1f}x")
        print(f"  → SFT训练模型生成~{token_mean:.1f}词，GRPO时被MIN={MIN_NEW_TOKENS_TRAIN}约束")
        print(f"  → EOS Suppressor强制禁止前{MIN_NEW_TOKENS_TRAIN}个token → 模型被迫续写")
        print(f"  → 不知道说什么 → 生成最确定token → Entropy崩溃")
        recommended = int(token_mean * 0.7)
        print(f"  ✅ 建议: MIN_NEW_TOKENS_TRAIN = {MIN_NEW_TOKENS_TRAIN} → {recommended}")
    elif ratio > 1.5:
        recommended = int(token_mean * 0.8)
        print(f"  🟡 中等不匹配: Target平均 {token_mean:.1f} > MIN {MIN_NEW_TOKENS_TRAIN}")
        print(f"  ✅ 建议: MIN_NEW_TOKENS_TRAIN → {recommended}")
    else:
        print(f"  ✅ 基本匹配 (ratio={ratio:.1f})")

    # Token分布
    bins = [0, 5, 10, 20, 50, 100, 200, 500]
    labels = ["0-5", "5-10", "10-20", "20-50", "50-100", "100-200", "200+"]
    counts = [0] * len(labels)

    for tl in token_lengths:
        for i in range(len(bins)-1):
            if bins[i] <= tl < bins[i+1]:
                counts[i] += 1
                break
        else:
            counts[-1] += 1

    print(f"\nToken分布:")
    for label, count in zip(labels, counts):
        pct = count / len(token_lengths) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:>10}: {count:>4} ({pct:>5.1f}%) {bar}")

    # 样本展示
    print(f"\n📝 随机样本 (3个):")
    for i in random.sample(range(len(targets)), min(3, len(targets))):
        t = targets[i]
        preview = t[:80] + "..." if len(t) > 80 else t
        print(f"\n  #{i} ({len(t)}字符, ~{tokenize_rough(t)} tokens):")
        print(f"    {preview}")

    return {"mean": token_mean, "median": token_median, "min": token_min, "max": token_max}

# ============================================================================
# 主逻辑：加载BBQ数据
# ============================================================================
print(f"\n{'='*80}")
print("📦 加载 BBQ 数据...")
print(f"{'='*80}")

bbq_targets = []

if BBQ_DIR and BBQ_DIR.exists():
    per_cat = N_BBQ_TRAIN // len(BBQ_FILES)

    for cat, filename in BBQ_FILES.items():
        fp = BBQ_DIR / filename
        if not fp.exists():
            print(f"⚠️  {filename} 不存在，跳过")
            continue

        data = read_jsonl(fp)
        if not data:
            continue

        # 自适应采样（与trainer.py一致）
        ambig = [x for x in data if x.get("context_condition") == "ambig"]
        disambig = [x for x in data if x.get("context_condition") != "ambig"]

        if disambig:
            n_disambig = int(per_cat * 0.6)  # 60% disambig
            n_ambig = per_cat - n_disambig

            picked = random.sample(disambig, min(n_disambig, len(disambig)))
            if ambig:
                picked += random.sample(ambig, min(n_ambig, len(ambig)))
        else:
            picked = random.sample(data, min(per_cat, len(data)))

        for item in picked:
            target = build_bbq_target(item)
            bbq_targets.append(target)

        print(f"  {cat}: {len(picked)} 样本")

    print(f"\nBBQ总计: {len(bbq_targets)} 样本")
else:
    print("❌ BBQ数据集路径不存在")

# ============================================================================
# 主逻辑：加载HaluEval数据
# ============================================================================
print(f"\n{'='*80}")
print("📦 加载 HaluEval 数据...")
print(f"{'='*80}")

halu_targets = []

if HALUEVAL_DIR and HALUEVAL_DIR.exists():
    per_subset = N_HALU_TRAIN // len(HALUEVAL_FILES)

    for subset, filename in HALUEVAL_FILES.items():
        fp = HALUEVAL_DIR / filename
        if not fp.exists():
            print(f"⚠️  {filename} 不存在，跳过")
            continue

        data = read_jsonl(fp)
        if not data:
            continue

        sampled = random.sample(data, min(per_subset, len(data)))

        for item in sampled:
            target = build_halueval_target(item, subset)
            halu_targets.append(target)

        print(f"  {subset}: {len(sampled)} 样本")

    print(f"\nHaluEval总计: {len(halu_targets)} 样本")
else:
    print("❌ HaluEval数据集路径不存在")

# ============================================================================
# 分析结果
# ============================================================================
bbq_stats = analyze_targets(bbq_targets, "BBQ (Fairness)")
halu_stats = analyze_targets(halu_targets, "HaluEval (Hallucination)")

all_targets = bbq_targets + halu_targets
all_stats = analyze_targets(all_targets, "总体 (BBQ + HaluEval)")

# ============================================================================
# 最终建议
# ============================================================================
print(f"\n{'='*80}")
print("💡 最终建议")
print(f"{'='*80}")

if all_stats:
    avg_len = all_stats["mean"]
    ratio = avg_len / MIN_NEW_TOKENS_TRAIN

    print(f"\n当前配置:")
    print(f"  MIN_NEW_TOKENS_TRAIN = {MIN_NEW_TOKENS_TRAIN}")
    print(f"  SFT Target 平均长度 ≈ {avg_len:.1f} tokens")
    print(f"  比例: {ratio:.1f}x")

    if ratio > 3:
        recommended = int(avg_len * 0.7)
        print(f"\n🔴 严重不匹配！这很可能是Entropy崩溃的根本原因！")
        print(f"\n修复步骤:")
        print(f"  1. 找到 trainer.py 中的 MIN_NEW_TOKENS_TRAIN 配置")
        print(f"  2. 修改: MIN_NEW_TOKENS_TRAIN = 5 → {recommended}")
        print(f"  3. 可选：同时提升 ENTROPY_COEF = 0.2 → 0.5")
        print(f"  4. 重新开始GRPO训练（SFT checkpoint可以保留）")
        print(f"\n预期效果:")
        print(f"  - EOS Suppressor 触发率: 100% → 30-50%")
        print(f"  - Entropy: 0.005 → 0.5-1.5 (短期)")
        print(f"  - Max prob: 99.9% → 60-90%")
    elif ratio > 1.5:
        recommended = int(avg_len * 0.8)
        print(f"\n🟡 中等不匹配，建议优化:")
        print(f"  MIN_NEW_TOKENS_TRAIN = {MIN_NEW_TOKENS_TRAIN} → {recommended}")
    else:
        print(f"\n✅ Target长度与MIN_NEW_TOKENS基本匹配")
        print(f"\nEntropy崩溃可能由其他原因引起，尝试:")
        print(f"  1. 提升 ENTROPY_COEF: 0.2 → 1.0")
        print(f"  2. 提升 TEMPERATURE: 0.9 → 1.2")
        print(f"  3. 降低 REP_PENALTY: 1.18 → 1.05")
else:
    print("\n❌ 无法分析（数据集未加载）")
    print("请检查数据路径配置")

print(f"\n{'='*80}")
print("✅ 分析完成！")
print(f"{'='*80}")
