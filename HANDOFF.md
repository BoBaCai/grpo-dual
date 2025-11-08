# GRPO Multi-Objective Training - Handoff Document

**Last Updated:** 2025-11-08
**Current Branch:** `claude/open-trainer-py-011CUp9RqkPbRBQPMVzBRuJ3`
**Status:** A+B修复已完成，等待训练验证

---

## 📋 项目概述

### 目标
使用 GRPO (Group Relative Policy Optimization) 对 Llama-3-8B-Instruct 进行多目标强化学习微调：
- **Fairness (BBQ数据集)**: 减少偏见，公平回答问题
- **Hallucination (HaluEval数据集)**: 减少幻觉，基于证据回答

### 技术栈
- Base Model: `meta-llama/Meta-Llama-3-8B-Instruct`
- Method: GRPO + LoRA + Branched KL Control
- Framework: PyTorch + Transformers + PEFT

---

## 🔥 关键问题历史

### 问题1：熵塌陷（Entropy Collapse）
**症状：**
- 模型输出高度确定性（max prob ≈ 0.99999）
- 熵值极低（0.2-0.7，正常应 >1.5）
- 所有生成都是相同模板："The context does not provide sufficient information..."
- 同一prompt的K个候选几乎相同 → advantage=0 → 无梯度信号

**根本原因：**
1. `MIN_NEW_TOKENS_TRAIN=30` 强制所有回答≥30 tokens
2. Judge对"安全废话模板"给正分
3. 导致模型收敛到单一模板输出

**修复方案（A+B，已完成）：**
- ✅ A: `MIN_NEW_TOKENS_TRAIN: 30 → 5` (trainer.py:226)
- ✅ B: 模板检测器，惩罚逃避回答 (trainer.py:1594-1621)

**代码位置：**
```
grpo-dual/src/grpo/trainer.py
  - Line 226: MIN_NEW_TOKENS_TRAIN = 5
  - Line 1586-1621: MultiCloudJudge.evaluate() 中的模板检测器
```

---

### 问题2：数据集使用问题（Pi专家分析）

#### BBQ数据集（已正确处理 ✅）
**关键点：**
- Unknown选项位置是动态的（不固定在A/B/C某个位置）
- 必须通过 `answer_info[*][1]=="unknown"` 判定
- Ambiguous样本：必须选Unknown
- Disambiguated样本：有明确正确答案

**代码验证：**
```python
# Line 1126-1133: _find_unknown_option 正确使用answer_info
if val[1]=="unknown":
    return chr(65+idx)  # 动态确定A/B/C
```
✅ 已正确实现

#### HaluEval数据集（部分问题 ⚠️）

**问题2.1: General子集噪声严重**
- "幻觉"概念混用：事实错误、不完整回答、能力声明、格式问题全混在一起
- 815个yes标注中，约13个hallucination_spans为空
- 约200+个涉及"As an AI language model..."被标为幻觉
- **影响：** reward信号互相矛盾，模型倾向保守模板

**问题2.2: 配对样本未充分利用**
- qa/dialogue子集有 `right_answer` 和 `hallucinated_answer`
- 当前只用了 `right_answer` 做SFT/target
- 未做对比学习（positive vs negative）
- **影响：** 模型只知道"正确"，不知道"幻觉"长什么样

**代码位置：**
```
grpo-dual/src/grpo/trainer.py
  - Line 1163-1276: HaluEvalAdapter.load_samples()
  - Line 1220: 只用right_answer（待改进）
  - Line 1234: 只用right_response（待改进）
  - Line 1255-1274: general子集处理（可能需要过滤）
```

---

## ✅ 已完成的修复

### Commit: f140a1c "Fix entropy collapse: reduce MIN_NEW_TOKENS and penalize template responses"

**修改1: 放松解码策略**
```python
# trainer.py:226
MIN_NEW_TOKENS_TRAIN = 5  # 从30降到5
```
**效果：** 允许短回答，同一prompt的K个候选产生差异 → 恢复梯度信号

**修改2: 模板检测器**
```python
# trainer.py:1594-1621
# 检测6种模板短语：
template_phrases = [
    "does not provide sufficient information",
    "cannot be determined",
    "not enough information",
    "insufficient information",
    "unable to determine",
    "context does not"
]

# 分层惩罚：
- BBQ disambiguated: -0.7（有明确答案却逃避）
- HaluEval qa/dialogue/summarization: -0.5（有knowledge却逃避）
- Ambiguous/general: 0.0（勉强可以，但不给正分）
```
**效果：** 模板不再是"安全最优策略"

---

## 📊 当前训练状态

### 配置 (trainer.py:126-286)
```python
# 基础
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
GRPO_STEPS = 500
GRPO_BATCH_SIZE = 2
K_ROLLOUTS = 4  # 每个prompt生成4个候选
GRADIENT_ACCUMULATION_STEPS = 2

# 生成配置（已修复）
MAX_NEW_TOKENS_TRAIN = 128
MIN_NEW_TOKENS_TRAIN = 5  # ⚡ 已修复
TEMPERATURE_TRAIN = 0.9
TOP_K_TRAIN = 100
TOP_P_TRAIN = 0.9
REP_PENALTY_TRAIN = 1.18

# 数据
N_BBQ_TRAIN = 1100
N_HALU_TRAIN = 400

# Reward Scale（多目标平衡）
FAIRNESS_REWARD_SCALE = 0.7
HALLUCINATION_REWARD_SCALE = 1.0

# KL控制（分支化）
beta_f_init = 0.30  # Fairness
beta_h_init = 0.30  # Hallucination
```

### 待观察指标（前50步最关键）

#### 🔥 优先级1：熵是否恢复（前5步）
```
[Fairness诊断@stepX]
  Entropy: X.XXX
```
**期望：** >1.0（理想>1.5）
**如果：** 仍<0.5 → A+B修复失败

#### 🔥 优先级2：梯度信号（前10步）
```
Reward统计:
  Fairness: std=X.XXX
  Hallucination: std=X.XXX

梯度信号强度:
  Fairness signal: X.XXXX
  Hallucination signal: X.XXXX
```
**期望：** std>0.1, signal>0
**如果：** 大量std=0 → 仍有模式坍塌

#### ⭐ 优先级3：模板检测器是否生效（前20步）
```
[Judge@stepX] providers={'template_detector': X, ...}
```
**期望：** 前5步有一定比例，5-20步逐步减少
**如果：** 持续>50% → 策略仍锁在模板

#### ⭐ 优先级4：生成多样性
**期望：** 不再全是"insufficient information"，有基于context的实质回答

---

## 🛠️ 待修复问题（根据训练结果决定优先级）

### Plan B1: 过滤HaluEval-general噪声（如果Hallucination信号仍弱）

**代码位置：** trainer.py:1255-1274

**修复方案：**
```python
if sub == "general":
    filtered = []
    for it in data:
        if it.get("hallucination") == "no":
            filtered.append(it)
        elif it.get("hallucination") == "yes":
            spans = it.get("hallucination_spans", [])
            # 只保留明确事实错误，排除不完整/能力声明/格式问题
            if spans and not any(keyword in str(spans).lower()
                                 for keyword in ["incomplete", "cannot", "ai language model", "format"]):
                filtered.append(it)
    data = filtered
```

### Plan B2: 启用HaluEval配对对比学习（提升效果）

**代码位置：** trainer.py:1220, 1234

**修复方案：**
```python
# qa子集：生成positive和negative两个样本
# Positive
out.append(Sample(
    id=f"halu_{sub}_{i}_pos",
    task="hallucination",
    prompt=prompt,
    target=build_target(it['right_answer']),
    meta={**meta, "label": "positive"}
))

# Negative
out.append(Sample(
    id=f"halu_{sub}_{i}_neg",
    task="hallucination",
    prompt=prompt,
    target=build_target(it['hallucinated_answer']),
    meta={**meta, "label": "negative"}
))

# Judge需要相应调整，对negative样本给负分
```

### Plan B3: 降低General权重（快速方案）

**代码位置：** trainer.py:1164

**修复方案：**
```python
# 加权采样，降低general子集权重
per_weights = {
    "qa": 1.5,
    "dialogue": 1.5,
    "general": 0.5,  # 降低权重
    "summarization": 1.0
}
per = max(1, int(n_total * per_weights[sub] / sum(per_weights.values())))
```

### Plan B4: 检查Tokenization截断问题（如果两个任务都仍弱）

**代码位置：** trainer.py:2087, 2256, 2501, 2506, 2532

**潜在问题：**
- 多处使用 `truncation=True, max_length=896`
- BBQ context可能被截断，导致disambiguated样本看起来像ambiguous
- SFT时可能把target截掉

**验证脚本：**
```python
# 检查BBQ样本tokenization后的长度
for sample in bbq[:100]:
    formatted = apply_chat_template(tokenizer, sample.prompt, system_msg)
    tokens = tokenizer(formatted, truncation=False)
    if len(tokens['input_ids']) > 700:
        print(f"⚠️ Sample {sample.id}: {len(tokens['input_ids'])} tokens (可能被截断)")
        print(f"Context condition: {sample.meta.get('context_condition')}")
```

---

## 🎯 决策树（20步后）

```
训练20步后观察结果
│
├─ Fairness恢复 + Hallucination恢复
│  └─> ✅ A+B成功，继续训练到100-200步
│
├─ Fairness恢复 + Hallucination仍弱
│  └─> ⚡ General噪声主导
│     └─> 实施Plan B1（过滤general）或B3（降低权重）
│
├─ Fairness仍弱 + Hallucination仍弱
│  └─> 🔍 其他问题
│     └─> 实施Plan B4（检查截断）
│     └─> 检查ambig/disambig采样比例
│
└─ Fairness仍弱 + Hallucination恢复
   └─> 🤔 不太可能，深入诊断BBQ
```

---

## 📁 关键文件位置

### 主训练脚本
```
grpo-dual/src/grpo/trainer.py (3509行)
  - Line 126-286: Config配置
  - Line 226: MIN_NEW_TOKENS_TRAIN (⚡已修复)
  - Line 970-1009: read_json_flex (JSONL读取)
  - Line 1031-1157: BBQAdapter
  - Line 1126-1133: _find_unknown_option (✅正确实现)
  - Line 1158-1276: HaluEvalAdapter (⚠️待优化)
  - Line 1369-1646: MultiCloudJudge
  - Line 1586-1621: evaluate() + 模板检测器 (⚡已添加)
  - Line 2681-3220: grpo_train (主训练循环)
```

### 数据文件
```
/workspace/data/bbq/
  - Gender_identity.jsonl (5672条，JSONL格式)
  - Disability_status.jsonl (1556条，JSONL格式)
  - ... (其他9个类别)

/workspace/data/halueval/
  - qa_data.json (10000条，JSONL格式，有配对)
  - dialogue_data.json (10000条，JSONL格式，有配对)
  - general_data.json (4507条，JSONL格式，⚠️噪声严重)
  - summarization_data.json (JSONL格式)
```

### 输出文件
```
/workspace/multiobjective_llama/<RUN_ID>/
  - train_step_metrics.csv (逐步指标)
  - train_step_metrics.jsonl (备用)
  - training_metrics_summary.json (最终汇总)
```

---

## 🔬 诊断方法

### 快速检查清单（训练前）
```bash
# 1. 确认配置已更新
grep "MIN_NEW_TOKENS_TRAIN = 5" grpo-dual/src/grpo/trainer.py
grep "template_phrases = \[" grpo-dual/src/grpo/trainer.py

# 2. 检查git状态
git log -1 --oneline
# 应该看到: f140a1c Fix entropy collapse...

# 3. 确认在正确分支
git branch --show-current
# 应该是: claude/open-trainer-py-011CUp9RqkPbRBQPMVzBRuJ3
```

### 训练中观察（前20步）
关注终端输出中的：
1. **Fairness诊断模块** - 熵值、生成长度、生成内容
2. **Reward Scale诊断** - 信号强度、std、EMA比值
3. **Judge provider分布** - template_detector出现频率
4. **长度惩罚统计** - 多少样本被惩罚

### 训练后分析（50步+）
```bash
# 查看CSV（推荐用pandas）
import pandas as pd
df = pd.read_csv("/workspace/multiobjective_llama/<RUN_ID>/train_step_metrics.csv")

# 关键列
df[['step', 'kl_f', 'kl_h', 'reward_f_mean', 'reward_h_mean',
    'clip_frac', 'gen_len_f_mean', 'gen_len_h_mean',
    'trunc_frac_f', 'trunc_frac_h']].head(20)

# 绘制趋势
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
df.plot(x='step', y='reward_f_mean', ax=axes[0,0], title='Fairness Reward')
df.plot(x='step', y='reward_h_mean', ax=axes[0,1], title='Hallucination Reward')
df.plot(x='step', y='gen_len_f_mean', ax=axes[1,0], title='Fairness Length')
df.plot(x='step', y='gen_len_h_mean', ax=axes[1,1], title='Hallucination Length')
plt.tight_layout()
plt.savefig('training_trends.png')
```

---

## 📚 参考资料

### Pi专家的分析要点

#### BBQ数据集（已验证无问题）
- ✅ 结构良好，标注一致
- ✅ Ambiguous/Disambiguated设计正确
- ✅ 标签分布均衡（各选项、各极性都成对出现）
- ⚠️ Unknown选项位置是动态的（必须用answer_info判定）

#### HaluEval数据集
**General子集问题：**
- "幻觉"概念混用（事实错误+不完整+能力声明+格式问题）
- 约815个yes标注，其中13个spans为空
- 约200+个"As an AI..."被标为幻觉
- 结论：需要过滤，只保留明确事实错误

**Dialogue子集（质量好）：**
- ✅ 成对标注（right vs hallucinated）
- ✅ 基于knowledge的一致性
- ✅ 只有轻微噪声

**QA子集（质量好）：**
- ✅ 成对标注
- ✅ 清晰的事实依据

### 算法参考

#### BAPO (Balanced Advantage Policy Optimization)
- 动态调整PPO裁剪边界（不对称）
- c_low: 0.6→0.9, c_high: 1.2→3.0
- 目标：平衡positive/negative样本的梯度贡献
- 适用场景：防止熵塌陷，鼓励探索

#### DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)
- Clip-Higher: 上界从1.2→1.28
- Dynamic Sampling: 过滤accuracy=1或0的prompt组
- Token-Level Loss: 对所有token聚合
- Overlong Reward Shaping: 惩罚超长回答

---

## 🚀 下一步计划

### 短期（今天）
1. ✅ 用A+B修复的代码跑50步
2. ✅ 观察前20步诊断输出
3. ✅ 根据结果决定是否需要Plan B

### 中期（如果A+B成功）
1. 继续训练到100-200步
2. 观察收敛情况
3. 评估Pareto前沿

### 中期（如果需要Plan B）
1. 根据决策树选择对应方案
2. 实施修复
3. 重新训练50步验证

### 长期（优化方向）
1. 实施HaluEval配对对比学习（Plan B2）
2. 考虑BAPO/DAPO技术（不对称裁剪、动态采样）
3. 增加Best-of-N或Rejection Sampling

---

## ⚠️ 已知限制和注意事项

### 环境
- GPU显存限制：已降低batch size和LoRA rank
- API配额：OpenAI + Claude双云，有heuristic兜底
- 网络稳定性：已实现重试和指数退避

### 数据
- BBQ只用了2个类别（Gender, Disability），还有9个类别未用
- HaluEval的general子集噪声需要处理
- 配对样本未充分利用

### 训练
- SFT target可能与RL阶段的模板检测器有轻微冲突（ambiguous样本）
- Tokenization可能截断长context（待验证）
- KL控制的β值可能需要根据训练进度调整

---

## 📞 交接检查点

接手此项目的人应该能够：

### 必需了解
1. ✅ 熵塌陷问题的根本原因和修复方案
2. ✅ BBQ数据集的正确使用方式（动态unknown）
3. ✅ HaluEval数据集的问题和潜在修复方案
4. ✅ 如何解读训练日志（熵、梯度信号、provider分布）

### 必需掌握
1. ✅ 运行训练并观察前20步
2. ✅ 根据决策树选择对应的Plan B
3. ✅ 修改代码实施Plan B（如果需要）
4. ✅ 分析CSV文件和绘制趋势图

### 可选技能
1. BAPO/DAPO算法的实现
2. Pareto前沿优化
3. 更复杂的reward shaping

---

## 📝 更新日志

**2025-11-08:**
- ✅ 完成A+B修复（MIN_NEW_TOKENS降低 + 模板检测器）
- ✅ Commit f140a1c推送到远程分支
- ✅ 整理Pi专家的数据集分析
- ✅ 准备Plan B方案（待训练结果决定）
- 📋 创建本交接文档

**待更新（训练完成后）：**
- [ ] 前20步的实际观察结果
- [ ] 是否需要实施Plan B（哪个）
- [ ] 最终训练效果和收敛情况

---

**文档结束。如有疑问，请参考trainer.py中的详细注释或重新阅读本文档的相关章节。**
