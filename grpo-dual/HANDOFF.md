# GRPO Multi-Objective Training - Handoff Document

**Last Updated:** 2025-11-08
**Current Branch:** `claude/open-trainer-py-011CUp9RqkPbRBQPMVzBRuJ3`
**Status:** 关键工程问题修复完成（串行生成 + Advantage计算 + 熵奖励 + KL约束 + 模板检测增强）

---

## 📋 项目概述

### 目标
使用 GRPO (Group Relative Policy Optimization) 对 Llama-3-8B 进行多目标强化学习微调：
- **Fairness (BBQ数据集)**: 减少偏见，公平回答问题
- **Hallucination (HaluEval数据集)**: 减少幻觉，基于证据回答

### 技术栈
- Base Model: `meta-llama/Meta-Llama-3-8B-Instruct` **【实验后回到Instruct】Base model无法理解格式**
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

### 问题3：Advantage计算抹平梯度信号（Pi专家发现，最致命！）

**症状：**
- 训练日志显示：`Reward (归一化后): 0.000`, `信号强度: 0.0000`
- 大量步骤的Fairness和Hallucination信号强度都<1e-5
- 即使reward有差异，advantage仍为0
- 训练实际上"原地踏步"

**根本原因：**
Pi通过分析训练日志发现：`compute_group_advantages` 使用**组内标准化**：

```python
# Line 2569-2577: compute_group_advantages
r = rewards.view(B, k)  # [batch_size, K个候选]
mean = r.mean(dim=1, keepdim=True)  # 组内均值
std = r.std(dim=1, keepdim=True).clamp_min(1e-6)  # 组内标准差
adv = ((r - mean) / std).view(-1)  # 组内z-score标准化
```

**问题机制：**
1. 当同一prompt的K=4个候选都输出相同模板（熵塌陷）
2. 它们的reward完全相同（如都是-0.7）
3. `std = 0` (clamp到1e-6)
4. `adv = (r - mean) / 1e-6 ≈ 0`
5. **这一组的4个样本梯度全部为0！**
6. 如果50%以上的组都这样 → 整个batch几乎没有学习信号

**与问题1的关系：**
- 问题1（熵塌陷）导致K个候选相同
- 问题3（组内标准化）将"相同"转化为"梯度为0"
- 形成恶性循环：熵塌陷 → 无梯度 → 策略不动 → 继续塌陷

**修复方案（C2，已完成）：**
- ✅ C2: 组内std监控和警告 (trainer.py:2933-2965)
- 🔄 预期A+B修复能让大部分组产生差异（std>0.01）
- ⚠️ 如果>50%组仍然std<0.01，需要Plan C1（全局baseline重构）

**代码位置：**
```
grpo-dual/src/grpo/trainer.py
  - Line 2569-2577: compute_group_advantages (问题根源)
  - Line 2933-2965: C2监控逻辑（已添加）
```

**Pi的其他发现：**
1. **KL控制过严：** 目标KL=0.035过小，实际KL=2.x，β却继续增大 → 锁死策略
2. **Judge逻辑可疑：** 对模板短语的奖励不一致
3. **缺少熵正则化：** 应该给policy加entropy bonus

---

### 问题4：批量生成导致K个候选相同（工程问题，最致命！）

**症状（从训练日志观察）：**
- 即使MIN_NEW_TOKENS=5，K=4个候选仍然高度相同
- 同组内reward全为1或全为-1，std≈0
- 模板检测器可能在工作，但无效（4个都是模板→4个都得-1→std仍然=0）

**根本原因：**
发现于 `generate_candidates_batch` (trainer.py:2063-2248)

```python
# 旧代码（问题）
batch_prompts = []
for p in formatted_prompts:
    batch_prompts.extend([p]*k)  # 每个prompt重复k=4次

inputs = tokenizer(batch_prompts, ...)  # 一次性tokenize所有
out = model.generate(**inputs, do_sample=True, ...)  # 一次性generate
```

**问题机制：**
1. 同一prompt的k个副本在**同一个forward pass**中
2. 即使`do_sample=True`，在同一个batch中，random state对同一input是相同的
3. 当模型概率分布极度尖锐（Pi观察到top-1 prob 0.94~0.999999）时：
   - Sampling几乎总是选择top-1 token
   - K个候选产生相同输出
4. **即使有模板检测器，如果4个都是模板→全得-1.0→std=0→无梯度**

**为什么之前没发现：**
- A+B+C修复都聚焦在"如何让模型不输出模板"
- 但忽略了"即使模型想输出不同内容，生成机制也不允许"
- 这是**工程实现问题**，不是超参或算法问题

**修复方案（已实施）：**
- ✅ 改为串行生成：对每个prompt独立生成k次
- 每次generate调用，random state都会变化
- 即使top-1 prob很高，多次采样也能产生差异

**代码位置：**
```
grpo-dual/src/grpo/trainer.py
  - Line 2063-2178: generate_candidates_batch（完全重写为串行模式）
```

---

### 问题5：即使候选文本不同，Judge仍给出相同分数（工程问题）

**症状（从实验观察）：**
- 串行生成修复后，K=4个候选文本确实不同（4/4唯一）
- 但所有候选都选择正确答案 → 都得满分 → std=0
- 简单问题上无法产生梯度信号

**根本原因：**
BBQ Judge评分逻辑过于粗糙（trainer.py:1471-1524）

```python
# 旧逻辑：二元评分
if chosen_answer == correct_answer:
    score = 1.0  # 全对
elif chosen_answer == unknown_option:
    score = -0.3
else:
    score = -1.0  # 全错
```

**问题机制：**
1. 即使4个候选reasoning质量不同（有的详细引用context，有的简略）
2. 只要都选择正确答案 → 都得1.0分
3. reward完全相同 → std=0 → advantage=0 → 无梯度
4. **模型无法学到"如何更好地reasoning"**

**实验验证（test_improved_judge.py）：**
使用之前实验的4个真实候选（都选B) Teacher，但reasoning略有不同）：
- 旧Judge：[1.0, 1.0, 1.0, 1.0] → std=0
- 新Judge：[0.70, 1.00, 1.00, 0.70] → std=0.15 ✅

**修复方案（Option A，已实施）：**
- ✅ 改进BBQ Judge，不只看答案正确性，还评估reasoning质量
- ✅ 添加`_assess_reasoning_quality()`方法，评估3个标准：
  1. 是否引用context关键词（未引用 -0.3）
  2. Justification长度是否合理（<5词 -0.2，>50词 -0.1，缺失 -0.3）
  3. 是否过度模板化（≥2个模板短语 -0.1）
- ✅ 分数从二元（1.0/-1.0）变为多级（1.0优秀 / 0.7良好 / 0.5差劲）

**代码位置：**
```
grpo-dual/src/grpo/trainer.py
  - Line 1471-1524: _evaluate_bbq_fairness（修改为调用质量评估）
  - Line 1526-1586: _assess_reasoning_quality（新增方法）
test_improved_judge.py（验证脚本）
```

**效果：**
- ✅ 即使所有候选都选对，也能产生分数差异（std=0.15 >> 0.05阈值）
- ✅ 鼓励模型学习更好的reasoning（引用context、合理长度、避免模板化）
- ✅ 对Base model换回Instruct的补充方案

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

### Commit: (待提交) "Add C2 fix: monitor and warn about zero-gradient groups"

**修改3: C2组内std监控**
```python
# trainer.py:2933-2965
# 在compute_group_advantages之后添加监控逻辑

# 检测每组的reward std
for i in range(B):
    group_rewards = rewards_list[i*K : (i+1)*K]
    group_std = np.std(group_rewards)

    if group_std < 0.01:  # 组内几乎相同
        zero_gradient_groups += 1
        # 前20步详细打印该组的rewards和responses

# 统计并警告
if zero_gradient_groups > 0:
    ratio = zero_gradient_groups / B
    print(f"⚠️ {zero_gradient_groups}/{B} 组({ratio:.1%})的reward std<0.01，梯度信号被抹平")

    if ratio > 0.5:
        print("⚠️⚠️⚠️ 超过50%的组无梯度！A+B修复可能未生效")
```

**效果：**
1. **实时监控**：立即发现有多少组的梯度被抹平
2. **早期预警**：如果>50%无梯度，说明A+B未生效，需要Plan C1
3. **详细诊断**：前20步打印每个无梯度组的rewards和responses，方便定位问题
4. **不影响训练**：只是监控和警告，不修改梯度计算（A+B应该能让大部分组产生差异）

---

### Commit: (待提交) "Implement Plan C: fix advantage calculation and enhance exploration"

**基于Pi专家的训练日志诊断，实施全面修复：**

#### 修改4: Advantage计算修复（最核心！）
```python
# trainer.py:2569-2608
# 原逻辑：组内z-score标准化
adv = (r - mean) / std  # std=0时梯度为0

# 新逻辑：检测std，退化到安全模式
if group_std < 0.01:
    # 整组同奖，直接用reward（已过全局归一化）
    group_adv = group_rewards
else:
    # 有多样性，用中心化（不除std，保留scale）
    group_adv = group_rewards - group_mean
```

**效果：**
- ✅ 避免除以0导致的梯度抹平（即使K个候选完全相同）
- ✅ 保留GRPO组内相对优势概念（有多样性时）
- ✅ reward已过全局归一化，可直接当advantage使用
- ✅ 退化模式：无多样性时，至少有一致的梯度方向（鼓励/抑制）

#### 修改5: 增强熵正则化
```python
# trainer.py:203
ENTROPY_COEF = 2.0  # 从0.5→2.0
```

**原因：** 策略极度尖锐(top-1 prob 0.94~0.999999)，需要更强的熵奖励对抗塌陷

**效果：** Loss中entropy项权重增大4倍，显著鼓励探索

#### 修改6: 降低KL约束
```python
# trainer.py:2786-2788
beta_f_init = 0.05  # 从0.30→0.05
beta_h_init = 0.05  # 从0.30→0.05
```

**原因：**
- 严格KL约束(β=0.30)锁住模型，几乎不更新
- 参考DeepSeekMath使用0.04
- 给模型更多自由度偏离参考模型

**效果：** KL惩罚降低6倍，模型可以更大胆地探索

#### 修改7: 模板检测器增强
```python
# trainer.py:1595-1628
# 扩展短语列表：6种→13种
template_phrases = [
    "does not provide sufficient information",
    "cannot be determined",
    # ... 新增7种
    "ambiguous", "unclear from the context", "not specified", ...
]

# 加大惩罚力度：
- BBQ disambiguated: -0.7 → -1.0（最大负分）
- HaluEval qa/dialogue/summarization: -0.5 → -0.8
- Ambiguous/general: 0.0 → -0.2（轻微负分）
```

**效果：**
- 更全面的模板识别
- 更强的惩罚信号
- 配合Advantage修复，即使全组是模板也能产生梯度

#### 修改8: 串行生成修复（最关键的工程修复！）
```python
# trainer.py:2063-2178
# 旧逻辑：批量生成所有prompt*k（同一forward pass）
batch_prompts.extend([p]*k)  # 重复k次
out = model.generate(**inputs)  # 一次性生成

# 新逻辑：对每个prompt串行生成k次
for prompt_idx, formatted_prompt in enumerate(formatted_prompts):
    for candidate_idx in range(k):
        # 每次独立generate，random state变化
        out = model.generate(**inputs, do_sample=True, ...)
        # decode and collect...
```

**原因：**
- 批量生成时，同一prompt的k个副本在同一forward pass中
- Random state相同 + 模型概率极度尖锐 → k个输出相同
- 即使do_sample=True也无效

**效果：**
- ✅ 每次generate独立采样，即使top-1 prob很高也能产生差异
- ✅ 直接解决"同组reward相同→std=0→无梯度"的根本原因
- ✅ 这是工程问题，不是超参问题！

**代价：**
- 生成时间增加k倍（但batch size=2很小，影响可接受）
- 比"无法训练"的代价小得多

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

# 熵正则化（Plan C增强）
ENTROPY_COEF = 2.0  # ⚡ 从0.5提升到2.0，对抗严重熵塌陷

# KL控制（分支化，Plan C降低）
beta_f_init = 0.05  # ⚡ 从0.30降到0.05，给模型更多自由度
beta_h_init = 0.05  # ⚡ 从0.30降到0.05
```

### 待观察指标（前50步最关键）

#### 🔥🔥🔥 优先级0：训练是否真正开始学习（观察前10步）
```
⚠️ [Step X] Y/B 组(Z%)的reward std<0.01
```
**Plan C已实施，即使检测到零梯度组，也不影响训练（已修复advantage计算）**

**新的关注点：**
1. **模型是否开始探索？** 观察生成多样性（不再全是"insufficient information"）
2. **熵是否上升？** Entropy从<0.5上升到>1.0
3. **KL是否合理？** beta=0.05下，KL应该在0.1-0.5之间（比之前大）
4. **Reward是否有波动？** 不应该全是常数

**关键：** Plan C已修复advantage计算，即使std<0.01也有梯度。现在要看的是模型是否真的在动。

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

### ~~Plan C1: 全局Baseline重构~~ ✅ 已实施为Plan C修改4

**状态：** ✅ 已完成（实施了混合方案：检测std并退化）

**代码位置：** trainer.py:2569-2608 (compute_group_advantages)

**实施的方案（混合方案）：**
```python
# 检测std，选择合适的advantage计算方式
if group_std < 0.01:
    # 整组同奖 → 直接用reward（避免除以0）
    group_adv = group_rewards
else:
    # 有多样性 → 用中心化（保留scale）
    group_adv = group_rewards - group_mean
```

**下面是原计划的其他方案（供参考）：**

**备选方案1：使用全局EMA baseline：**
```python
# 在grpo_train函数初始化时添加
global_baseline_ema = {"fairness": 0.0, "hallucination": 0.0}

# 修改compute_group_advantages
def compute_group_advantages(rewards: torch.Tensor, tasks: List[str], k: int,
                             global_baseline: Dict[str, float]) -> torch.Tensor:
    """使用全局EMA baseline而非组内mean"""
    Bk = rewards.numel()
    B = Bk // k
    adv = torch.zeros_like(rewards)

    for i in range(B):
        task = tasks[i]  # 这组的任务类型
        group_rewards = rewards[i*k : (i+1)*k]

        # 使用全局baseline（跨batch EMA）
        baseline = global_baseline.get(task, 0.0)
        group_adv = group_rewards - baseline

        adv[i*k : (i+1)*k] = group_adv

    return adv.clamp(-config.ADV_CLIP, config.ADV_CLIP)

# 每步更新全局baseline
for task in ["fairness", "hallucination"]:
    task_rewards = rewards[task_mask].mean().item()
    global_baseline_ema[task] = 0.99 * global_baseline_ema[task] + 0.01 * task_rewards
```

**修复方案（方案2：检测并跳过std过小的组）：**
```python
# 在compute_group_advantages内部
std = r.std(dim=1, keepdim=True).clamp_min(1e-6)
std_too_small = (std < 0.01).squeeze()

adv = torch.zeros_like(r)
if std_too_small.any():
    # 对std过小的组，直接用 r - mean（不除以std）
    adv[std_too_small] = (r - mean)[std_too_small]
if (~std_too_small).any():
    # 对正常组，做标准化
    adv[~std_too_small] = ((r - mean) / std)[~std_too_small]
```

**优点：**
- 方案1：彻底解决问题，即使所有组都相同也有梯度
- 方案2：简单，保留大部分原有逻辑

**缺点：**
- 方案1：改动较大，需要仔细测试
- 方案2：治标不治本，如果所有组都相同仍然问题大

**推荐：** 如果>70%组无梯度，用方案1；如果30-50%，可以尝试方案2

---

## 🎯 决策树（20步后）- Plan C已实施版

### 第一步：模型是否在学习？（前10步）

```
观察前10步的关键指标
│
├─ 熵仍然很低（<0.5）+ 生成仍高度相似 + KL几乎为0
│  └─> 🚨 致命！模型被锁死
│     ├─ 可能原因1：ENTROPY_COEF=2.0仍不够，继续增大到5.0
│     ├─ 可能原因2：beta=0.05仍太大，降到0.01
│     ├─ 可能原因3：基座模型先验太强，考虑增加temperature
│     └─> ⚡ 调整超参后重新训练
│
├─ 熵有上升（0.5→1.0）+ 生成有多样性 + KL在0.1-0.5
│  └─> ✅ Plan C生效！模型开始探索
│     └─> 继续观察20-50步，看是否收敛
│
└─ 熵剧烈波动 + Reward崩溃（全是极端值）
   └─> ⚠️ 探索过头，不稳定
      ├─ 降低ENTROPY_COEF（2.0→1.0）
      ├─ 或增大beta（0.05→0.10）
      └─> 重新训练
```

### 第二步：如果模型开始学习，观察任务表现（20-50步）

```
训练20步后观察结果
│
├─ Fairness恢复 + Hallucination恢复
│  └─> ✅✅✅ 完全成功，继续训练到100-200步
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

**关键：** Plan C已修复advantage计算，现在关注点是模型是否真的在动（熵上升、生成多样化）。

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

**2025-11-08 (Session 1):**
- ✅ 完成A+B修复（MIN_NEW_TOKENS降低 + 模板检测器）
- ✅ Commit f140a1c推送到远程分支
- ✅ 整理Pi专家的数据集分析
- ✅ 添加C2监控（零梯度组检测）
- ✅ Commit 7294249推送（C2监控 + HANDOFF创建）

**2025-11-08 (Session 2 - 当前):**
- ✅ 接收Pi专家的训练日志诊断（6点总结）
- ✅ **实施Plan C全面修复：**
  - ✅ 修改compute_group_advantages（避免梯度抹平）
  - ✅ 增强熵正则化（ENTROPY_COEF: 0.5→2.0）
  - ✅ 降低KL约束（beta: 0.30→0.05）
  - ✅ 增强模板检测器（13种短语，更大惩罚）
- ✅ Commit f3f5c7d推送（Plan C全面修复）
- 🔍 **发现工程根本问题：批量生成导致K个候选相同**
- ✅ **实施串行生成修复（问题4）：**
  - ✅ 完全重写generate_candidates_batch为串行模式
  - ✅ 每个prompt独立生成k次，确保random state变化
  - ✅ 直接解决"同组reward相同→std=0→无梯度"
- ✅ 更新HANDOFF.md（记录串行生成修复）
- ✅ Commit 9a6f525推送（串行生成修复）
- ✅ 创建test_serial_generation.py实验脚本
- ✅ Commit 5c5e710推送（实验脚本）
- 🧪 **运行Instruct model实验验证串行生成效果：**
  - ✅ 串行生成确实产生字面差异（4/4唯一）
  - ⚠️ 但熵仍极低（0.2-0.4），实质内容高度相似
  - ⚠️ 简单问题上4个候选都选对 → reward相同 → std=0
- 🧪 **Base model实验（失败）：**
  - 💡 假设：Base model没有过强先验，更容易产生多样性
  - ✅ 创建test_base_model.py并运行实验
  - ❌ 结果：Base model输出乱码，完全不理解格式
  - ❌ 熵仍然低（0.27-0.52），且内容不可用
  - 💡 **决策：回到Instruct model，改进Judge评分**
- ✅ **实施Option A：改进Judge产生分数差异（问题5）：**
  - ✅ 回到Instruct model（BASE_MODEL: Meta-Llama-3-8B-Instruct）
  - ✅ 改进BBQ Judge评分，不只看答案正确性，还评估reasoning质量
  - ✅ 添加_assess_reasoning_quality()方法（引用context、长度、模板化）
  - ✅ 分数从二元（1.0/-1.0）变为多级（1.0/0.7/0.5）
  - ✅ 创建test_improved_judge.py验证
  - ✅ 实验结果：scores=[0.70, 1.00, 1.00, 0.70], std=0.15 ✅
- ✅ Commit e51938b推送（Option A: Improve Judge scoring）
- ✅ Commit bb009a5推送（Update HANDOFF.md）

**2025-11-08 (Session 3 - 激进修复):**
- 🔥 **用户反馈："这真的是新日志"** - 前面修复仍不够，问题依然存在
- 🔍 **用户提供详细诊断（直接翻译版）：**
  - 熵塌陷严重（0.017-0.055，极低）
  - Reward std=0（所有candidates得分相同）
  - 模型输出模板化："The context does not provide sufficient information"
  - >50%组无梯度
  - 核心问题："模型学会用单一、安全模板糊弄所有probe → reward常数 → RL无法学习"
- 🎯 **用户建议5大措施：**
  1. 增加候选多样性（去重、重采）
  2. 多级reward（不只1.0/-0.7两档）
  3. 对模板式输出直接负奖励
  4. 放宽采样参数（top-p, temperature）
  5. 精细化reward设计
- ✅ **实施激进修复（核选项）：**
  - ✅ 超严格reasoning quality评估（检测13种逃避短语→-0.5，实体引用，长度10-40词，模板短语1次扣分，词汇重复度检查）
  - ✅ Jaccard去重机制（相似度>0.65→丢弃重采，最多3次重试）
  - ✅ 激进采样参数（temp=1.1, top_k=150, top_p=0.95, rep_penalty=1.25）
  - ✅ 创建test_aggressive_judge.py验证
- ✅ Commit 5495a32推送（AGGRESSIVE FIX: all user-recommended measures）
- 📊 **预期效果：**
  - Candidates必须35%+不同（Jaccard<0.65）否则重采
  - 模板输出得-0.5至-1.0惩罚
  - 即使正确答案，根据reasoning质量得分0.3-1.0
  - 应产生reward_std>0.1，即使在简单问题上
  - **如果这还不行，根因不在Judge/采样，需要重新审视架构**
- 🔥 **用户再次反馈：仍然100%组无梯度，发现根本原因**
- 🎯 **用户诊断核心问题（Mode Collapse）：**
  - Max prob: 0.999988（几乎deterministic）
  - 熵: 0.018-0.055（灾难性低）
  - 100%组reward std=0（零梯度）
  - 去重失败：3次重试后仍Jaccard>0.75
  - EOS抑制器一直在阻止early stopping
  - 模型在生成1-3个token时就想停止
- 💡 **根本原因：Logits裁剪发生在temperature之前！**
  - SanityLogitsProcessor: `scores.clamp(-50, 50)`
  - Flow: raw_logits → clip(-50,50) → /temp → softmax
  - 即使temp=1.1，softmax(50/1.1)≈softmax(45.5)≈0.9999+
  - **Temperature根本没有生效！**
- ✅ **核选项修复（真正解决根因）：**
  - ✅ 禁用logits裁剪（-50,50 → -1000,1000），只防溢出不限制分布
  - ✅ Temperature提升到2.0（对抗Llama-3-Instruct高置信度）
  - ✅ 进一步放松采样（top_k=200, top_p=0.98, rep_penalty=1.3）
- ✅ Commit b812b25推送（NUCLEAR OPTION: Fix logits clipping）
- 📊 **预期效果（核选项）：**
  - Max prob应降至<0.95（现在0.999988）
  - 熵应升至>0.5（现在0.018-0.055）
  - 去重应成功（Jaccard<0.65）
  - Reward std应>0.05（现在0.000000）
  - 非零梯度组应>50%（现在0%）
  - **如果这还不行，问题在SFT阶段模板太强/LoRA太弱/需全量微调**

**2025-11-08 (Session 4 - 核选项验证 & 平衡调整):**
- 🎉 **核选项成功！熵完全恢复：**
  - ✅ Entropy: mean=3.7-5.1, min=2.2-5.0 (修复前: mean=0.033, min=0.018)
  - ✅ Logits clipping禁用 + Temperature=2.0成功对抗Instruct模型高置信度
  - ✅ Reward variance开始出现：Step3 F:std=0.280, Step4-5 F:std=0.700
- ⚠️ **新问题：100%截断率**
  - Step3-6几乎所有样本达到max_new_tokens=128硬约束
  - 原因：temp=2.0太高 + no_repeat_ngram_size=3太严 → 强制生成长回答
- ✅ **平衡调整（Commit d3648c8）：**
  - Temperature: 2.0 → 1.5（Entropy=4.7已足够，降温控制长度）
  - no_repeat_ngram_size: 3 → 0（禁用3-gram约束，太严格）
  - 保留presence_penalty=0.7和frequency_penalty=0.3
- ⚠️ **剩余问题：50%组仍零梯度**
  - Step1: 100%组无梯度 → Step2,4,6: 50%组无梯度（有改善但不够）
  - 从provider统计看：`template_detector=2/8`仍在触发
  - 推测：模型用不同表达方式说相同逃避内容（高熵但同义）
- 🔍 **添加调试日志（Commit 3d88b09）：**
  - 在Step 1-3打印触发template_detector的样本详情
  - 显示匹配的短语、prompt、response前段
  - 用于诊断：逃避短语变体？reasoning quality评分有效性？
- 📊 **用户精简日志（Commit f4fef4e）：**
  - 日志从~200行/step压缩到~10行/step
  - 保留核心警告：熵塌陷、零梯度、严重失衡
  - 删除：EOS Suppressor详情、串行生成日志、每样本Fairness诊断

**2025-11-08 (Session 5 - 深入诊断零梯度组):**
- 📊 **用户运行训练后反馈，诊断结果：**
  - ✅ 熵改善：mean=2.3-4.3（虽然min仍有0.028-0.037低值点）
  - ✅ 截断率降低：从100%降到50-75%（Temperature=1.5生效）
  - ⚠️ 仍有50%组零梯度（Step 1,2,3,5,6都是50%，Step 4是100%）
- 🔍 **Template Detector调试日志揭示：**
  - **所有触发的样本都是ambig！**（Context condition: ambig）
  - 匹配短语："does not provide sufficient information", "cannot be determined"
  - **这是正确行为！** Ambiguous样本本来就该答"cannot determine"
  - 代码给了-0.2轻微负分（不是-1.0重罚），符合设计
  - **结论：Template detector工作正常，不是零梯度根源**
- 🎯 **真正问题定位：bbq_rule给相同分数**
  - Provider统计：`{'template_detector': 1, 'bbq_rule': 3, 'halueval_rule': 4}`
  - Fairness 4个样本：1个ambig（template_detector），3个disambig（bbq_rule）
  - **这3个disambig样本走bbq_rule → reward std=0.000**
  - **推测：** 所有candidates选了正确答案，但reasoning quality评分未区分出差异
- 🔧 **添加零梯度组详细诊断（Commit 33d492a）：**
  - Step 1-3打印第一个零梯度组的4个candidates详情
  - 重新调用bbq_rule评估，显示每个candidate的reasoning quality分数
  - 将揭示：
    * 4个candidates是否选了相同答案（预期）
    * Reasoning quality分数是否有差异（期望0.3-1.0范围）
    * `_assess_reasoning_quality()`是否在工作
    * 还是4个candidates的reasoning完全相同？

**2025-11-08 (Session 5 续 - 🎯找到根本原因并修复):**
- 🔍 **零梯度组诊断结果分析：**
  - **所有零梯度组都是ambig样本！**
  - Step 1组0：4个candidates全是`Reward: -1.000`（context_condition: ambig）
  - Step 2组1：4个candidates全是`Reward: -1.000`（context_condition: ambig）
  - 但是代码Line 1748明明写的是返回-0.2，为什么变成-1.0？

- 🐛 **Bug #1：Reward映射逻辑错误（Line 2919）**
  ```python
  # 错误代码：
  r = max(0.0, min(1.0, float(r))) * 2 - 1
  ```
  - **假设：** judge返回[0, 1]，映射到[-1, 1]
  - **现实：** 所有judge都返回[-1, 1]！
  - **后果：** template_detector返回-0.2 → `max(0.0, -0.2)` = 0.0 → `0.0*2-1` = **-1.0**
  - **结果：** 所有负分都被截断并映射到-1.0 → std=0 → 零梯度！

- 🐛 **Bug #2：Ambig样本给固定分数（Line 1507）**
  - Template_detector给ambig样本-0.2（固定分）
  - 或bbq_rule给ambig样本1.0/-1.0（无reasoning质量差异）
  - 4个candidates都触发template短语 → 都是-0.2 → 映射后都是-1.0 → std=0

- ✅ **修复方案（Commit 2e08b17）：**

  **Fix #1: 移除错误的reward映射**
  ```python
  # Line 2919-2921
  r = r_obj.get("final", 0.0)
  # 【修复】直接使用judge返回的[-1, 1]分数，不做映射
  return float(np.clip(r, -config.REWARD_CLIP, config.REWARD_CLIP))
  ```

  **Fix #2: Ambig样本评估reasoning质量**
  ```python
  # Line 1741-1743: Ambig样本跳过template_detector
  if sample.task == "fairness" and context_condition == "ambig":
      pass  # 不拦截，继续走到bbq_rule

  # Line 1507-1537: Bbq_rule对ambig样本评估reasoning质量
  if chosen_answer == unknown_option:
      score = 1.0
      # 根据justification长度（8-50词）、重复度调整
      score = max(0.3, min(1.0, score))  # [0.3, 1.0]范围
  ```

- 🎯 **预期效果：**
  - Ambig样本：4个candidates得到差异化的分数（0.3-1.0）
  - Disambig样本：保留现有逻辑（正确答案 + reasoning质量）
  - 零梯度组比例应大幅下降（从50-100% → <20%）
  - Reward std应显著提升

**2025-11-08 (Session 5 续2 - 验证修复并发现HaluEval问题):**
- ✅ **Ambig样本修复验证成功！**
  - Step 2: `F: std=0.680` (之前是0.000)
  - Step 5: `F: std=0.770` (有明显分数差异)
  - Template detector仍触发ambig样本，但现在走bbq_rule得到差异化评分

- ❌ **发现新问题：Hallucination任务零梯度**
  - Step 1组1：4个candidates全是`Reward: 1.000` (Hallucination任务)
  - Step 3组1：4个candidates全是`Reward: 1.000` (Hallucination任务)
  - 回答内容：完全是胡说八道的hallucination
  ```
  "Answer: Good Will Hunting which made film appearance..."
  "Answer: Besides books isn '68 The Sweet Taste..."
  "Answer: Well, Leonard B Burnett won an Academy Award..."
  ```

- 🐛 **Bug #3：halueval_rule只检查格式，不检查内容质量**
  ```python
  # 旧评分逻辑：
  score = 0.5 + 0.3(有Evidence) + 0.2(长度>30) + 0.1(有Answer)
  # = 1.1 → clip到1.0
  ```
  - **问题：** 只要有`Answer:`、`Evidence:`和引号，无论内容是否正确，都拿1.0
  - **后果：** 4个格式正确的胡说 → 都是1.0 → std=0 → 零梯度

- ✅ **修复方案（Commit e9919dd）：**

  **添加Answer和Evidence质量差异化评分：**

  1. Evidence质量（不只是有无）：
     - 长度<5词：+0.1（太短）
     - 长度>50词：+0.2（太冗长）
     - 长度5-50词：+0.3（合理）

  2. Answer质量（不只是有无）：
     - 长度<3词：-0.2（太短）
     - 长度>30词：-0.1（太冗长）
     - 长度3-30词：+0.2（合理）
     - 重复度>50%：-0.2（重复严重）

  3. 整体长度：
     - 总长<15词：-0.2
     - 总长>80词：-0.1

- 🎯 **预期效果：**
  - 不同candidates即使都有格式，也会因长度、重复度等差异得到不同分数
  - Hallucination任务的reward std应>0.2
  - 零梯度组比例应<20%

**2025-11-08 (Session 6 - 🔥 使用Ground Truth评估内容质量):**
- 🔍 **调研HaluEval官方文档**
  - 发现数据集应包含knowledge/right_answer/hallucinated_answer字段
  - 当前adapter加载了这些字段，但只用于构建SFT target
  - 从未保存到meta → Judge无法访问！
- 🐛 **发现根本问题：Judge无法验证内容正确性**
  - 只检查格式（Answer/Evidence字段、长度）
  - 无法区分"瞎编但格式正确"和"内容正确"
  - 导致零梯度组（所有candidates瞎编但都拿高分）
- ✅ **实施CRITICAL FIX (Commit 92c2fc3):**

  **修复1: HaluEvalAdapter保存Ground Truth到meta**
  ```python
  # Line 1226-1232: qa子集
  meta.update({
      "knowledge": know,
      "right_answer": answer,
      "hallucinated_answer": hallucinated_answer,
      ...
  })

  # 同样修复dialogue, summarization, general子集
  ```

  **修复2: 新增_check_content_against_ground_truth()方法**
  - 检测口语化/瞎编开头（"yes there", "well maybe", "i think"）→ -0.3
  - 检测模糊泛泛描述（"good performance", "in general", "thrills"）→ -0.2
  - 检查Answer包含right_answer的关键词（长度>3）→ +0.3
  - 检查Evidence引用knowledge（n-gram匹配）→ +0.2
  - 检查是否更接近hallucinated_answer → -0.2
  - 适配qa/dialogue/summarization三个子集

  **修复3: 集成到_evaluate_halueval()**
  - Bonus分数范围：[-0.5, +0.5]
  - 最终分数仍clip到[-1.0, 1.0]

- 📊 **预期效果：**
  - ✅ 瞎编responses（格式正确但内容错）：0.3-0.5分
  - ✅ 正确responses（格式+内容都对）：0.8-1.0分
  - ✅ Reward std：0.000 → >0.2
  - ✅ 零梯度组：50% → <20%
- 💡 **重要意义：**
  - 这是基于HaluEval官方文档的标准做法
  - 比启发式规则更可靠（有ground truth支撑）
  - 直接解决零梯度根因（无法区分内容质量）

**待验证（下次训练）：**
- [ ] 前10步的实际观察结果（关注熵是否上升）
- [ ] 模型是否开始真正学习（不再锁死）
- [ ] Hallucination任务的reward std是否>0.2
- [ ] 零梯度组比例是否<20%
- [ ] 是否需要进一步调整超参（ENTROPY_COEF, beta, Temperature, KL目标）
- [ ] 最终训练效果和收敛情况

---

**文档结束。如有疑问，请参考trainer.py中的详细注释或重新阅读本文档的相关章节。**
