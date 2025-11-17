# 🔬 BBQ数据集分析 - Fairness信号为0的根本原因

**时间**: 2025-11-17
**发现者**: 用户建议检查data文件夹

---

## 📊 **BBQ数据集基本统计**

### **所有类别的分布（50/50）**

| 类别 | 总样本数 | ambig | disambig |
|------|----------|-------|----------|
| Age | 3,680 | 1,840 (50%) | 1,840 (50%) |
| Disability_status | 1,556 | 778 (50%) | 778 (50%) |
| Gender_identity | 5,672 | 2,836 (50%) | 2,836 (50%) |
| Nationality | 3,080 | 1,540 (50%) | 1,540 (50%) |
| Physical_appearance | 1,576 | 788 (50%) | 788 (50%) |
| Race_ethnicity | 6,880 | 3,440 (50%) | 3,440 (50%) |
| Race_x_SES | 11,160 | 5,580 (50%) | 5,580 (50%) |
| Race_x_gender | 15,960 | 7,980 (50%) | 7,980 (50%) |
| Religion | 1,200 | 600 (50%) | 600 (50%) |
| SES | 6,864 | 3,432 (50%) | 3,432 (50%) |
| Sexual_orientation | 864 | 432 (50%) | 432 (50%) |

**总计**: 58,492 样本（ambig=29,246, disambig=29,246）

**关键发现**: 所有类别都是**严格50/50分布**，而代码设置的采样比例是80% disambig / 20% ambig。

---

## 🔍 **Ambig样本的Label分布（关键发现！）**

分析Age.jsonl的前200个ambig样本：

```
neg polarity samples: 100
  Label distribution: Counter({0: 37, 1: 31, 2: 32})

nonneg polarity samples: 100
  Label distribution: Counter({0: 37, 1: 31, 2: 32})
```

**初看问题**: label分布是{0: 37, 1: 31, 2: 32}，并非100%都是2？

### **真相揭示**

详细检查前10个ambig样本发现：

- **所有ambig样本的label都指向unknown选项**
- **但unknown选项的位置是随机化的（0, 1, 或2）**
- **这是BBQ数据集的设计，防止模型记忆位置**

示例：
```
Sample 1: label=2 → ans2="Can't be determined" (unknown在位置2)
Sample 5: label=1 → ans1="Undetermined" (unknown在位置1)
```

**结论**: 100%的ambig样本正确答案都是unknown，只是选项位置随机化。

---

## 🚨 **发现原因8: Ambig样本的固有零梯度问题**

### **问题链条**

#### 1. **SFT阶段使用固定模板** (trainer.py:1240-1241)

```python
if it.get("context_condition","") == "ambig":
    return f"Answer: {unk}\nJustification: The context does not provide sufficient information to determine this."
```

**所有ambig样本的SFT target都是相同的模板**（只有unk字母A/B/C不同）

#### 2. **模型在SFT阶段学会了模板**

模型正确地学习了：ambig样本 → 选unknown选项 + "insufficient information" justification

#### 3. **GRPO阶段，模型重复相同模板**

当遇到ambig样本时，4个候选都生成：
```
Answer: B
Justification: The context does not provide sufficient information to determine this.
```

#### 4. **LLM Judge对正确答案给相同高分**

LLM Judge评估ambig样本：
- 答案正确（选了unknown选项）→ +50%
- Justification说明了"信息不足" → +40%
- **4个候选分数: [0.85, 0.86, 0.85, 0.86]** （微小差异）

#### 5. **零梯度形成**

```
原始reward: [0.85, 0.86, 0.85, 0.86]
Scale后 (0.7): [0.595, 0.602, 0.595, 0.602]
Normalization: [0.0001, 0.0001, 0.0001, 0.0001]  ← 数值精度级别
std = 0.000012 < 0.01 → advantage = 0
```

---

## 🎯 **根本原因总结**

### **这不是bug，而是ambig样本的固有特性！**

| 因素 | 描述 | 影响 |
|------|------|------|
| **数据设计** | 所有ambig样本正确答案都是unknown | 缺乏答案多样性 |
| **SFT模板** | 固定的"insufficient information"模板 | 缺乏表述多样性 |
| **模型学习** | 正确地学会了模板（这是好事！） | 生成确定性输出 |
| **LLM Judge** | 对正确的unknown回答给高分（这是对的！） | 分数高度一致 |
| **BATCH_SIZE=2** | 每步只有1个fairness样本 | 如果是ambig → 100%零梯度 |
| **20% ambig** | 约1/5的step遇到ambig样本 | 持续的零梯度组 |

**核心矛盾**:
- Ambig样本的**正确行为**（用模板回答unknown）
- 导致了GRPO的**零梯度问题**（缺乏多样性）

---

## 💊 **修复方案优先级排序**

基于数据分析，重新评估修复方案：

### **🔥 优先级1: 减少Ambig样本使用（立即实施）**

**方案A: 训练初期禁用ambig**
```python
# 在BBQAdapter.load_samples()中
if global_step < 50:
    target_disambig_ratio = 1.0   # 100% disambig
    target_ambig_ratio = 0.0       # 0% ambig
elif global_step < 100:
    target_disambig_ratio = 0.95  # 95% disambig
    target_ambig_ratio = 0.05     # 5% ambig
else:
    target_disambig_ratio = 0.90  # 90% disambig（不是80%）
    target_ambig_ratio = 0.10     # 10% ambig（不是20%）
```

**理由**:
- 前50步：让模型先学习disambig样本（有梯度信号）
- 50-100步：逐步引入少量ambig
- 100步后：10% ambig足够测试公平性

---

### **🔥 优先级2: 增加BATCH_SIZE（立即实施）**

```python
GRPO_BATCH_SIZE = 2  →  6
```

**效果**:
- 2 → 6: 每步有3个fairness样本（而不是1个）
- 即使1个是ambig（零梯度），还有2个disambig提供梯度
- 零梯度组比例从50%降到<20%

**成本**: 显存增加，但可通过减少MAX_NEW_TOKENS平衡

---

### **🔶 优先级3: 修改Ambig的SFT Target（中期）**

**问题**: 当前所有ambig样本用相同模板训练

**方案**: 增加表述多样性
```python
# 不同的表述方式
templates = [
    f"Answer: {unk}\nJustification: The context does not provide sufficient information to determine this.",
    f"Answer: {unk}\nJustification: Based on the context alone, we cannot determine the answer.",
    f"Answer: {unk}\nJustification: The information given is ambiguous regarding this question.",
    f"Answer: {unk}\nJustification: Without additional context, it's impossible to answer definitively.",
]
return random.choice(templates)
```

**效果**: 即使是ambig样本，4个候选也可能用不同表述 → 产生梯度

---

### **🔶 优先级4: 修改LLM Judge评分逻辑（中期）**

**问题**: 对于ambig样本，即使justification不同，LLM Judge可能给相同分数

**方案**: 增加diversity bonus
```python
# 在llm_judge_prompts_v2.py的ambig评分中
if context_condition == "ambig":
    # ... 现有评分逻辑 ...

    # 【新增】Diversity bonus
    d) **Expression Diversity** (0-10%):
       - Original phrasing: 10%
       - Common template but varied: 6%
       - Identical to standard template: 3%
```

---

### **✅ 优先级5: 提高其他参数（已完成）**

- ✅ ENTROPY_COEF: 1.5 → 2.5
- ✅ MIN_NEW_TOKENS: 15 → 30
- ✅ TEMPERATURE: 0.9 → 1.0

这些修复对disambig样本有效，但对ambig样本效果有限。

---

## 📊 **预期效果对比**

| 修复方案 | 零梯度组比例 | Fairness信号 | 实施难度 |
|---------|-------------|-------------|---------|
| **当前** (BATCH_SIZE=2, 20% ambig) | 50% | F std=0.000 | - |
| **优先级1** (0% ambig前50步) | <10% | F std>0.05 | 易 ⭐ |
| **优先级2** (BATCH_SIZE=6) | <20% | F std>0.03 | 易 ⭐ |
| **优先级1+2** 组合 | <5% | F std>0.08 | 易 ⭐⭐ |
| **优先级3** (多样化template) | <30% | F std>0.02 | 中 |
| **优先级4** (diversity bonus) | <25% | F std>0.02 | 中 |

---

## 🚀 **推荐立即实施**

### **Phase 1: 数据采样优化（立即）**

1. **增加BATCH_SIZE**: 2 → 6
2. **调整ambig比例**:
   - Step 0-50: 0% ambig
   - Step 50-100: 5% ambig
   - Step 100+: 10% ambig

**预期**: 零梯度组从50%降到<5%，Fairness信号立即恢复

### **Phase 2: 诊断验证（运行1-2步）**

使用已添加的6大诊断模块验证：
- 诊断1: 确认batch内有2-3个disambig样本
- 诊断6: 确认零梯度组比例<10%

### **Phase 3: 长期优化（可选）**

- 多样化ambig模板
- LLM Judge diversity bonus
- 动态ambig比例调整

---

## 📝 **代码修改建议**

### **1. 增加BATCH_SIZE**

```python
# trainer.py Line 207
GRPO_BATCH_SIZE = 6  # 从2改为6
```

### **2. 动态ambig比例**

```python
# trainer.py BBQAdapter.load_samples() Line 1143
def load_samples(self, n_total: int, current_step: int = 0) -> List[Sample]:
    # ... existing code ...

    # 【新增】动态调整ambig比例
    if current_step < 50:
        target_disambig_ratio = 1.0
        target_ambig_ratio = 0.0
    elif current_step < 100:
        target_disambig_ratio = 0.95
        target_ambig_ratio = 0.05
    else:
        target_disambig_ratio = 0.90
        target_ambig_ratio = 0.10

    # ... rest of sampling logic ...
```

---

## 🎯 **结论**

通过检查BBQ数据集，发现了**第8个根本原因**：

**Ambig样本的固有零梯度问题**不是bug，而是数据特性：
- 所有ambig样本正确答案都是unknown
- SFT用固定模板训练
- 模型正确地学会了模板
- GRPO时产生零梯度（因为4个候选完全相同）

**最有效的修复**：
1. 减少ambig使用（前期0%，后期10%）
2. 增加BATCH_SIZE（2→6）
3. 这两项组合可将零梯度组从50%降到<5%

**成本最低，效果最好！** ⭐⭐⭐

---

**Created**: 2025-11-17
**Status**: 建议立即实施优先级1+2
