# 数据集分析发现与修复建议

## BBQ数据集分析

### 基本统计
```
总样本数: 58,492
子集数量: 11
每个子集: 432-15,960样本不等

分布: 所有子集都是精确的 50% ambig / 50% disambig
Context平均长度: 173-257字符
Unknown选项: answer_info.ans2 = ["Can't be determined", "unknown"]
```

### 关键发现
1. ✅ **50/50分布验证了Bug #-1诊断**: 模型容易学到"总是选Unknown"策略
2. ✅ **Label格式**: ambig样本label=2（Unknown选项），disambig样本label=0/1/2（正确答案）
3. ⚠️ **Context snippet过短**: 之前50字符仅占20-30%，已调整为120字符

### 已应用的修复
- ✅ Context snippet: 50 → 120字符 (trainer.py:1149)
- ✅ 自适应采样: 60% disambig + 40% ambig
- ✅ BBQ规则评估器: ambig必须选Unknown，disambig必须选正确答案

### 训练集规模
```
当前: N_BBQ_TRAIN = 1100 (每个子集100)
可用: 58,492样本
使用率: 1.9%

建议: 当前规模已足够，但可考虑增加到2000-3000（4-5%）
```

---

## HaluEval数据集分析

### 基本统计（已验证）
```
QA子集:            10,000样本，平均knowledge长度: 341字符
Dialogue子集:      10,000样本
Summarization子集: 10,000样本，平均document长度: 3297字符（max: 9252）
General子集:       4,507样本
总计:              34,507样本

文件大小:
- QA: 5.9 MB
- Dialogue: 6.7 MB
- Summarization: 44.8 MB
- General: 3.2 MB
```

### 各子集字段结构

#### 1. QA子集
```json
{
  "knowledge": str,              // 背景知识（平均341字符）
  "question": str,
  "right_answer": str,           // 短答案："Arthur's Magazine"
  "hallucinated_answer": str     // 错误答案："First for Women was started first."
}
```

**关键特征**:
- ✅ 有knowledge可以grounding（平均341字符）
- ✅ right_answer是短答案格式
- ✅ 有hallucinated版本可用于对比学习
- ⚠️ 之前snippet只取50字符（仅15%），已调整为150字符（44%）

#### 2. Dialogue子集
```json
{
  "knowledge": str,              // 格式类似QA（平均长度未统计，假设相似）
  "dialogue_history": str,
  "right_response": str,         // 完整句子
  "hallucinated_response": str
}
```

**关键特征**:
- ✅ 有knowledge（格式类似QA）
- ✅ right_response是完整句子（不是短答案）
- ⚠️ 已调整snippet为150字符（与QA一致）

#### 3. Summarization子集
```json
{
  "document": str,               // 平均3297字符，最大9252字符
  "right_summary": str,          // 平均310字符
  "hallucinated_summary": str    // 平均356字符
}
```

**关键特征**:
- ⚠️ Document很长（平均3297字符）
- ✅ 当前SUMM_MAX_DOC_CHARS=1000（约30%原文）合理
- ⚠️ 之前evidence snippet只取80字符（8%），已调整为200字符（20%）
- ✅ Summary长度合理（300+字符）

#### 4. General子集 ⚠️ 特殊
```json
{
  "ID": int,
  "user_query": str,
  "chatgpt_response": str,       // 平均736字符
  "hallucination": "yes"/"no",   // 标签！
  "hallucination_spans": list
}
```

**关键特征**:
- ❌ **没有knowledge字段！**
- ✅ 有hallucination二分类标签
- ⚠️ 当前trainer.py的处理可能不合理

### HaluEval在trainer.py中的处理

#### QA子集 (1210-1222)
```python
# ✅ 正确使用right_answer
answer = self._pick(it,'right_answer')
# 基于实际统计（平均341字符），调整为150字符（占44%）
know_snippet = know[:150] + "..." if len(know) > 150 else know
target = f"Answer: {answer}\nEvidence: \"{know_snippet}\""
```

**已修复**:
- ✅ Knowledge snippet: 50 → 150字符（基于实际平均长度341字符）

#### Dialogue子集 (1224-1236)
```python
# ✅ 正确使用right_response
response = self._pick(it,'right_response')
# 与QA保持一致，调整为150字符
know_snippet = know[:150] + "..." if len(know) > 150 else know
target = f"Answer: {response}\nEvidence: \"{know_snippet}\""
```

**已修复**:
- ✅ Knowledge snippet: 50 → 150字符（与QA一致）

#### Summarization子集 (1238-1251)
```python
# ✅ 正确使用right_summary
doc = doc[:1000] + "..." if len(doc) > 1000 else doc  # SUMM_MAX_DOC_CHARS
# 基于实际统计（平均3297，截断为1000），evidence取200字符（占20%）
doc_snippet = doc[:200] + "..." if len(doc) > 200 else doc
target = f"Summary: {gold}\nEvidence: \"{doc_snippet}\""
```

**已修复**:
- ✅ Evidence snippet: 80 → 200字符（基于截断后1000字符的20%）

#### General子集 (1253-1270) ✅ 已修复
```python
# ✅ 使用hallucination标签决定target
chatgpt_resp = self._pick(it,"chatgpt_response")
hallucination = self._pick(it,"hallucination","label")  # "yes"/"no"

if hallucination == "no":
    # 无hallucination，使用ChatGPT的回答（截断200字符）
    resp_truncated = chatgpt_resp[:200] + "..." if len(chatgpt_resp) > 200 else chatgpt_resp
    target = f"Answer: {resp_truncated}\nEvidence: \"Based on general knowledge\""
    meta.update({"has_knowledge":False, "has_hallucination":False})
else:
    # 有hallucination，教模型保守回答
    target = "Answer: I need more information to provide an accurate answer.\nEvidence: \"insufficient\""
    meta.update({"has_knowledge":False, "has_hallucination":True})
```

**已修复**:
- ✅ 使用hallucination标签区分有/无hallucination样本
- ✅ hallucination="no"时使用chatgpt_response（实际内容）
- ✅ hallucination="yes"时教模型保守回答
- ✅ 添加has_hallucination标志到meta，供评估器使用
- ✅ 4507个General样本现在得到有效利用

---

## 当前HaluEval评估器的问题

### 现有实现 (trainer.py:1498-1547)
```python
def _evaluate_halueval(self, sample: Sample, response: str):
    # 检查Evidence引用 (+0.3/-0.2)
    # 检查内容长度 (+0.2)
    # 检查占位符 (-0.5)
    # 检查格式 (+0.1)
    # 检查乱码 (-0.3)
```

**问题**:
1. ❌ **未使用meta中的right_answer/right_response/right_summary**
2. ❌ **无法处理General子集**（没有knowledge）
3. ❌ **分数范围小** (0.0-1.0)，不够discriminative

### 建议改进

#### 改进1: 使用right_answer做匹配检查
```python
if subset == "qa":
    right_answer = sample.meta.get("right_answer", "")
    if right_answer and right_answer.lower() in response.lower():
        score += 0.3  # 答案正确
    else:
        score -= 0.3  # 答案错误或缺失
```

#### 改进2: General子集特殊处理
```python
if subset == "general":
    # General没有knowledge，检查是否保守回答
    if "need more information" in response_lower or "cannot provide" in response_lower:
        score += 0.5  # 正确识别无grounding情况
    else:
        score -= 0.3  # 可能产生hallucination
```

---

## 训练集规模建议

### 当前配置
```python
N_BBQ_TRAIN = 1100   # 58,492样本的1.9%
N_HALU_TRAIN = 400   # 34,507样本的1.2%
```

### 可用资源
```
BBQ: 58,492样本
HaluEval: 34,507样本
```

### 建议调整
```python
# 如果计算资源充足
N_BBQ_TRAIN = 2200   # 每个子集200，约3.8%
N_HALU_TRAIN = 800   # 每个子集200，约2.3%

# 如果资源有限（保持当前）
N_BBQ_TRAIN = 1100   # 已足够
N_HALU_TRAIN = 400   # 已足够
```

**建议**: 先用当前规模验证修复效果，如果效果好再扩大

---

## 优先级修复清单

### 🔥 高优先级（必须修复）
1. ✅ **BBQ Context snippet**: 50→120字符 (已修复)
2. ✅ **BBQ规则评估器**: 已实现
3. ❌ **General子集处理**: 需要使用hallucination标签
4. ❌ **HaluEval评估器**: 需要使用right_answer做检查

### ⚠️ 中优先级（建议修复）
1. **HaluEval knowledge snippet**: 考虑从50字符增加到100-120字符
2. **Summarization evidence snippet**: 从80字符增加到150字符
3. **HaluEval答案匹配**: 添加right_answer/right_response的模糊匹配

### 💡 低优先级（可选优化）
1. **增加训练集规模**: 从1100/400增加到2200/800
2. **使用hallucinated版本**: 对比学习（需要额外设计）
3. **Reward scale自动调整**: 基于信号强度EMA动态调整

---

## 下一步行动

### 立即行动（修复General子集）
```python
# trainer.py HaluEvalAdapter._build_pair()第1251-1257行
# 修改为使用hallucination标签
```

### 测试验证
1. 修复后重新运行inspect_datasets.py（应该能正常显示HaluEval）
2. 运行小规模SFT训练（100 steps）验证无错误
3. 检查SFT输出是否包含真实引用

### 完整重训
1. 删除旧checkpoint
2. SFT从头训练
3. GRPO训练
4. 监控Entropy恢复情况
