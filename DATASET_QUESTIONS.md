# 数据集关键问题清单

## BBQ数据集（11个子集）

### 问题1: 每个子集的样本数量和ambig/disambig分布
**为什么需要**: 验证自适应采样策略是否合理，确认是否会出现数据不足

请运行：
```bash
cd /workspace/data/bbq
for f in *.jsonl; do
  echo "=== $f ==="
  total=$(wc -l < "$f")
  ambig=$(grep -c '"context_condition":"ambig"' "$f" || echo 0)
  disambig=$((total - ambig))
  echo "Total: $total"
  echo "Ambig: $ambig ($((ambig * 100 / total))%)"
  echo "Disambig: $disambig ($((disambig * 100 / total))%)"
  echo ""
done
```

**关键指标**:
- 每个子集总量是否都>100？（当前N_BBQ_TRAIN=1100，per_cat=100）
- Disambig比例是否都>=50%？
- 是否有子集ambig比例>50%（可能需要特殊处理）

---

### 问题2: BBQ的label分布
**为什么需要**: 确认label=-1的情况有多少，这些会fallback到ambiguous处理

请从一个子集抽样检查：
```bash
head -20 Age.jsonl | jq -r '{context_condition, label, ans0, ans1, ans2}' | head -60
```

**关键问题**:
- Disambig样本的label是否总是0/1/2？
- Ambig样本的label是什么？（-1还是也有0/1/2？）
- answer_info里的"unknown"字段格式是什么？

---

### 问题3: BBQ context长度
**为什么需要**: 当前取context[:50]作为snippet，需要确认是否合理

请运行：
```bash
head -100 Age.jsonl | jq -r '.context' | awk '{print length($0)}' | sort -n | tail -20
```

**关键指标**:
- Context平均长度？
- 最短/最长？
- 50字符是否足够覆盖关键信息？

---

## HaluEval数据集（4个子集）

### 问题4: 各子集样本数量
**为什么需要**: 确认当前N_HALU_TRAIN=400，per_subset=100是否合理

请运行：
```bash
cd /workspace/data/halueval
for f in *.json; do
  echo "=== $f ==="
  jq '. | length' "$f"
done
```

**关键问题**:
- 每个子集是否都>=100样本？
- summarization有40MB，样本数是多少？会不会太多导致采样偏斜？

---

### 问题5: HaluEval数据格式（最关键！）
**为什么需要**: 确定meta字段内容，验证我的评估器实现

#### QA子集格式
请运行：
```bash
head -3 qa_data.json | jq '.[0] | keys'
head -3 qa_data.json | jq '.[0] | {question, knowledge, right_answer, hallucinated_answer}'
```

**关键问题**:
- `right_answer`格式是什么？短答案（"Paris"）还是完整句子（"The capital is Paris."）？
- 是否有`hallucinated_answer`字段？
- knowledge平均长度？

#### Dialogue子集格式
```bash
jq '.[0] | keys' dialogue_data.json
jq '.[0] | {dialogue_history, knowledge, right_response}' dialogue_data.json | head -50
```

**关键问题**:
- `right_response`格式？
- dialogue_history长度？

#### Summarization子集格式
```bash
jq '.[0] | keys' summarization_data.json
jq '.[0] | {document, right_summary, hallucinated_summary} | {doc_len: (.document | length), summary_len: (.right_summary | length)}' summarization_data.json | head -10
```

**关键问题**:
- document平均长度？（如果太长，当前SUMM_MAX_DOC_CHARS可能需要调整）
- right_summary vs hallucinated_summary格式？
- 是否所有样本都有hallucinated_summary？

#### General子集格式
```bash
jq '.[0] | keys' general_data.json
jq '.[0]' general_data.json | head -50
```

**关键问题**（最重要！）:
- **是否真的没有knowledge字段？**
- 只有hallucinated/not_hallucinated标签？
- 如何生成prompt？如何评估？

---

### 问题6: HaluEval的right_answer准确性匹配
**为什么需要**: 确定是否需要精确匹配检查

请抽样看几个qa样本的question+knowledge+right_answer：
```bash
jq '.[5:10][] | {q: .question, k: (.knowledge[:100] + "..."), ans: .right_answer}' qa_data.json
```

**关键问题**:
- right_answer是否可以通过简单的substring match验证？
- 还是需要语义匹配（用embedding或NLI）？
- 例如：question="What is the capital of France?", right_answer可能是"Paris"还是"The capital of France is Paris"？

---

## 额外问题

### 问题7: 训练集大小合理性
当前配置：
- N_BBQ_TRAIN = 1100（11个子集 × 100）
- N_HALU_TRAIN = 400（4个子集 × 100）

**问题**:
- BBQ总样本量？如果每个子集有10000样本，1100太少
- HaluEval总样本量？
- 是否需要增加训练集大小？

### 问题8: 实际训练日志的问题
从你之前的日志看到：
```
样本 #4-7: Answer: C, Justification: The context does not provide sufficient information...
```

**问题**:
- 这是哪个BBQ子集？
- 该样本的实际context_condition是ambig还是disambig？
- 如果是disambig，正确答案应该是什么（label=?）？
- 可以提供一个完整的样本内容吗（context+question+options+label）？

这能帮我验证BBQ reward评估器是否会正确工作。
