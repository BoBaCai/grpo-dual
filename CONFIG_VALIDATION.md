# 配置验证清单（根据完整logging总结）

## 已验证 ✅

### 1. Chat Template一致性
**代码位置**: `apply_chat_template()` line 2054-2071

**SFT使用**:
```python
formatted_prompt = apply_chat_template(tokenizer, prompt, system_msg)
full_text = formatted_prompt + target
```

**GRPO使用**:
```python
formatted_prompts = [apply_chat_template(tokenizer, p, system_msg) for p in prompts]
```

**结论**: ✅ SFT和GRPO使用完全相同的chat template函数

---

### 2. SFT Labels掩码
**代码位置**: `tokenize_sft_pair()` line 2207-2234

**实现**:
```python
formatted_prompt = apply_chat_template(tokenizer, prompt, system_msg)
# 例: "...<|start_header_id|>assistant<|end_header_id|>\n\n"

prompt_ids = tokenizer(formatted_prompt, ...)
full_ids = tokenizer(formatted_prompt + target, ...)

labels = full_ids["input_ids"].clone()
prompt_len = prompt_ids["input_ids"].shape[1]
labels[:, :prompt_len] = -100  # Mask prompt
```

**Causal LM Loss机制**:
```
logits[i] 预测 labels[i+1]
shift_logits = logits[:, :-1]
shift_labels = labels[:, 1:]
loss只计算labels != -100的位置
```

**验证**:
- Prompt部分: labels[:, :prompt_len] = -100 → 不计算loss ✓
- Response部分: labels[:, prompt_len:] = target_ids → 计算loss ✓
- 边界: logits[prompt_len-1]预测labels[prompt_len]（第一个response token）✓

**结论**: ✅ 无off-by-one错误

---

### 3. 多终止符配置
**代码位置**: `get_eos_token_ids()` line 2049-2091

**LLaMA-3-Instruct标准配置**:
```
<|end_of_text|>  128001  (文档结束)
<|eot_id|>       128009  (对话轮次结束)
```

**实现**:
```python
eos_ids = []
if '<|end_of_text|>' in vocab:
    eos_ids.append(128001)
if '<|eot_id|>' in vocab:
    eos_ids.append(128009)
# 返回: [128001, 128009]
```

**使用**:
```python
model.generate(..., eos_token_id=eos_ids)  # 两者都能终止生成
```

**结论**: ✅ 正确配置多终止符

---

### 4. Padding配置
**代码位置**: line 2118-2120

**实现**:
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # pad_id = 128001
tokenizer.padding_side = "left"
```

**LLaMA-3标准**: 没有专门的pad token，使用eos作为padding是**标准做法**

**影响**:
- ✅ attention_mask正确标记padding为0
- ✅ Model不会attend to padding
- ⚠️ 使用`skip_special_tokens=False`显示时，padding显示为`<|end_of_text|>`
  - 这是**显示问题**，不影响训练
  - 短prompt可能显示几十个padding tokens

**结论**: ✅ 配置正确，显示混淆是预期行为

---

### 5. 边界计算（已修复）
**Bug #4 & #5修复**:
- `generate_candidates_batch`: 使用`original_input_len`而非`src_lens`
- `_tokenize_concat`: `resp_start = T - resp_len`（LEFT padding下正确）

**结论**: ✅ 所有边界计算已修复

---

## 需要运行时验证 ⚠️

### 1. Tokenizer特殊Token检查

**在训练开始时应打印**:
```
pad_token_id: 128001
eos_token_id: 128001
bos_token_id: 128000
eot_token_id: 128009
```

**检查**:
- [ ] pad_token_id与eos_token_id相同 → **正常**（LLaMA-3标准）
- [ ] pad_token_id与eot_token_id不同 → **必须**（否则padding会被当成轮次结束）
- [ ] 无重复ID（除了pad=eos）

**如果pad_token_id=eot_token_id (128009)**:
- ❌ **严重错误** - padding会被当成对话结束
- 修复: `tokenizer.pad_token_id = tokenizer.eos_token_id  # 应该是128001，不是128009`

---

### 2. Chat Template输出格式

**正常格式**（`add_generation_prompt=True`）:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful, accurate, and unbiased assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

[prompt]<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

**检查**:
- [ ] 包含2个`<|eot_id|>`（system结束 + user结束）
- [ ] 末尾是`assistant<|end_header_id|>\n\n`
- [ ] 不应该有3个或更多`<|eot_id|>`

**异常情况**:
- 如果formatted_prompt包含>2个`<|eot_id|>` → 模板被重复应用
- 如果末尾不是`\n\n` → 可能影响response拼接

---

### 3. 边界诊断输出（Step 0-1）

**正常输出**:
```
Token统计:
  Prompt: 2个<|eot_id|> tokens, 50个padding  ← padding多是正常的（短prompt）
  Response: 1个<|eot_id|> tokens              ← 回答结束
  pad_token_id=128001, eot_token_id=128009   ← 不相等✓
```

**异常输出需要立即停止**:
```
Token统计:
  Prompt: 2个<|eot_id|> tokens, 0个padding
  Response: 35个<|eot_id|> tokens             ← 模型在生成中反复输出eot
⚠️ 异常: Response包含35个<|eot_id|> tokens（正常≤1）
```

或：
```
pad_token_id=128009, eot_token_id=128009      ← 相等！错误配置
⚠️ 警告: pad_token_id == eot_token_id (128009)
Prompt: 50个<|eot_id|> tokens, 50个padding   ← 所有padding都是eot!
```

---

### 4. 文本层Stop Sequence检查

**搜索代码中是否有**:
```python
stop_sequences=["<|eot_id|>", ...]  # 不应该存在
stop=["<|eot_id|>"]                 # 不应该存在
```

**正确做法**:
- ✅ 只使用`eos_token_id=[128001, 128009]`（token级别）
- ❌ 不要使用文本级别的stop参数

**代码检查**:
```bash
grep -n "stop_sequences\|stop=" trainer.py
```

如果找到 → 需要移除

---

## 总结：logging异常的解释

### 如果看到`<|eot_id|>`爆量（36/79/148）

**可能原因**:

1. **Padding显示（正常）**:
   - `pad_token_id = eos_token_id = 128001`
   - 短prompt用LEFT padding填充
   - Padding在`skip_special_tokens=False`时显示为`<|end_of_text|>`
   - **这不影响训练**

2. **配置错误（严重）**:
   - `pad_token_id = eot_token_id = 128009`
   - 所有padding被当成轮次结束
   - 模型训练错误

3. **模型生成错误（严重）**:
   - Response中实际生成了大量`<|eot_id|>` tokens
   - Bug #4/#5修复后应该消失

**验证方法**:
- 查看新诊断输出的"Token统计"
- 区分prompt vs response的eot数量
- 检查padding数量

---

### 如果看到Response包含system/user头

**原因**: Bug #4（response边界错误）
**状态**: ✅ 已修复
**验证**: 重训后检查诊断输出

---

### 如果看到熵≈0

**可能原因**:
1. Bug #5（comp_mask错误） → ✅ 已修复
2. SFT数据太短/模板化 → 需要检查SFT数据分布
3. Temperature太低 → 已调整为1.2

**验证**: 重训后检查前10步的熵统计

---

### 如果看到Fairness reward std=0

**原因**: 小批次中只有1个Fairness样本，或所有样本得分相同
**状态**: 代码已有检测，会跳过scale更新
**不影响**: 训练继续，只是该batch的reward scale建议无效

---

## 最终验证步骤

重训前：
1. [ ] 运行单元测试（如果有）
2. [ ] 手动检查tokenizer配置打印

重训Step 0-1：
3. [ ] 检查Token统计：prompt=2 eot, response≤1 eot
4. [ ] 检查pad_token_id ≠ eot_token_id
5. [ ] 检查Response不含`<|start_header_id|>`
6. [ ] 检查熵>0.5

如果全部通过 → 继续训练
如果任何一项失败 → 立即停止，检查配置
