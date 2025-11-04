# GRPO Training - Critical Bug Fixes & Retraining Required

## 状态：需要从头完全重训（SFT + GRPO）

---

## 已修复的Critical Bug（5个）

### Bug #1: SFT与RL使用不同的模板格式
- **位置**: `tokenize_sft_pair()` line 2165
- **问题**: SFT用`"\n\n"`简单拼接，RL用完整chat template
- **影响**: 模型从未学过chat template格式
- **修复**: SFT改为使用相同chat template

### Bug #2: Penalty应用于整个序列（含prompt）
- **位置**: `PresencePenaltyProcessor`, `FrequencyPenaltyProcessor` line 1713-1749
- **问题**: Prompt中的"the"出现20次→-8.0惩罚；EOS从未在prompt→0惩罚→相对提升3000x
- **影响**: 导致1-token生成
- **修复**: 只对response部分应用penalty

### Bug #3: KL/loss使用错误的prompt格式
- **位置**: `generate_candidates_batch()` line 1967, training loop line 2478
- **问题**: Generate用formatted_prompt（带template），KL/loss用original_prompt（无template）
- **影响**: 梯度计算在完全不同的token序列上
- **修复**: 返回并使用formatted_prompts

### Bug #4: LEFT padding下response提取错误
- **位置**: `generate_candidates_batch()` line 1863-1879
- **问题**:
```python
# 错误：用"计数"当"位置"
src_lens = (inputs["input_ids"] != pad).sum(dim=1)  # 3个非pad token
response = out[i, src_lens[i]:]  # 从位置3开始 → 包含prompt!
```
- **影响**: Judge看到的response包含prompt内容，所有reward污染
- **修复**:
```python
original_input_len = inputs["input_ids"].shape[1]  # 绝对长度
response = out[i, original_input_len:]  # 正确边界
```

### Bug #5: LEFT padding下comp_mask计算错误
- **位置**: `_tokenize_concat()` line 2230-2252
- **问题**:
```python
# 错误：假设RIGHT padding
resp_start = valid_len - resp_len  # 标记了prompt而非response
```
- **影响**: **Loss和KL在padding/prompt位置计算，所有梯度错误**
- **修复**:
```python
resp_start = T - resp_len  # LEFT padding下response在序列末尾
comp_end = T - 1
```

---

## 为什么必须重训

| Bug | 影响范围 | 需要重训 |
|-----|----------|----------|
| #1 | SFT学错了格式 | SFT + GRPO |
| #2 | 所有reward信号 | GRPO |
| #3 | 所有KL/loss梯度 | GRPO |
| #4 | 所有reward信号 | GRPO |
| #5 | 所有loss梯度（最严重）| GRPO |

**之前的checkpoint全部作废。**

---

## 重训前必须验证（P0）

添加单测到`test_padding.py`：
```python
# 1. 测试response提取
def test_response_extraction():
    # Mock left-padded generation
    inputs = tokenizer(["short", "longer prompt"], padding=True, return_tensors="pt")
    original_input_len = inputs["input_ids"].shape[1]

    # Simulate generation
    fake_response = torch.tensor([[100, 200], [100, 200]])
    out = torch.cat([inputs["input_ids"], fake_response], dim=1)

    # Extract
    response = out[:, original_input_len:]

    # Verify
    assert response.shape[1] == 2  # Only generated tokens
    assert not any(response == tokenizer.pad_token_id)  # No padding

# 2. 测试comp_mask
def test_comp_mask():
    prompts = ["Hello"]
    responses = ["World"]
    lengths = [1]  # "World" = 1 token

    full, comp_mask = _tokenize_concat(tokenizer, prompts, responses, lengths, device)

    T = full["input_ids"].shape[1]
    # Response在最后1个位置，logits最后预测位置是T-2
    assert comp_mask[0, T-2] == 1.0  # 应该标记
    assert comp_mask[0, :T-2].sum() >= 0  # 前面可能是prompt
```

**运行确认通过后再启动训练。**

---

## 首次训练监控（前2 steps）

### 关键诊断日志（已自动打印）

#### ✅ 边界正确
```
样本 0:
  Response部分(前120字符,含special): [不应该包含 <|start_header_id|> 等]
  边界附近Token: [prompt末尾应该是 assistant<|end_header_id|>\n\n]
```

#### ✅ 长度合理
```
生成长度: Fairness avg=XX, Hallucination avg=YY
```
预期：15-96 tokens（不应该是1-2或200+）

#### ✅ 熵非零
```
Fairness样本 0: entropy=X.XXX
```
预期：>0.5（如果≈0说明仍有问题）

#### ✅ EOT token数量正常
```
Token统计:
  Prompt: X个<|eot_id|> tokens, Y个padding
  Response: Z个<|eot_id|> tokens
```
预期：
- Prompt: 2个（system结束 + user结束）+ padding（可能很多，因为pad_token_id=eos_token_id）
- Response: ≤1个（回答结束）

如果看到：
- `pad_token_id == eot_token_id` → **正常**（LLaMA-3标准配置）
- Prompt有50+个eot + 50+个padding → **正常**（短prompt的LEFT padding显示）
- Response有10+个eot → **异常**（模型在回答中反复生成eot）

### ⚠️ 异常立即停止训练

如果看到：
- `⚠️ 异常: Response开头似乎包含chat header`
- `⚠️ 异常: Full包含36个<|eot_id|>`
- `熵=0.000`且大面积出现

→ **停止，检查代码**

---

## 采样参数（已调整，观察效果）

```python
MIN_NEW_TOKENS_TRAIN = 15      # 延迟EOS释放
TEMPERATURE_TRAIN = 1.2        # 对抗极尖分布
TOP_P_TRAIN = 0.95             # 放宽探索
PRESENCE_PENALTY = 0.3         # 降低（配合scope修复）
FREQUENCY_PENALTY = 0.2        # 降低（配合scope修复）
```

如果仍然熵崩塌，考虑：
- 提高temperature到1.5
- 降低entropy_coef（当前0.2可能过高）

---

## Commit记录

```bash
git log --oneline HEAD~7..HEAD
a07af9a CRITICAL: fix comp_mask calculation for left padding
fc99b9f CRITICAL: fix left padding response extraction boundary error
86a5902 CRITICAL: fix _tokenize_concat using wrong prompt format for KL/loss
6810389 CRITICAL: fix SFT→RL template inconsistency causing boundary corruption
be52b55 fix: address structural root causes of entropy collapse
```

---

## 文件修改汇总

### `src/grpo/trainer.py`

#### Line 1713-1749: Penalty Processors
- 添加`prompt_len`跟踪
- 只对response应用penalty

#### Line 1811-1967: generate_candidates_batch
- 返回`formatted_prompts`
- 使用`original_input_len`提取response
- 详细边界诊断日志（step<2时）

#### Line 2165-2207: tokenize_sft_pair
- 使用`apply_chat_template`（与GRPO一致）

#### Line 2230-2254: _tokenize_concat
- 修复LEFT padding下的`comp_mask`计算
- `resp_start = T - resp_len`

#### Line 2473-2491: Training Loop
- 使用`formatted_prompts`而非原始prompts

---

## 已知正确的配置

- ✅ Tokenizer: `padding_side = "left"`
- ✅ EOS tokens: `[eos_token_id, eot_token_id]` (128001, 128009)
- ✅ Chat template: LLaMA-3-Instruct格式
- ✅ Attention mask: 自动生成（正确）
- ✅ KL formula: DeepSeekMath Eq.4前向KL
- ✅ Reference model: 冻结，与policy相同dtype/device

---

## 启动训练命令

```bash
cd /home/user/grpo-dual
python src/grpo/trainer.py

# 首次训练密切关注step 0-1的边界诊断输出
# 确认无异常后可继续
```

---

## 问题排查

### 如果仍然看到1-token生成
1. 检查EOSSuppressionProcessor日志：EOS是否真的被设置为-inf
2. 检查边界诊断：response是否包含prompt

### 如果仍然熵崩塌
1. 可能是reward设计问题（偏好短答案）
2. 可能是SFT数据本身就短
3. 尝试提高temperature或降低KL penalty (β)

### 如果reward信号全是0
1. 检查judge是否正常工作
2. 检查batch中是否包含两个任务（Fairness和Hallucination）

---

**联系人**: 原开发者已离开，有问题查看`/tmp/CRITICAL_AUDIT_FINAL_REPORT.md`详细分析
