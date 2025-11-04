# 🚀 Entropy 崩溃快速修复指南

## 🐛 发现的 Bug

**严重bug：** 完整模式（Reward-only CAGrad）中，entropy 梯度符号反了，导致模型被训练成**降低** entropy 而不是**提高** entropy！

详细分析见：`BUG_ANALYSIS_ENTROPY_GRADIENT.md`

---

## ✅ 快速修复（1分钟测试）

### 方案 A：启用 LOW_MEMORY_MODE（推荐先测试）

**只需修改 1 行代码！**

**文件：** `grpo-dual/src/grpo/trainer.py`
**位置：** 第 286 行

```python
# 修改前
LOW_MEMORY_MODE = False  # True=简化为2次反传；False=完整4次反传

# 修改后
LOW_MEMORY_MODE = True  # ✅ 启用LOW_MEMORY_MODE，entropy梯度是正确的
```

**原理：**
- LOW_MEMORY_MODE 的代码逻辑是正确的，entropy 符号没问题
- 完整模式（当前默认）的 entropy 梯度符号反了

**优点：**
- ✅ 立即生效，只改1行
- ✅ Entropy 符号正确
- ✅ 可以快速验证是否是这个 bug 导致的崩溃

**缺点：**
- ⚠️ 牺牲了一些 CAGrad 的多目标优化优势
- ⚠️ 2次反传而不是4次（显存更友好，但 β 可解释性略微下降）

---

### 测试步骤

1. **修改代码：**
   ```bash
   cd /path/to/grpo-dual
   # 编辑 grpo-dual/src/grpo/trainer.py 第 286 行
   # LOW_MEMORY_MODE = False → True
   ```

2. **重新开始 GRPO 训练：**
   ```bash
   # 可选：清空旧的 GRPO checkpoint
   rm -rf checkpoints/grpo_*

   # 重新训练
   python -m src.grpo.trainer
   ```

3. **观察前 5-10 步的日志：**

   **期望看到：**
   ```
   [Step X Logits] Max logit=20-25, Gap=3-5, Top5=[0.70-0.90, ...]
   [Fairness诊断] Entropy(mean)=0.5-1.5  ✅ 恢复！
   ```

   **如果仍然：**
   ```
   Max logit=30+, Gap=7-10, Top5=[0.999+, ...]
   Entropy=0.005-0.05  ❌ 没恢复
   ```
   → 说明还有其他问题，参考 `deep_entropy_diagnosis.py` 继续诊断

---

### 判断标准

| 指标 | Bug修复前 | Bug修复后（预期） |
|------|-----------|-------------------|
| **Entropy mean** | 0.005-0.033 | **0.5-1.5** |
| **Max prob** | 99.9%+ | **60-85%** |
| **Logit gap** | 7-10 | **3-5** |
| **EOS Suppressor 触发率** | 100% (8/8) | **30-50%** |

**如果 Entropy 恢复到 > 0.3：** ✅ Bug 确认！这就是根本原因！

---

## 🔧 完整修复（方案 B）

如果方案 A 验证成功，可以应用完整修复（保留 CAGrad 优势）：

**文件：** `grpo-dual/src/grpo/trainer.py`
**位置：** 第 3172、3179 行

```python
# 修复前（第 3172 行）
entropy_loss_f = -sample_entropy[task_mask_f].mean() / config.GRADIENT_ACCUMULATION_STEPS

# 修复后
entropy_loss_f = sample_entropy[task_mask_f].mean() / config.GRADIENT_ACCUMULATION_STEPS  # 去掉负号

# 同样修复第 3179 行
entropy_loss_h = sample_entropy[task_mask_h].mean() / config.GRADIENT_ACCUMULATION_STEPS  # 去掉负号
```

然后把 `LOW_MEMORY_MODE` 改回 `False`。

---

## 📊 同时应用其他修复

由于我们之前发现了两个问题：
1. 🐛 Entropy 梯度符号反了（本文档修复）
2. ⚠️ MIN_NEW_TOKENS (5) 与 SFT target (48) 不匹配

**建议同时应用两个修复：**

```python
# trainer.py 第 226 行（已修复）
MIN_NEW_TOKENS_TRAIN = 30  # 从 5 → 30

# trainer.py 第 203 行（已修复）
ENTROPY_COEF = 0.5  # 从 0.2 → 0.5

# trainer.py 第 286 行（本次修复）
LOW_MEMORY_MODE = True  # 从 False → True
```

---

## 🎯 预期效果（所有修复组合）

| 修复 | 影响 |
|------|------|
| LOW_MEMORY_MODE=True | Entropy 梯度符号正确 → Entropy 恢复到 0.5-1.5 |
| MIN_NEW_TOKENS=30 | EOS Suppressor 触发率降低 → 生成更自然 |
| ENTROPY_COEF=0.5 | 增强探索激励 → 进一步提升 Entropy |

**综合效果：**
- Entropy: 0.005 → **1.0-2.0**
- Max prob: 99.9% → **50-70%**
- 训练稳定性大幅提升
- 模型输出多样性恢复

---

## ❓ 如果修复后仍然没效果？

参考 `deep_entropy_diagnosis.py` 进行 10+ 项深度诊断：

1. Base model entropy 检查
2. Reward 信号退化检查
3. Advantage 计算检查
4. LoRA 梯度检查
5. Temperature 配置检查
6. KL penalty 检查
7. ... 等

每个诊断都有详细的代码示例和判断标准。

---

## 📤 测试后请报告

测试后请把前 5-10 步的日志发给我，包括：
- Logits 统计（Max logit, Gap, Top5）
- Entropy 诊断（mean, range）
- EOS Suppressor 触发情况

我会根据结果判断：
- ✅ Bug 已修复
- 🟡 部分改善，需要进一步调整
- ❌ 无效果，需要深度诊断

---

## 总结

**最快验证路径：**
1. 改 1 行：`LOW_MEMORY_MODE = True`
2. 运行 5-10 步
3. 看 Entropy 是否 > 0.3
4. ✅ 是 → Bug 确认！应用完整修复
5. ❌ 否 → 运行深度诊断脚本

**时间成本：**
- 修改代码：< 1 分钟
- 重新训练 10 步：~5-10 分钟
- 判断结果：立即

Let's fix this! 🚀
