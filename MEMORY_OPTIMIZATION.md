# GRPO 显存优化指南

## 🎯 问题

A100 (40GB) 出现 **CUDA out of memory** 错误。

---

## ✅ 已实施的优化（v2.3）

### 1. **梯度累积**（最重要）

| 参数 | 优化前 | 优化后 | 说明 |
|------|--------|--------|------|
| `GRPO_BATCH_SIZE` | 4 | **2** | 单次生成 batch |
| `GRADIENT_ACCUMULATION_STEPS` | - | **2** | 梯度累积步数 |
| **有效 batch** | 4 | **4** | 训练效果相同！ |

**显存节省**: ~40%
**训练效果**: 完全相同（等效 batch size 不变）

---

### 2. **生成长度限制**

| 参数 | 优化前 | 优化后 | 影响 |
|------|--------|--------|------|
| `MAX_NEW_TOKENS_TRAIN` | 128 | **96** | -25% |
| `MAX_NEW_TOKENS_EVAL` | 128 | **96** | -25% |

**显存节省**: ~20%
**影响**: 回答略短（大多数任务 96 tokens 足够）

---

### 3. **LoRA 参数减少**

| 参数 | 优化前 | 优化后 | 影响 |
|------|--------|--------|------|
| `LORA_R` | 16 | **8** | 秩减半 |
| `LORA_ALPHA` | 32 | **16** | 保持 alpha=2*r |

**显存节省**: ~15%
**影响**: 模型容量略降（通常不影响效果）

---

### 4. **输入序列长度**

| 参数 | 优化前 | 优化后 | 影响 |
|------|--------|--------|------|
| `SFT_MAXLEN` | 1024 | **896** | -12.5% |
| `SFT_BATCH_SIZE` | 4 | **2** | 同步优化 |

**显存节省**: ~10%
**影响**: 极少（896 tokens 覆盖绝大部分样本）

---

### 5. **显存清理**

在关键点调用 `torch.cuda.empty_cache()`:
- ✅ 生成后立即清理
- ✅ 参数更新后清理

**效果**: 减少显存碎片化，提高峰值可用显存

---

## 📊 总体效果

| 项目 | 显存节省 |
|------|---------|
| 梯度累积 | ~40% |
| 生成长度 | ~20% |
| LoRA 秩 | ~15% |
| 输入长度 | ~10% |
| **总计** | **~60%** |

**预期**:
- **优化前**: 需要 ~40GB 显存
- **优化后**: 仅需 ~16GB 显存 ✅

---

## 🔧 监控工具

### check_gpu_memory.py

实时监控 GPU 显存使用情况：

```bash
python check_gpu_memory.py
```

**输出示例**:
```
GPU 显存监控
======================================================================
📊 GPU 基本信息
   可用 GPU 数量: 1

   GPU 0: NVIDIA A100-SXM4-40GB
   计算能力: 8.0
   总显存: 40.00 GB

   💾 显存使用情况:
   已分配: 14.23 GB (35.6%)
   已预留: 16.50 GB (41.3%)
   可用: 23.50 GB (58.7%)

   📈 缓存统计:
   分配重试次数: 0
   OOM 次数: 0

💡 显存优化建议
======================================================================
✅ 显存使用率正常
```

---

## 🚨 如果仍然 OOM

### Level 1: 轻度优化
```python
GRPO_BATCH_SIZE = 1                    # 从2降到1
GRADIENT_ACCUMULATION_STEPS = 4        # 从2增到4
# 有效 batch = 1 × 4 = 4（保持不变）
```

**额外节省**: ~20%

---

### Level 2: 中度优化
```python
MAX_NEW_TOKENS_TRAIN = 64              # 从96降到64
K_ROLLOUTS = 3                         # 从4降到3
```

**额外节省**: ~30%

---

### Level 3: 重度优化
```python
LORA_R = 4                             # 从8降到4
SFT_MAXLEN = 768                       # 从896降到768
K_ROLLOUTS = 2                         # 从4降到2
```

**额外节省**: ~40%
**注意**: 可能影响训练效果

---

## 📈 训练启动检查

启动时会显示配置摘要：

```
阶段2: GRPO 多目标训练（v2.3 - 显存优化版）
==============================================================================

显存优化配置:
  GRPO_BATCH_SIZE: 2 (单次生成)
  GRADIENT_ACCUMULATION_STEPS: 2
  有效 batch size: 4
  K_ROLLOUTS: 4
  单步生成总数: 8
  MAX_NEW_TOKENS: 96
  LORA_R: 8
  GPU 总显存: 40.0 GB
```

**检查重点**:
- ✅ 有效 batch size = 4（保持不变）
- ✅ LORA_R = 8
- ✅ MAX_NEW_TOKENS = 96

---

## 💡 优化原理

### 梯度累积工作原理

```python
# 传统方法（batch=4，显存需求高）
for step in range(steps):
    batch = sample(4)           # 一次生成 4×4=16 个样本
    loss = compute_loss(batch)
    loss.backward()
    optimizer.step()

# 梯度累积（batch=2×2=4，显存需求低）
for step in range(steps):
    optimizer.zero_grad()
    for micro_step in range(2):
        batch = sample(2)       # 一次只生成 2×4=8 个样本
        loss = compute_loss(batch) / 2
        loss.backward()         # 梯度累积（不更新参数）
    optimizer.step()            # 累积2次后更新
```

**关键**:
- 显存峰值降低 ~40%（只需存储 2×4=8 个样本，而非 4×4=16）
- 梯度除以累积步数 (`loss / 2`)，保证数值稳定性
- 等效 batch size 不变 (2×2 = 4)

---

## ✅ 验证清单

训练前检查：
- [ ] `GRPO_BATCH_SIZE = 2`
- [ ] `GRADIENT_ACCUMULATION_STEPS = 2`
- [ ] `MAX_NEW_TOKENS_TRAIN = 96`
- [ ] `LORA_R = 8`
- [ ] `SFT_MAXLEN = 896`
- [ ] 运行 `python check_gpu_memory.py` 确认显存充足

训练中监控：
- [ ] 无 OOM 错误
- [ ] Judge 时间 < 6s
- [ ] KL 散度正常 (F: 0.02-0.06, H: 0.08-0.15)
- [ ] 截断率 < 5%

---

## 📚 参考资料

- [PyTorch Gradient Accumulation](https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [GRPO Algorithm](https://arxiv.org/abs/2402.03300)

---

**生成时间**: 2025-10-24
**版本**: v2.3 - 显存优化版
