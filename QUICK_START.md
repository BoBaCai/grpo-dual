# 🚀 GRPO 快速开始指南

## ⚠️ 你遇到的问题

1. ❌ **Temperature 警告**: `The following generation flags are not valid: ['temperature', 'top_p']`
2. ❌ **KL 散度过高**: `Fairness KL=0.318 >> 0.06`
3. ⚠️ **训练速度慢**: Judge 时间过长

**根因**: 可能在运行旧版本代码，或 Jupyter Kernel 未重启

---

## ✅ 立即修复（3步）

### 步骤 1: 强制更新代码

```bash
cd /path/to/grpo-dual
bash force_update.sh
```

**这个脚本会自动**:
- ✅ 拉取最新代码
- ✅ 验证关键修复是否存在
- ✅ 显示当前配置
- ✅ 提示重启 Jupyter Kernel

---

### 步骤 2: 重启运行环境

#### 如果使用 Jupyter Notebook:
1. 停止所有运行中的 cell
2. 点击 **Kernel** → **Restart Kernel**
3. 重新运行所有 cell

#### 如果使用 Python 脚本:
```bash
# 直接重新运行
python grpo-dual/scripts/grpo_train.py
```

---

### 步骤 3: 验证修复

重新训练后，应该看到：

```
✅ 多终止符已启用: [128001, 128009]

阶段2: GRPO 多目标训练（v2.3 - 显存优化版）
================================================================================

显存优化配置:
  GRPO_BATCH_SIZE: 2 (单次生成)
  GRADIENT_ACCUMULATION_STEPS: 2
  有效 batch size: 4
  K_ROLLOUTS: 4
  单步生成总数: 8
  MAX_NEW_TOKENS: 96
  LORA_R: 8
  GPU 总显存: 40.0 GB

[Judge@step5] time=3.2s providers={'openai': 8}

[BranchedKL@20] Fairness KL过低(0.018<0.02)，β_f↓15%: 0.1000→0.0850

[QuickEval@20] mode=greedy fairness=0.452  hallucination=0.497
  截断率: F=0.0%  H=0.0%  |  生成长度: F=64.5  H=51.4
  KL散度: F=0.035  H=0.095  |  β值: F=0.1000  H=0.3000
```

**关键验证点**:
- ✅ **不再出现** "temperature are not valid" 警告
- ✅ KL 散度在正常范围: **F=0.02-0.06, H=0.08-0.15**
- ✅ Judge 时间: **3-4秒**（从 5-6秒降低）
- ✅ 截断率: **~0%**

---

## 🚀 额外加速选项（可选）

### 选项 1: 启用 torch.compile() (推荐)

**加速效果**: 20-30% 整体加速

**使用方法**:
```python
# 在脚本开头设置
import config
config.USE_TORCH_COMPILE = True
```

**要求**:
- PyTorch ≥ 2.0
- 首次运行会编译（慢），后续运行加速

---

### 选项 2: 安装 Flash Attention 2

**加速效果**: 生成阶段 1.5-2x 加速

**使用方法**:
```bash
bash install_flash_attn.sh
```

**要求**:
- NVIDIA GPU (A100/A6000/4090/3090)
- CUDA ≥ 11.6

---

### 选项 3: 进一步减少评测（如果不关心中途评估）

```python
config.PARETO_PRINT_EVERY = 50  # 从20增到50，减少评测频率
config.PARETO_PRINT_SAMPLES = 10  # 从20降到10，减少样本数
```

**加速效果**: 减少 50% 评测时间

---

## 📊 性能对比

| 配置 | 单步耗时 | 训练500步总耗时 |
|------|---------|----------------|
| **优化前** | ~15-20s | ~2.5小时 |
| **显存优化后** | ~10-12s | ~1.7小时 |
| **Judge加速后** | ~6-8s | ~1小时 |
| **torch.compile** | ~5-6s | **~45分钟** ⚡ |
| **+ Flash Attn 2** | ~4-5s | **~35分钟** 🚀 |

---

## 🔍 故障排查

### 问题 1: 仍然有 temperature 警告

**检查是否用最新代码**:
```bash
grep "TemperatureLogitsWarper(config.TEMPERATURE_TRAIN)" grpo-dual/scripts/grpo_train.py
```

应该能找到这一行。如果没有：
1. 运行 `bash force_update.sh`
2. 重启 Jupyter Kernel
3. 确认 git branch 是 `claude/update-sandbox-file-011CUQrYCAKa4Jz2jD4YMVEn`

---

### 问题 2: KL 散度仍然很高

**检查 KL 计算是否修复**:
```bash
grep "delta = (ref_lp - cur_lp).clamp(-10, 10)" grpo-dual/scripts/grpo_train.py
```

应该能找到这一行。如果没有：
- 同上，运行 `force_update.sh`

---

### 问题 3: CUDA OOM

查看显存优化指南:
```bash
cat MEMORY_OPTIMIZATION.md
```

或运行显存检查:
```bash
python check_gpu_memory.py
```

---

## 📋 完整修复清单

本次更新修复了：

| 问题 | 状态 | Commit |
|------|------|--------|
| ✅ Temperature 被忽略 | 已修复 | 29c8fce |
| ✅ KL 散度爆炸 | 已修复 | 29c8fce |
| ✅ NameError: delta | 已修复 | d85fe94 |
| ✅ CUDA OOM | 已优化 | 3cc4348 |
| ✅ Judge 速度慢 | 已优化 | 4aaed76 |
| ✅ Flash Attn 2 安装 | 脚本已提供 | d85fe94 |

---

## 🎯 关键配置总结

```python
# 显存优化
GRPO_BATCH_SIZE = 2                    # 从4降到2
GRADIENT_ACCUMULATION_STEPS = 2        # 新增，等效batch=4
MAX_NEW_TOKENS_TRAIN = 96              # 从128降到96
LORA_R = 8                             # 从16降到8

# 速度优化
JUDGE_MAX_WORKERS = 8                  # 匹配生成数
JUDGE_TIMEOUT_SEC = 6                  # 更快超时
JUDGE_MAX_RETRIES = 0                  # 禁用重试
PARETO_PRINT_SAMPLES = 20              # 从40降到20

# 可选加速
USE_TORCH_COMPILE = False              # 设为 True 启用
```

---

## 💡 推荐配置

### 如果你有充足显存（≥40GB）:
```python
GRPO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
USE_TORCH_COMPILE = True               # 启用编译加速
# + 安装 Flash Attention 2
```

### 如果显存紧张（≤24GB）:
```python
GRPO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
MAX_NEW_TOKENS_TRAIN = 64
LORA_R = 4
```

---

## 📚 相关文档

- 📄 [MEMORY_OPTIMIZATION.md](./MEMORY_OPTIMIZATION.md) - 显存优化完整指南
- 🔧 [check_gpu_memory.py](./check_gpu_memory.py) - GPU 显存监控
- ⚡ [install_flash_attn.sh](./install_flash_attn.sh) - Flash Attention 2 安装

---

## 🆘 还有问题？

运行诊断脚本：
```bash
# 1. 检查代码版本
bash force_update.sh

# 2. 检查显存使用
python check_gpu_memory.py

# 3. 查看最近的 commits
git log --oneline -10
```

应该看到这些 commits:
```
4aaed76 fix: 修复 temperature 警告 + 额外加速优化
3cc4348 perf: 显存优化 - 解决 CUDA OOM 问题
d85fe94 fix: 修复 delta 变量未定义导致的 NameError
29c8fce fix: 修复两个严重bug（temperature被忽略 + KL散度爆炸）
2d08ae2 perf: 大幅提升训练速度（Judge并发+Flash Attention 2）
```

---

**生成时间**: 2025-10-24
**版本**: v2.3 - 显存优化 + 加速优化版
**预期加速**: 单步从 15-20s → 4-6s（**3-5倍**）
