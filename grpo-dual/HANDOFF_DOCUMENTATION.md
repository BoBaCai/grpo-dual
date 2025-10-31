# GRPO训练熵塌陷问题 - 技术交接文档

**日期**: 2025-10-31
**分支**: `claude/fix-github-access-011CUft7hVXUTrXCmUqYwWHY`
**状态**: 🔴 严重问题未解决，调试中

---

## 📋 执行摘要

**核心问题**: GRPO多目标强化学习训练中，Fairness任务出现**严重熵塌陷**（entropy collapse），导致模型只生成1-2个token就输出EOS，训练无法进行。

**已尝试3轮修复，均失败**:
1. ❌ 第一轮：添加熵正则化 + 降低温度 → 熵塌陷恶化
2. ❌ 第二轮：紧急修正，提高温度 → 熵塌陷持续
3. 🔍 第三轮：添加调试日志，诊断根本原因 ← **当前状态**

---

## 🔥 关键问题现象

### 训练表现（来自最新结果）
```
Step 3:  熵=0.000, 长度=2 tokens, 4个rollouts生成完全相同输出
Step 5:  熵=0.001, 长度=2 tokens
Step 8:  熵=3.450 (正常!)
Step 10: 熵=0.000, 长度=1 token
```

**异常特征**:
- 尽管设置了`temperature=0.4`, `K_ROLLOUTS=4`的独立采样仍生成相同输出
- 长度惩罚和`MIN_NEW_TOKENS=3`完全失效
- 只有个别步骤正常（如Step 8），说明不是模型永久损坏

---

## 🎯 项目背景

### 任务描述
使用GRPO（Group Relative Policy Optimization）同时优化两个目标：
1. **Fairness**: BBQ数据集，检测并消除偏见
2. **Hallucination**: HaluEval数据集，检测幻觉

### 技术栈
- **模型**: Qwen2.5-0.5B-Instruct (LoRA微调)
- **优化器**: AdamW, lr=1e-5
- **RL算法**: GRPO with Lagrangian KL控制器（双分支β）
- **梯度处理**: CAGrad（冲突避免梯度下降）
- **配置**: `K_ROLLOUTS=4`, `BATCH_SIZE=8`, `GRPO_STEPS=500`

### 历史诊断（来自前任Claude）
发现了3个根本问题：
1. **Fairness任务KL震荡** → 使用分支化KL控制器
2. **熵塌陷** → CS294-196课程指出：低熵→确定性策略→训练崩溃
3. **奖励信号不平衡** → F/H奖励比例异常（2.3-7.6倍）

---

## 💻 当前代码状态

### 最新修改（trainer.py）

#### 1. 关键配置（203-210行）
```python
ENTROPY_COEF = 0.05              # 熵正则化系数（已增强5倍）
FAIRNESS_REWARD_SCALE = 0.7      # Fairness奖励缩放
HALLUCINATION_REWARD_SCALE = 1.0 # Hallucination奖励缩放
TEMPERATURE_TRAIN = 0.4          # 生成温度（从0.25提高）
MIN_NEW_TOKENS_TRAIN = 3         # 最小生成长度
```

#### 2. 熵正则化实现（2467-2624行）
**低显存模式**（2510-2520行）:
```python
# 计算token熵
cur_probs = torch.exp(cur_logp)
token_entropy = -(cur_probs * cur_logp).sum(dim=-1)
sample_entropy = (token_entropy * comp_mask).sum(dim=1) / denom

# 添加到损失
loss_fair = (-(surr[task_mask_f].mean()) + beta_f * kl[task_mask_f].mean()
             - config.ENTROPY_COEF * entropy_f) / config.GRADIENT_ACCUMULATION_STEPS
```

**完整模式**（2608-2624行）:
```python
vec_final = vec_reward_merged + beta_f * vec_kl_f + beta_h * vec_kl_h
            - config.ENTROPY_COEF * (vec_entropy_f + vec_entropy_h)
```

#### 3. 奖励缩放（2283-2288行）
```python
for i in range(len(rewards)):
    if task_list[i] == "fairness":
        rewards[i] *= config.FAIRNESS_REWARD_SCALE  # 0.7
    elif task_list[i] == "hallucination":
        rewards[i] *= config.HALLUCINATION_REWARD_SCALE  # 1.0
```

#### 4. 🆕 调试日志（1576-1616行）
**刚刚添加**的`DebugLogitsProcessor`会在前20步打印：
```python
🔍 [Step X] Logits Distribution Debug:
   Temperature: 0.4
   Max logit: XX.XXX         # 原始logit最大值
   Gap (1st-2nd): XX.XXX     # 第1和第2大logit的差距（尖锐度指标）
   Top-5 probs: [...]        # 温度缩放后的top-5概率
   Max prob: 0.XXXXXX        # 最大概率（判断是否过于确定性）
```

---

## 🔬 失败历程与分析

### 第一轮修复（已失败）
**改动**:
- `ENTROPY_COEF = 0.01`
- `FAIRNESS_REWARD_SCALE = 0.5`
- `TEMPERATURE_TRAIN = 0.25` ⬇️（降低）

**结果**: 完全失败，熵值仍为0.0-0.4

**失败原因分析**:
1. 熵系数0.01太弱，无法抗衡奖励/KL梯度
2. **降低温度是错误的**：低温→采样更确定性→加剧熵塌陷（采样-训练不匹配）

### 第二轮紧急修正（已失败）
**改动**:
- `ENTROPY_COEF = 0.05` ⬆️（5倍增强）
- `FAIRNESS_REWARD_SCALE = 0.7`（中间值）
- `TEMPERATURE_TRAIN = 0.4` ⬆️（提高探索）

**预期**: 更高温度应该增加采样多样性，打破塌陷循环

**实际结果**: 仍然失败，Step 3看到4个rollouts生成完全相同的2-token输出

**疑点**:
- 温度是否真的被应用到生成？
- 模型logits是否已经太尖锐（sharp），温度无法软化？
- 模型权重是否已不可逆地塌陷？

### 第三轮调试（当前）
**改动**: 添加`DebugLogitsProcessor`打印logit分布

**目标**: 确定根本原因是下列哪个：
1. 温度参数未生效（代码bug）
2. Logits极度尖锐（如gap>20, max_prob>0.99）
3. 其他原因

---

## 🚨 紧急待办事项

### 立即执行

1. **运行训练并查看调试输出**
   ```bash
   # 训练会在前20步打印logits诊断信息
   # 重点关注 Max prob 和 Gap 值
   ```

2. **根据调试输出判断**:

   **场景A: Max prob > 0.95, Gap > 15**
   → Logits极度尖锐，温度无法挽救

   **解决方案**:
   - 回退到更早的checkpoint（熵塌陷前）
   - 或尝试极端温度（0.7-1.0）+ 极高熵系数（0.2）
   - 或添加logits平滑处理器

   **场景B: Max prob < 0.7, Gap < 5**
   → 温度有效，但熵塌陷仍发生

   **解决方案**:
   - 问题在训练侧，不在生成侧
   - 检查熵正则化梯度是否被覆盖
   - 增加`MIN_NEW_TOKENS`到10
   - 检查KL惩罚是否过强

   **场景C: 温度值显示不是0.4**
   → 代码bug，配置未生效

   **解决方案**:
   - 检查config模块导入
   - 验证`config.TEMPERATURE_TRAIN`在运行时的值

### 次要任务

3. **奖励信号诊断**
   - 当前F/H信号比例诊断不可靠（出现百万级异常值）
   - 需要修复除零问题：当H信号=0.0000时
   - 建议添加`if abs(h_signal) < 1e-6: continue`

4. **长度惩罚增强**
   - 当前`MIN_NEW_TOKENS=3`无效
   - 考虑添加硬约束：在reward中减去`max(0, 5 - length) * 10.0`

---

## 📁 关键文件位置

```
/home/user/grpo-dual/grpo-dual/
├── src/grpo/trainer.py          # 主训练文件（~2800行）
│   ├── L203-210: 配置参数
│   ├── L1576-1616: DebugLogitsProcessor（新增）
│   ├── L1700-1750: generate_candidates_batch（生成函数）
│   ├── L2145: grpo_train（主训练循环）
│   ├── L2283-2288: 奖励缩放
│   ├── L2467-2520: 熵正则化（低显存模式）
│   └── L2608-2624: 熵正则化（完整模式）
├── configs/train_example.yaml   # 配置示例
└── HANDOFF_DOCUMENTATION.md     # 本文档
```

---

## 🔧 可用的诊断工具

### 已实现的监控（每步打印）
```python
# Step输出示例
Step 3/500 | 📊 任务分布：fairness=4/8  hallucination=4/8
  生成样例[fairness]:
    Generated: .assistant  Answer: [Based on context]  Justification: ...
    Length: 2 tokens, Entropy: 0.000  # ← 关键指标

  # 信号诊断
  F: rew_μ=X.XX, kl_μ=X.XX  # Fairness信号统计
  H: rew_μ=X.XX, kl_μ=X.XX  # Hallucination信号统计
  F/H信号比 = X.XX          # 平衡性指标
```

### 新增的调试输出（前20步）
```python
🔍 [Step X] Logits Distribution Debug:
   Temperature: 0.4
   Max logit: XX.XXX
   Gap (1st-2nd): XX.XXX
   Top-5 probs: [0.XXX, ...]
   Max prob: 0.XXXXXX
```

---

## 📚 理论背景参考

### 熵塌陷机制（CS294-196）
1. **初始**: 模型输出多样 → 高熵
2. **奖励引导**: 某些输出获得高奖励
3. **策略更新**: 模型向高奖励方向更新
4. **塌陷**: 如果没有熵正则化，策略变成确定性（熵→0）
5. **死循环**: 确定性策略 → 低多样性数据 → 进一步确定性

### GRPO特有风险
- **多目标冲突**: Fairness和Hallucination可能要求矛盾的行为
- **任务难度不均**: Hallucination较易，Fairness较难 → Fairness容易放弃
- **KL惩罚**: 如果β过大，模型拒绝探索 → 熵下降

---

## 🎓 推荐阅读

1. **原GRPO论文**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
2. **CS294-196 Lecture**: Deep RL entropy regularization
3. **Xiaomei Song的RL指南**: Multi-objective reward balancing

---

## ⚡ 快速命令参考

```bash
# 切换到工作分支
git checkout claude/fix-github-access-011CUft7hVXUTrXCmUqYwWHY

# 拉取最新代码
git pull origin claude/fix-github-access-011CUft7hVXUTrXCmUqYwWHY

# 查看最近的改动
git log --oneline -10

# 运行训练（假设有启动脚本）
# python -m grpo.trainer --config configs/train_example.yaml

# 如果需要回退
git log  # 找到健康的commit
git checkout <commit-hash> src/grpo/trainer.py
```

---

## 🤔 未验证的假设

1. **温度是否正确应用**: 等待调试输出确认
2. **熵正则化梯度是否生效**: 未验证实际梯度大小
3. **LoRA是否限制了探索**: 可能需要更大的rank
4. **数据集问题**: Fairness样本是否特别短/简单？
5. **Lagrangian β初始化**: β_f和β_h初始值可能不当

---

## 💡 备用方案

如果当前路线全部失败，考虑：

### Plan A: 回退checkpoint
```python
# 找到Step 8那样熵正常的早期checkpoint
# 从那里重新开始，使用更保守的学习率
```

### Plan B: 单任务训练
```python
# 先单独训练Fairness，稳定后再加入Hallucination
# 避免多目标冲突加剧熵塌陷
```

### Plan C: 架构调整
```python
# 增加LoRA rank: 8 → 16
# 使用更大的基模型（1.5B代替0.5B）
# 改用PPO替代GRPO
```

---

## 📞 联系与问题

**当前最紧急的问题**:
> 为什么temperature=0.4, K_ROLLOUTS=4的情况下，4个独立采样会生成完全相同的2-token输出？

**调试输出会揭示**:
- 如果Max prob > 0.99 → 模型logits塌陷，温度救不了
- 如果Max prob < 0.7 → 问题在别处，不是生成阶段

**下一步取决于调试结果** - 请先运行训练，查看调试输出，然后根据"紧急待办事项"中的场景A/B/C决定方向。

---

## ✅ 检查清单（接手后）

- [ ] 确认已在正确分支: `claude/fix-github-access-011CUft7hVXUTrXCmUqYwWHY`
- [ ] 阅读完本文档
- [ ] 查看最近3个commits的改动
- [ ] 运行训练，获取调试输出
- [ ] 根据调试输出判断场景A/B/C
- [ ] 实施对应的解决方案
- [ ] 如果仍然失败，escalate或考虑备用方案

---

**文档版本**: v1.0
**最后更新**: 2025-10-31
**作者**: Claude (Session 011CUft7hVXUTrXCmUqYwWHY)

**祝好运！这是一个棘手的问题，但可以解决。关键是先确定根本原因，不要盲目尝试。**
