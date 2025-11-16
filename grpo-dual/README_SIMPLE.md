# 超简单运行指南

## 📁 需要上传的文件

1. **必须上传**：
   - `src/grpo/trainer.py`
   - `src/judges/llm_judge_prompts_v2.py`
   - `data/bbq/*.jsonl` （BBQ数据集）
   - `data/halueval/*.json` （HaluEval数据集）

2. **可选（如果需要完整功能）**：
   - `src/models/lora_setup.py`
   - `src/evals/metrics.py`

---

## 🔧 安装依赖（第一次运行前）

```bash
pip install openai anthropic torch transformers peft datasets accelerate
```

或者：

```bash
pip install -r requirements.txt
```

---

## 🚀 运行训练

### 1. 设置 API Key

```bash
export OPENAI_API_KEY="sk-your-key"
```

### 2. 直接运行

```bash
python trainer.py
```

**就这么简单！** ✅

---

## 📝 trainer.py 会自动：

- ✅ 找到 `llm_judge_prompts_v2.py`（动态路径）
- ✅ 只使用 OpenAI Judge（已配置）
- ✅ 加载 BBQ 和 HaluEval 数据
- ✅ 开始训练

---

## ⚙️ 如果需要修改配置

在 `trainer.py` 文件中找到这些配置：

```python
class GRPOConfig:
    # LLM Judge 配置
    LLM_JUDGE_VERSION = "v2"  # 使用自适应 prompt

    # Judge Provider
    JUDGE_PROVIDERS = [
        {"name": "openai", "model": "gpt-4o-mini"}
    ]

    # 训练配置
    BATCH_SIZE = 8
    K_CANDIDATES = 4
    NUM_EPOCHS = 3

    # 数据集数量
    NUM_SAMPLES_FAIRNESS = 100
    NUM_SAMPLES_HALLUCINATION = 100
```

直接修改数字即可。

---

## 🐛 常见问题

### Q: ModuleNotFoundError: No module named 'llm_judge_prompts_v2'

**A:** 确保 `llm_judge_prompts_v2.py` 和 `trainer.py` 在同一个 workspace，trainer.py 会自动找到它。

### Q: No module named 'openai' / 'torch'

**A:** 运行安装命令：
```bash
pip install openai torch transformers peft
```

### Q: OPENAI_API_KEY not found

**A:** 设置环境变量：
```bash
export OPENAI_API_KEY="sk-your-key"
```

或在 Python 中：
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key"
```

---

## 📊 在 Jupyter Notebook 中运行

```python
import os

# 1. 设置 API Key
os.environ["OPENAI_API_KEY"] = "sk-your-key"

# 2. 导入并运行
from trainer import GRPOConfig, GRPOTrainer

config = GRPOConfig()
trainer = GRPOTrainer(config)
trainer.train()
```

---

就这么简单！🎉
