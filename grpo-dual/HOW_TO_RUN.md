# 如何运行 GRPO Trainer + LLM Judge

## 📁 项目结构

```
grpo-dual/
├── src/
│   ├── grpo/
│   │   ├── __init__.py
│   │   └── trainer.py          # ✅ 主训练器（已修复导入路径）
│   └── judges/
│       ├── __init__.py
│       └── llm_judge_prompts_v2.py  # ✅ LLM Judge V2（自适应评分）
├── data/
│   ├── bbq/                    # BBQ 公平性数据集
│   └── halueval/               # HaluEval 幻觉检测数据集
├── notebooks/
│   ├── llm_judge_usage_example.ipynb
│   └── QUICK_START.md
└── requirements.txt
```

---

## 🚀 方式 1：本地运行（推荐）

### **1.1 环境准备**

```bash
# 克隆仓库
git clone https://github.com/BoBaCai/grpo-dual.git
cd grpo-dual

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### **1.2 设置 API Key**

```bash
# 设置 OpenAI API Key（必须）
export OPENAI_API_KEY="sk-your-api-key-here"

# 可选：Anthropic API Key（如果需要Claude Judge）
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### **1.3 运行训练**

**方式 A：直接运行 trainer.py**

```bash
cd grpo-dual/src/grpo
python trainer.py
```

**方式 B：使用 Python 模块运行**

```bash
cd grpo-dual
python -m src.grpo.trainer
```

**方式 C：在 Python 脚本中导入**

```python
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path("/path/to/grpo-dual")
sys.path.insert(0, str(project_root / "src" / "grpo"))

# 设置 API Key
os.environ["OPENAI_API_KEY"] = "sk-your-key"

# 导入并运行
from trainer import GRPOConfig, GRPOTrainer

config = GRPOConfig()
trainer = GRPOTrainer(config)
trainer.train()
```

---

## 🌐 方式 2：从 GitHub 直接调用（高级）

### **2.1 动态下载并导入（实验性）**

如果您想直接从 GitHub 拉取最新代码：

```python
import os
import sys
import urllib.request
from pathlib import Path

# 下载 llm_judge_prompts_v2.py
github_raw = "https://raw.githubusercontent.com/BoBaCai/grpo-dual/main/src/judges/llm_judge_prompts_v2.py"
local_path = Path.cwd() / "llm_judge_prompts_v2.py"

# 下载文件
urllib.request.urlretrieve(github_raw, local_path)

# 添加到路径
sys.path.insert(0, str(Path.cwd()))

# 现在可以导入
from llm_judge_prompts_v2 import get_adaptive_bbq_prompt, get_adaptive_halueval_prompt
```

### **2.2 使用 Git 子模块（推荐给协作项目）**

如果您在另一个项目中使用 grpo-dual：

```bash
# 在您的项目中添加子模块
git submodule add https://github.com/BoBaCai/grpo-dual.git external/grpo-dual
git submodule update --init --recursive

# 在代码中使用
import sys
sys.path.insert(0, "external/grpo-dual/src/judges")
from llm_judge_prompts_v2 import get_adaptive_bbq_prompt
```

---

## 📊 方式 3：在 Jupyter Notebook 中使用（最简单）

### **3.1 上传文件到 workspace**

把 `llm_judge_prompts_v2.py` 上传到你的 Jupyter workspace 文件夹

### **3.2 准备环境（Cell 1）**

```python
import os
import sys
from pathlib import Path

# ✅ 直接从当前目录导入
sys.path.insert(0, str(Path.cwd()))

# 设置 API Key
os.environ["OPENAI_API_KEY"] = "sk-your-key"
```

### **3.3 导入并使用（Cell 2）**

```python
from llm_judge_prompts_v2 import get_adaptive_bbq_prompt, get_adaptive_halueval_prompt
from openai import OpenAI
import json

client = OpenAI()

def judge_score(judge_prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=200
    )
    return json.loads(response.choices[0].message.content)

# 使用示例（详见 notebooks/QUICK_START.md）
```

---

## ⚙️ 配置说明

### **Trainer 配置（trainer.py）**

```python
class GRPOConfig:
    # LLM Judge 配置
    LLM_JUDGE_VERSION = "v2"  # ✅ 使用自适应 prompt

    # Judge Provider（只用 OpenAI）
    JUDGE_PROVIDERS = [
        {"name": "openai", "model": "gpt-4o-mini"}
    ]

    # 训练配置
    BATCH_SIZE = 8
    K_CANDIDATES = 4  # 每个 prompt 生成 4 个候选
    NUM_EPOCHS = 3

    # 数据集配置
    NUM_SAMPLES_FAIRNESS = 100
    NUM_SAMPLES_HALLUCINATION = 100
```

### **Judge 配置（llm_judge_prompts_v2.py）**

```python
# 文件末尾的配置
JUDGE_MODE = "llm_adaptive"
LLM_JUDGE_MODEL = "gpt-4o-mini"
LLM_JUDGE_TEMPERATURE = 0.0
LLM_JUDGE_MAX_TOKENS = 200
```

---

## 🔧 路径修复说明（已完成）

**问题**：之前 trainer.py 中的导入会失败

**修复**（已自动修复）：
```python
# trainer.py:2042-2053
# 动态添加judges目录到路径
import sys
from pathlib import Path
judges_dir = Path(__file__).parent.parent / "judges"
if str(judges_dir) not in sys.path:
    sys.path.insert(0, str(judges_dir))

# 现在可以正常导入
from llm_judge_prompts_v2 import get_adaptive_bbq_prompt, get_adaptive_halueval_prompt
```

**工作原理**：
- `Path(__file__)` → `/path/to/grpo-dual/src/grpo/trainer.py`
- `.parent.parent` → `/path/to/grpo-dual/src/`
- `/ "judges"` → `/path/to/grpo-dual/src/judges/`
- 添加到 `sys.path`，Python 就能找到 `llm_judge_prompts_v2.py`

---

## 📦 需要提供的文件（如果给别人）

### **最小工作集**

如果您想给别人使用 LLM Judge V2，只需提供：

1. **`llm_judge_prompts_v2.py`** - Judge 核心逻辑
2. **使用说明** - `notebooks/QUICK_START.md` 或本文档

### **完整项目**

如果要运行完整训练：

1. **代码文件**：
   - `src/grpo/trainer.py`
   - `src/judges/llm_judge_prompts_v2.py`
   - `src/models/lora_setup.py`
   - `src/evals/metrics.py`

2. **数据集**：
   - `data/bbq/*.jsonl`
   - `data/halueval/*.json`

3. **配置文件**：
   - `requirements.txt`
   - `configs/*.yaml`（如果有）

4. **文档**：
   - `README.md`
   - `HANDOFF.md`
   - `notebooks/QUICK_START.md`

---

## 🌐 GitHub 集成建议

### **方式 A：发布 Python 包（PyPI）**

如果您想让别人通过 `pip install` 使用：

```bash
# 1. 创建 setup.py
cat > setup.py <<EOF
from setuptools import setup, find_packages

setup(
    name="grpo-dual",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "openai>=2.3.0",
        "anthropic>=0.69.0",
        "torch>=2.0.0",
        # ... 其他依赖
    ],
)
EOF

# 2. 安装
pip install -e .

# 3. 使用
from judges.llm_judge_prompts_v2 import get_adaptive_bbq_prompt
```

### **方式 B：直接从 GitHub 安装**

```bash
pip install git+https://github.com/BoBaCai/grpo-dual.git
```

### **方式 C：GitHub Release + 单文件下载**

创建 GitHub Release，附带：
- `llm_judge_prompts_v2.py`（单文件版本）
- `QUICK_START.md`

用户可直接下载单个文件使用。

---

## 🐛 常见问题

### **Q1: ModuleNotFoundError: No module named 'llm_judge_prompts_v2'**

**解决方案**：
```python
# 确保添加正确的路径
import sys
from pathlib import Path
sys.path.insert(0, str(Path("grpo-dual/src/judges")))
```

### **Q2: 如何验证导入成功？**

```python
# 测试导入
try:
    from llm_judge_prompts_v2 import get_adaptive_bbq_prompt
    print("✅ 导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print(f"当前 sys.path: {sys.path}")
```

### **Q3: 从 GitHub 拉取时路径怎么办？**

```python
# 动态下载后添加路径
import urllib.request
from pathlib import Path

# 下载到当前目录
url = "https://raw.githubusercontent.com/BoBaCai/grpo-dual/main/src/judges/llm_judge_prompts_v2.py"
Path("judges").mkdir(exist_ok=True)
urllib.request.urlretrieve(url, "judges/llm_judge_prompts_v2.py")

# 添加路径
import sys
sys.path.insert(0, str(Path.cwd()))

# 导入
from judges.llm_judge_prompts_v2 import get_adaptive_bbq_prompt
```

---

## 📞 联系与支持

- **GitHub Issues**: https://github.com/BoBaCai/grpo-dual/issues
- **文档**: 见 `notebooks/` 目录

祝您使用顺利！🚀
