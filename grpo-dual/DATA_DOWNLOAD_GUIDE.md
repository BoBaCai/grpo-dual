# 数据集下载指南

本项目需要 BBQ 和 HaluEval 数据集。数据集可以从 GitHub 自动下载。

## 方法 1: 自动下载（推荐）

训练脚本会在运行时自动检查并下载缺失的数据集文件。你只需要：

```python
# 在 Jupyter notebook 中运行训练脚本
%run src/grpo/trainer.py
```

或

```bash
# 在命令行运行
python src/grpo/trainer.py
```

脚本会自动：
1. 检查数据文件是否存在
2. 如果缺失，从 GitHub 下载
3. 下载完成后开始训练

## 方法 2: 手动下载（Jupyter Notebook）

如果你想提前下载数据集，可以在 Jupyter notebook 中运行以下代码：

### 方式 A: 使用独立脚本

```python
# 直接运行下载脚本
%run src/grpo/download_data_notebook.py
```

### 方式 B: 复制粘贴到 Cell

将以下代码复制到 notebook cell 并运行：

```python
import os
import urllib.request
import urllib.error
from pathlib import Path
import time

# 配置
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/BoBaCai/grpo-dual/main/grpo-dual/data"
DATA_DIR = Path("./data")  # 修改为你想要的数据存储目录

# BBQ 数据集文件
BBQ_FILES = [
    "Age.jsonl",
    "Disability_status.jsonl",
    "Gender_identity.jsonl",
    "Nationality.jsonl",
    "Physical_appearance.jsonl",
    "Race_ethnicity.jsonl",
    "Race_x_SES.jsonl",
    "Race_x_gender.jsonl",
    "Religion.jsonl",
    "SES.jsonl",
    "Sexual_orientation.jsonl",
]

# HaluEval 数据集文件
HALUEVAL_FILES = [
    "dialogue_data.json",
    "general_data.json",
    "qa_data.json",
    "summarization_data.json",
]

# 创建目录
BBQ_DIR = DATA_DIR / "bbq"
HALUEVAL_DIR = DATA_DIR / "halueval"
BBQ_DIR.mkdir(parents=True, exist_ok=True)
HALUEVAL_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, dest_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"  下载: {dest_path.name} ... ", end="", flush=True)
            urllib.request.urlretrieve(url, dest_path)
            if dest_path.exists() and dest_path.stat().st_size > 0:
                print(f"✓ ({dest_path.stat().st_size / 1024:.1f} KB)")
                return True
            else:
                print(f"✗ (文件为空)")
                dest_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"✗ ({type(e).__name__}: {e})")
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    return False

# 下载 BBQ
print("="*80)
print(f"下载 BBQ 数据集")
print("="*80)
for filename in BBQ_FILES:
    dest_path = BBQ_DIR / filename
    if not dest_path.exists():
        download_file(f"{GITHUB_RAW_BASE}/bbq/{filename}", dest_path)
    else:
        print(f"  跳过: {filename} (已存在)")

# 下载 HaluEval
print("\n" + "="*80)
print(f"下载 HaluEval 数据集")
print("="*80)
for filename in HALUEVAL_FILES:
    dest_path = HALUEVAL_DIR / filename
    if not dest_path.exists():
        download_file(f"{GITHUB_RAW_BASE}/halueval/{filename}", dest_path)
    else:
        print(f"  跳过: {filename} (已存在)")

print("\n✓ 下载完成!")
```

## 方法 3: 使用命令行工具

```bash
# 下载所有数据集
python -m src.grpo.data_downloader --data-dir ./data

# 检查数据集状态
python -m src.grpo.data_downloader --data-dir ./data --check-only

# 强制重新下载
python -m src.grpo.data_downloader --data-dir ./data --force
```

## 数据集说明

### BBQ 数据集 (11 个文件)
用于评估模型的公平性（减少偏见）

- Age.jsonl - 年龄偏见
- Disability_status.jsonl - 残疾状况偏见
- Gender_identity.jsonl - 性别认同偏见
- Nationality.jsonl - 国籍偏见
- Physical_appearance.jsonl - 外貌偏见
- Race_ethnicity.jsonl - 种族/族裔偏见
- Race_x_SES.jsonl - 种族与社会经济地位交叉偏见
- Race_x_gender.jsonl - 种族与性别交叉偏见
- Religion.jsonl - 宗教偏见
- SES.jsonl - 社会经济地位偏见
- Sexual_orientation.jsonl - 性取向偏见

### HaluEval 数据集 (4 个文件)
用于评估模型的幻觉问题（减少事实错误）

- dialogue_data.json - 对话数据
- general_data.json - 通用数据
- qa_data.json - 问答数据
- summarization_data.json - 摘要数据

## 默认数据目录

训练脚本默认从以下位置读取数据：

```
/home/ubuntu/workspace/data/
├── bbq/
│   ├── Age.jsonl
│   ├── Disability_status.jsonl
│   └── ...
└── halueval/
    ├── dialogue_data.json
    ├── general_data.json
    ├── qa_data.json
    └── summarization_data.json
```

如果你的数据在其他位置，需要修改 `trainer.py` 中的 `Config.DATA_DIR` 配置。

## 故障排除

### 下载失败
- 检查网络连接
- 确认可以访问 GitHub
- 脚本会自动重试 3 次

### 文件已存在但损坏
使用 `--force` 参数强制重新下载：

```bash
python -m src.grpo.data_downloader --data-dir ./data --force
```

或在 notebook 中删除相应文件后重新运行下载脚本。

## GitHub 数据源

数据源地址：
https://github.com/BoBaCai/grpo-dual/tree/main/grpo-dual/data

原始 GitHub raw URL 格式：
```
https://raw.githubusercontent.com/BoBaCai/grpo-dual/main/grpo-dual/data/bbq/Age.jsonl
https://raw.githubusercontent.com/BoBaCai/grpo-dual/main/grpo-dual/data/halueval/qa_data.json
```
