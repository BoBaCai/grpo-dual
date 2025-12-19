# 适用于 Jupyter Notebook Cell 的数据下载脚本
# 可以直接复制粘贴到 notebook cell 中运行

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
        except urllib.error.HTTPError as e:
            print(f"✗ (HTTP {e.code})")
            if e.code == 404:
                print(f"    文件不存在: {url}")
                return False
        except Exception as e:
            print(f"✗ ({type(e).__name__}: {e})")

        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            print(f"    等待 {wait_time}s 后重试...")
            time.sleep(wait_time)
    return False

# 下载 BBQ 数据集
print("="*80)
print(f"下载 BBQ 数据集 ({len(BBQ_FILES)} 个文件)")
print(f"目标目录: {BBQ_DIR}")
print("="*80)

bbq_downloaded = 0
bbq_skipped = 0

for filename in BBQ_FILES:
    dest_path = BBQ_DIR / filename
    if dest_path.exists():
        print(f"  跳过: {filename} (已存在)")
        bbq_skipped += 1
        continue
    url = f"{GITHUB_RAW_BASE}/bbq/{filename}"
    if download_file(url, dest_path):
        bbq_downloaded += 1

print(f"\nBBQ 下载完成: {bbq_downloaded} 个新文件, {bbq_skipped} 个已存在")

# 下载 HaluEval 数据集
print("\n" + "="*80)
print(f"下载 HaluEval 数据集 ({len(HALUEVAL_FILES)} 个文件)")
print(f"目标目录: {HALUEVAL_DIR}")
print("="*80)

halu_downloaded = 0
halu_skipped = 0

for filename in HALUEVAL_FILES:
    dest_path = HALUEVAL_DIR / filename
    if dest_path.exists():
        print(f"  跳过: {filename} (已存在)")
        halu_skipped += 1
        continue
    url = f"{GITHUB_RAW_BASE}/halueval/{filename}"
    if download_file(url, dest_path):
        halu_downloaded += 1

print(f"\nHaluEval 下载完成: {halu_downloaded} 个新文件, {halu_skipped} 个已存在")

# 总结
print("\n" + "="*80)
print("✓ 所有数据集下载完成!")
print(f"数据目录: {DATA_DIR.absolute()}")
print(f"  BBQ: {len(BBQ_FILES)} 个文件")
print(f"  HaluEval: {len(HALUEVAL_FILES)} 个文件")
print("="*80)
