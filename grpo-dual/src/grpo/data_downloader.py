# -*- coding: utf-8 -*-
"""
数据集自动下载模块
从 GitHub 仓库下载 BBQ 和 HaluEval 数据集
"""

import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List
import time


class DatasetDownloader:
    """从 GitHub 下载数据集"""

    # GitHub raw 文件 URL 基础路径
    GITHUB_RAW_BASE = "https://raw.githubusercontent.com/BoBaCai/grpo-dual/main/grpo-dual/data"

    # BBQ 数据集文件列表
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

    # HaluEval 数据集文件列表
    HALUEVAL_FILES = [
        "dialogue_data.json",
        "general_data.json",
        "qa_data.json",
        "summarization_data.json",
    ]

    def __init__(self, data_dir: Path):
        """
        初始化下载器

        Args:
            data_dir: 数据存储根目录
        """
        self.data_dir = Path(data_dir)
        self.bbq_dir = self.data_dir / "bbq"
        self.halueval_dir = self.data_dir / "halueval"

        # 创建目录
        self.bbq_dir.mkdir(parents=True, exist_ok=True)
        self.halueval_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, dest_path: Path, max_retries: int = 3) -> bool:
        """
        下载单个文件，支持重试

        Args:
            url: 文件 URL
            dest_path: 目标路径
            max_retries: 最大重试次数

        Returns:
            bool: 下载是否成功
        """
        for attempt in range(max_retries):
            try:
                print(f"  下载: {dest_path.name} ... ", end="", flush=True)

                # 下载文件
                urllib.request.urlretrieve(url, dest_path)

                # 验证文件存在且非空
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

            # 重试前等待
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                print(f"    等待 {wait_time}s 后重试...")
                time.sleep(wait_time)

        return False

    def download_bbq(self, force: bool = False) -> Dict[str, bool]:
        """
        下载 BBQ 数据集

        Args:
            force: 是否强制重新下载

        Returns:
            Dict[str, bool]: 文件名 -> 下载结果
        """
        print(f"\n{'='*80}")
        print(f"下载 BBQ 数据集 ({len(self.BBQ_FILES)} 个文件)")
        print(f"目标目录: {self.bbq_dir}")
        print(f"{'='*80}")

        results = {}
        downloaded = 0
        skipped = 0

        for filename in self.BBQ_FILES:
            dest_path = self.bbq_dir / filename

            # 检查是否已存在
            if dest_path.exists() and not force:
                print(f"  跳过: {filename} (已存在)")
                results[filename] = True
                skipped += 1
                continue

            # 构建 URL
            url = f"{self.GITHUB_RAW_BASE}/bbq/{filename}"

            # 下载
            success = self.download_file(url, dest_path)
            results[filename] = success

            if success:
                downloaded += 1

        print(f"\nBBQ 下载完成: {downloaded} 个新文件, {skipped} 个已存在")
        return results

    def download_halueval(self, force: bool = False) -> Dict[str, bool]:
        """
        下载 HaluEval 数据集

        Args:
            force: 是否强制重新下载

        Returns:
            Dict[str, bool]: 文件名 -> 下载结果
        """
        print(f"\n{'='*80}")
        print(f"下载 HaluEval 数据集 ({len(self.HALUEVAL_FILES)} 个文件)")
        print(f"目标目录: {self.halueval_dir}")
        print(f"{'='*80}")

        results = {}
        downloaded = 0
        skipped = 0

        for filename in self.HALUEVAL_FILES:
            dest_path = self.halueval_dir / filename

            # 检查是否已存在
            if dest_path.exists() and not force:
                print(f"  跳过: {filename} (已存在)")
                results[filename] = True
                skipped += 1
                continue

            # 构建 URL
            url = f"{self.GITHUB_RAW_BASE}/halueval/{filename}"

            # 下载
            success = self.download_file(url, dest_path)
            results[filename] = success

            if success:
                downloaded += 1

        print(f"\nHaluEval 下载完成: {downloaded} 个新文件, {skipped} 个已存在")
        return results

    def download_all(self, force: bool = False) -> bool:
        """
        下载所有数据集

        Args:
            force: 是否强制重新下载

        Returns:
            bool: 所有文件是否下载成功
        """
        print(f"\n{'='*80}")
        print(f"开始下载所有数据集")
        print(f"数据目录: {self.data_dir}")
        print(f"{'='*80}")

        bbq_results = self.download_bbq(force=force)
        halueval_results = self.download_halueval(force=force)

        # 检查结果
        all_success = all(bbq_results.values()) and all(halueval_results.values())

        print(f"\n{'='*80}")
        if all_success:
            print(f"✓ 所有数据集下载成功!")
        else:
            print(f"⚠️ 部分文件下载失败:")
            for filename, success in {**bbq_results, **halueval_results}.items():
                if not success:
                    print(f"  ✗ {filename}")
        print(f"{'='*80}\n")

        return all_success

    def check_datasets(self) -> Dict[str, Dict[str, bool]]:
        """
        检查数据集文件是否存在

        Returns:
            Dict: 数据集名 -> {文件名 -> 是否存在}
        """
        bbq_status = {f: (self.bbq_dir / f).exists() for f in self.BBQ_FILES}
        halueval_status = {f: (self.halueval_dir / f).exists() for f in self.HALUEVAL_FILES}

        return {
            "bbq": bbq_status,
            "halueval": halueval_status,
        }

    def print_status(self):
        """打印数据集状态"""
        status = self.check_datasets()

        print(f"\n{'='*80}")
        print(f"数据集状态检查")
        print(f"{'='*80}")

        # BBQ
        bbq_total = len(status["bbq"])
        bbq_exists = sum(status["bbq"].values())
        print(f"\nBBQ 数据集: {bbq_exists}/{bbq_total} 个文件存在")
        for filename, exists in status["bbq"].items():
            symbol = "✓" if exists else "✗"
            print(f"  {symbol} {filename}")

        # HaluEval
        halu_total = len(status["halueval"])
        halu_exists = sum(status["halueval"].values())
        print(f"\nHaluEval 数据集: {halu_exists}/{halu_total} 个文件存在")
        for filename, exists in status["halueval"].items():
            symbol = "✓" if exists else "✗"
            print(f"  {symbol} {filename}")

        print(f"\n{'='*80}\n")

        return bbq_exists == bbq_total and halu_exists == halu_total


def ensure_datasets(data_dir: Path, force_download: bool = False) -> bool:
    """
    确保数据集存在，如果不存在则自动下载

    Args:
        data_dir: 数据目录
        force_download: 是否强制重新下载

    Returns:
        bool: 数据集是否准备就绪
    """
    downloader = DatasetDownloader(data_dir)

    # 检查现有文件
    if not force_download:
        all_exists = downloader.print_status()
        if all_exists:
            print("✓ 所有数据集文件已存在，跳过下载")
            return True

    # 下载缺失的文件
    print("\n开始下载缺失的数据集文件...")
    return downloader.download_all(force=force_download)


# 命令行使用示例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="下载 GRPO 数据集")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="数据存储目录 (默认: ./data)")
    parser.add_argument("--force", action="store_true",
                        help="强制重新下载所有文件")
    parser.add_argument("--check-only", action="store_true",
                        help="仅检查文件状态，不下载")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    downloader = DatasetDownloader(data_dir)

    if args.check_only:
        downloader.print_status()
    else:
        downloader.download_all(force=args.force)
