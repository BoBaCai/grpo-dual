#!/usr/bin/env python3
import argparse
from src.judges.runner import run_judges

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to judge config YAML")
    args = ap.parse_args()
    run_judges(args.config)
