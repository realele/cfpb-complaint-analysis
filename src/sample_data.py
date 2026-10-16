#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CFPB 消费者投诉数据 —— 分块读取与随机抽样。

在项目根目录下运行::

    python src/sample_data.py

原始文件 data/complaints.csv 约 8.4 GB，无法一次性载入内存，因此全程使用
pandas 的 ``chunksize`` 分块读取，内存占用恒定。

处理流程：
  1. 只读取分析所需的 9 个字段，跳过 Consumer complaint narrative 等长文本字段；
  2. 丢弃 Company response to consumer 或 Product 为空的行；
  3. 以固定随机种子 42 随机抽样约 20 万行，保存为 data/sample.csv；
  4. 打印原始总行数、抽样行数以及抽样结果的基本分布。

注意：抽样比例 = 目标行数 / 有效总行数，需要先知道全量有效行数，
因此脚本会对文件读取两遍（第 1 遍只统计行数，第 2 遍按比例抽样）。
"""

from __future__ import annotations

import argparse
import random
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = "data/complaints.csv"   # 相对路径，基于项目根目录
OUTPUT_PATH = "data/sample.csv"

# 只读取这些列，避免读入超长的 Consumer complaint narrative，节省内存和时间
USECOLS = [
    "Date received",
    "Product",
    "Issue",
    "Company",
    "State",
    "Submitted via",
    "Company response to consumer",
    "Timely response?",
    "Complaint ID",
]

# 这些字段为空的行直接丢弃
REQUIRED_COLUMNS = ["Company response to consumer", "Product"]

CHUNK_SIZE = 200_000          # 每个分块的行数
TARGET_SAMPLE_ROWS = 200_000  # 目标抽样行数（约 20 万）
RANDOM_SEED = 42              # 固定随机种子，保证结果可复现


# --------------------------------------------------------------------------- #
# 分块读取
# --------------------------------------------------------------------------- #

def iter_chunks(path: Path, chunksize: int = CHUNK_SIZE):
    """按分块迭代读取 CSV，只取 USECOLS 指定的列。"""
    return pd.read_csv(
        path,
        usecols=USECOLS,
        dtype=str,
        chunksize=chunksize,
        low_memory=False,
        encoding="utf-8",
    )


def count_rows(path: Path) -> tuple[int, int]:
    """第 1 遍：统计原始总行数与清洗后的有效行数。"""
    total = 0
    valid = 0
    for i, chunk in enumerate(iter_chunks(path), start=1):
        total += len(chunk)
        valid += int(chunk[REQUIRED_COLUMNS].notna().all(axis=1).sum())
        if i % 20 == 0:
            print(f"  [第 1 遍] 已读取 {total:,} 行 ...", file=sys.stderr, flush=True)
    return total, valid


# --------------------------------------------------------------------------- #
# 抽样
# --------------------------------------------------------------------------- #

def draw_sample(path: Path, fraction: float) -> pd.DataFrame:
    """第 2 遍：按 fraction 比例在每个分块内随机抽样。

    每个分块使用由固定种子 42 派生出的不同随机数种子，
    既保证整体结果可复现，又避免"每块都抽同样的位置"造成系统性偏差。
    """
    fraction = min(1.0, max(0.0, fraction))
    rng = random.Random(RANDOM_SEED)
    parts: list[pd.DataFrame] = []
    taken = 0

    for i, chunk in enumerate(iter_chunks(path), start=1):
        chunk = chunk.dropna(subset=REQUIRED_COLUMNS)
        if not chunk.empty:
            take = min(len(chunk), max(1, round(len(chunk) * fraction)))
            parts.append(chunk.sample(n=take, random_state=rng.randint(0, 2 ** 31 - 1)))
            taken += take
        if i % 20 == 0:
            print(
                f"  [第 2 遍] 已读取 {i * CHUNK_SIZE:,} 行，累计抽中 {taken:,} 行 ...",
                file=sys.stderr,
                flush=True,
            )

    if not parts:
        return pd.DataFrame(columns=USECOLS)

    sample = pd.concat(parts, ignore_index=True)

    # 各分块分别取整可能略微超出目标行数，这里随机裁到目标行数（保持原文件顺序）
    if len(sample) > TARGET_SAMPLE_ROWS:
        keep = sorted(rng.sample(range(len(sample)), TARGET_SAMPLE_ROWS))
        sample = sample.iloc[keep].reset_index(drop=True)

    return sample


# --------------------------------------------------------------------------- #
# 报告输出
# --------------------------------------------------------------------------- #

def _disp_width(text: str) -> int:
    """终端显示宽度：中日韩全角字符占 2 列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _disp_width(text))


def _print_distribution(title: str, series: pd.Series, top: int | None = None) -> None:
    """打印一列取值的数量与占比（按数量降序）。"""
    counts = series.value_counts(dropna=False)
    if top is not None:
        counts = counts.head(top)

    labels = ["<缺失>" if pd.isna(x) else str(x) for x in counts.index]
    labels = [lb if len(lb) <= 58 else lb[:55] + "..." for lb in labels]
    width = max([_disp_width(lb) for lb in labels] + [_disp_width("类别")])
    total = len(series)

    print(f"\n{title}")
    print(_pad("类别", width) + f"  {'数量':>12}  {'占比':>9}")
    print("-" * (width + 26))
    for label, count in zip(labels, counts.to_numpy()):
        print(_pad(label, width) + f"  {count:>12,}  {count / total:>8.2%}")


def print_report(total_rows: int, valid_rows: int, sample: pd.DataFrame) -> None:
    n = len(sample)
    line = "=" * 68

    print(line)
    print("CFPB 消费者投诉数据 —— 抽样报告")
    print(line)
    print(f"原始文件总行数（不含表头） : {total_rows:,}")
    dropped = total_rows - valid_rows
    share = f"，占 {dropped / total_rows:.2%}" if total_rows else ""
    print(f"清洗后有效行数             : {valid_rows:,}（丢弃空值 {dropped:,} 行{share}）")
    print(f"抽样行数                   : {n:,}")
    if valid_rows:
        print(f"抽样比例                   : {n / valid_rows:.4%}（随机种子 {RANDOM_SEED}）")

    _print_distribution("Company response to consumer 分布", sample["Company response to consumer"])

    _print_distribution(
        f"Product 前 10 个类别分布（样本共 {sample['Product'].nunique()} 个类别）",
        sample["Product"],
        top=10,
    )

    # 注意：Date received 在同一列里混用了两种格式（"YYYY-MM-DD" 以及带时间戳的
    # "YYYY-MM-DDTHH:MM:SS.mmmZ"），pandas 默认会按第一个值推断整体格式，
    # 从而把另一种格式全部判为无法解析。这里显式指定 format="mixed" 逐元素解析。
    dates = pd.to_datetime(sample["Date received"], format="mixed", errors="coerce", utc=True)
    bad = int(dates.isna().sum())
    print("\nDate received 时间范围")
    print("-" * 40)
    if dates.notna().any():
        print(f"最早 : {dates.min().date()}")
        print(f"最晚 : {dates.max().date()}")
    else:
        print("无可解析的日期")
    if bad:
        print(f"（另有 {bad:,} 行日期无法解析）")

    print("\n" + line)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    # 让标准输出按行刷新，避免输出重定向到文件时进度信息与报告顺序错乱
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="CFPB 投诉数据分块读取与随机抽样")
    parser.add_argument("--input", default=INPUT_PATH, help=f"输入 CSV（默认 {INPUT_PATH}）")
    parser.add_argument("--output", default=OUTPUT_PATH, help=f"输出 CSV（默认 {OUTPUT_PATH}）")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    if not input_path.exists():
        print(f"找不到输入文件：{input_path}", file=sys.stderr)
        return 1

    size_gb = input_path.stat().st_size / 1024 ** 3
    print(f"输入文件 : {input_path}（{size_gb:.2f} GB）")
    print(f"输出文件 : {output_path}")
    print("开始第 1 遍读取（统计总行数）...")

    total_rows, valid_rows = count_rows(input_path)
    if valid_rows == 0:
        print("清洗后没有有效数据，请检查 REQUIRED_COLUMNS 设置。", file=sys.stderr)
        return 1

    fraction = min(1.0, TARGET_SAMPLE_ROWS / valid_rows)
    print(f"开始第 2 遍读取（抽样比例 {fraction:.6f}）...")
    sample = draw_sample(input_path, fraction)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False, encoding="utf-8")
    print(f"样本已保存：{output_path}（{output_path.stat().st_size / 1024 ** 2:.1f} MB）")

    print_report(total_rows, valid_rows, sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
