#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CFPB 投诉数据的清洗规则（唯一实现，供多处复用）。

``src/prototype.py``（样本上的基线原型）与 ``src/split_temporal.py``（全量时间划分）
都从这里取清洗逻辑，避免两份实现各自漂移。清洗包含四件事：

1. 只保留三个目标类别（其余 In progress / Untimely response 等全部剔除）；
2. 按 Complaint ID 去重，保留第一次出现的记录；
3. Product 旧命名归并（CFPB 改过产品分类，历史数据里同一产品有多个名字）；
4. State 与 Submitted via 的空值填成 ``"Unknown"``，并去掉首尾空白。

数据规模提示：``clean_data`` 会在内存里整表操作，1600 万行、8 列字符串时峰值约
2.5 GB。跑全量数据请分侧（训练/测试）依次处理，并在读入时把类别列转成
``category``（见 ``split_temporal.py`` 的 ``load_raw``）。
"""

from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

TARGET = "Company response to consumer"

# 三个目标类别（顺序即报告与混淆矩阵中的顺序）
TARGET_CLASSES = [
    "Closed with explanation",
    "Closed with non-monetary relief",
    "Closed with monetary relief",
]

# 用于分类的结构化特征
FEATURES = ["Product", "Issue", "Company", "State", "Submitted via"]

# 需要把空值填充为 "Unknown" 的列
FILL_UNKNOWN_COLUMNS = ["State", "Submitted via"]

# 数据清洗所必需的列
REQUIRED_COLUMNS = ["Complaint ID", TARGET] + FEATURES

# Product 旧命名归一化：{出现过的名称（小写）: (统一后的名称, 合并理由)}
# CFPB 的产品分类经过多次调整，历史数据里同一产品存在多个名称，需归并后再建模。
PRODUCT_NORMALIZATION = {
    "credit reporting": (
        "Credit reporting or other personal consumer reports",
        "旧命名，即现在的征信类产品",
    ),
    "credit reporting, credit repair services, or other personal consumer reports": (
        "Credit reporting or other personal consumer reports",
        "旧命名（含 credit repair services）",
    ),
    "bank account or service": (
        "Checking or savings account",
        "旧命名，对应现在的活期/储蓄账户",
    ),
    "credit card or prepaid card": (
        "Credit card",
        "旧的「信用卡或预付卡」合并命名，主体为信用卡",
    ),
    "payday loan": (
        "Payday loan, title loan, personal loan, or advance loan",
        "旧命名，现在的短期贷款类产品",
    ),
    "payday loan, title loan, or personal loan": (
        "Payday loan, title loan, personal loan, or advance loan",
        "旧命名，现在的短期贷款类产品",
    ),
    "money transfers": (
        "Money transfer, virtual currency, or money service",
        "旧命名，现在的转账/货币服务类产品",
    ),
}

MAPPING_COLUMNS = ["原名称", "归一化后", "影响行数", "合并理由"]


# --------------------------------------------------------------------------- #
# 校验辅助
# --------------------------------------------------------------------------- #

def require_columns(df: pd.DataFrame, columns: list[str], where: str) -> None:
    """缺少必需列时抛出清晰的错误。"""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{where}：缺少必需列 {missing}；实际列名 = {list(df.columns)}"
        )


def require_not_empty(df: pd.DataFrame, where: str) -> None:
    """输入为空时抛出清晰的错误。"""
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError(f"{where}：需要 pandas.DataFrame，实际收到 {type(df).__name__}")
    if df.empty:
        raise ValueError(f"{where}：输入 DataFrame 为空，无法继续")


# --------------------------------------------------------------------------- #
# Product 归一化
# --------------------------------------------------------------------------- #

def normalize_product(series: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """按 PRODUCT_NORMALIZATION 归并 Product 旧命名。

    返回 (归一化后的列, 合并映射表)。映射表只包含本次数据中真实命中、
    且名称确实发生变化的条目。
    """
    keys = series.astype(str).str.strip().str.lower()
    mapped = keys.map({k: v[0] for k, v in PRODUCT_NORMALIZATION.items()})
    normalized = mapped.where(mapped.notna(), series)

    # 统计每条映射命中的行数，生成可打印的映射表
    hit = normalized.ne(series) & mapped.notna()
    mapping_rows = []
    for name in pd.unique(series[hit].astype(str)):
        target, reason = PRODUCT_NORMALIZATION[name.strip().lower()]
        mapping_rows.append({
            "原名称": name,
            "归一化后": target,
            "影响行数": int((series.astype(str) == name).sum()),
            "合并理由": reason,
        })
    # 没有任何旧命名命中时也要返回带列名的空表，否则后续按列名排序会报错
    mapping = (
        pd.DataFrame(mapping_rows, columns=MAPPING_COLUMNS)
        .sort_values("影响行数", ascending=False)
        .reset_index(drop=True)
    )
    return normalized, mapping


# --------------------------------------------------------------------------- #
# 主清洗流程
# --------------------------------------------------------------------------- #

def clean_data(
    df: pd.DataFrame,
    *,
    dedupe: bool = True,
    label: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """清洗：只保留三个目标类别、按 Complaint ID 去重、归一化 Product、填充空值。

    参数：
        dedupe: 是否按 Complaint ID 去重。全量数据在 SQL 阶段已去过重时可关掉，
                避免在 1600 万行上再做一次 hash。
        label:  打印时的前缀（例如 "训练集"），便于区分同一流程里跑了两遍。

    返回 (清洗后的 DataFrame, Product 合并映射表)。
    """
    require_not_empty(df, "clean_data")
    require_columns(df, REQUIRED_COLUMNS, "clean_data")

    prefix = f"  [{label}] " if label else "  "

    # --- 2.1 只保留三个目标类别 ---
    # 注意：布尔索引本身就会 copy，这里不再额外 .copy()，
    # 否则在 1600 万行上会白白多占一份内存。
    n0 = len(df)
    keep = df[TARGET].isin(TARGET_CLASSES)
    # categorical 的 value_counts 会把计数为 0 的类别也列出来，这里滤掉，只打真实的剔除量
    dropped = df.loc[~keep, TARGET].value_counts()
    dropped = dropped[dropped > 0]
    out = df[keep]
    print(f"{prefix}只保留三个目标类别：剔除 {n0 - len(out):,} 行")
    for name, cnt in dropped.items():
        print(f"{prefix}    剔除 {name}: {cnt:,} 行")
    if out.empty:
        raise ValueError(
            f"clean_data：筛选目标类别后没有剩余数据；"
            f"请检查 {TARGET} 的取值是否包含 {TARGET_CLASSES}"
        )

    # --- 2.2 按 Complaint ID 去重 ---
    if dedupe:
        before = len(out)
        out = out.drop_duplicates(subset=["Complaint ID"], keep="first").reset_index(drop=True)
        print(f"{prefix}按 Complaint ID 去重：删除 {before - len(out):,} 行重复记录")
    else:
        print(f"{prefix}按 Complaint ID 去重：已在上游完成，跳过")

    # --- 2.3 Product 名称归一化 ---
    n_categories_before = out["Product"].nunique()
    normalized, mapping = normalize_product(out["Product"])
    out["Product"] = normalized
    print(f"{prefix}Product 名称归一化：{len(mapping)} 条旧命名被合并，"
          f"类别数 {n_categories_before} → {out['Product'].nunique()}")

    # --- 2.4 空值填充 ---
    for col in FILL_UNKNOWN_COLUMNS:
        blank = out[col].isna() | (out[col].astype(str).str.strip() == "")
        n_blank = int(blank.sum())
        out[col] = out[col].astype(str).str.strip()
        out.loc[blank, col] = "Unknown"
        print(f"{prefix}{col} 空值填充为 \"Unknown\"：{n_blank:,} 行")

    print(f"{prefix}清洗完成：{len(out):,} 行 × {out.shape[1]} 列")
    return out, mapping
