#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CFPB 投诉处理结果三分类 —— 基线原型。

任务：用投诉的结构化特征（Product, Issue, Company, State, Submitted via）
预测处理结果（Company response to consumer），只保留三个类别：

    Closed with explanation / Closed with non-monetary relief / Closed with monetary relief

在项目根目录下运行::

    python src/prototype.py

脚本会完成：加载样本 → 清洗 → 80/20 分层划分 → 两个基线（多数类 / 查表）
→ 评估并对比（宏平均 F1、各类 precision/recall/F1、混淆矩阵）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = "data/sample.csv"

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

# 混淆矩阵 / 报告中使用的类别简称
SHORT_LABELS = {
    "Closed with explanation": "explanation",
    "Closed with non-monetary relief": "non-monetary",
    "Closed with monetary relief": "monetary",
}

RANDOM_SEED = 42
TEST_SIZE = 0.2

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


# --------------------------------------------------------------------------- #
# 校验辅助
# --------------------------------------------------------------------------- #

def _require_columns(df: pd.DataFrame, columns: list[str], where: str) -> None:
    """缺少必需列时抛出清晰的错误。"""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{where}：缺少必需列 {missing}；实际列名 = {list(df.columns)}"
        )


def _require_not_empty(df: pd.DataFrame, where: str) -> None:
    """输入为空时抛出清晰的错误。"""
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError(f"{where}：需要 pandas.DataFrame，实际收到 {type(df).__name__}")
    if df.empty:
        raise ValueError(f"{where}：输入 DataFrame 为空，无法继续")


# --------------------------------------------------------------------------- #
# 1. 加载
# --------------------------------------------------------------------------- #

def load_data(path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    """读取样本 CSV，并校验必需列。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    df = pd.read_csv(path, dtype=str, encoding="utf-8")
    _require_not_empty(df, f"加载 {path.name}")
    _require_columns(df, REQUIRED_COLUMNS, f"加载 {path.name}")

    print(f"  文件: {path}")
    print(f"  规模: {len(df):,} 行 × {df.shape[1]} 列")
    return df


# --------------------------------------------------------------------------- #
# 2. 清洗
# --------------------------------------------------------------------------- #

def _normalize_product(series: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """按 PRODUCT_NORMALIZATION 归并 Product 旧命名。

    返回 (归一化后的列, 合并映射表)。映射表只包含本次数据中真实命中、
    且名称确实发生变化的条目。
    """
    keys = series.astype(str).str.strip().str.lower()
    mapped = keys.map({k: v[0] for k, v in PRODUCT_NORMALIZATION.items()})
    normalized = mapped.where(mapped.notna(), series)

    # 统计每条映射命中的行数，生成可打印的映射表
    hit = normalized.ne(series) & mapped.notna()
    mapping_columns = ["原名称", "归一化后", "影响行数", "合并理由"]
    rows = []
    for name in pd.unique(series[hit].astype(str)):
        target, reason = PRODUCT_NORMALIZATION[name.strip().lower()]
        rows.append({
            "原名称": name,
            "归一化后": target,
            "影响行数": int((series.astype(str) == name).sum()),
            "合并理由": reason,
        })
    # 没有任何旧命名命中时也要返回带列名的空表，否则后续按列名排序会报错
    mapping = (
        pd.DataFrame(rows, columns=mapping_columns)
        .sort_values("影响行数", ascending=False)
        .reset_index(drop=True)
    )
    return normalized, mapping


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """清洗：只保留三个目标类别、按 Complaint ID 去重、归一化 Product、填充空值。

    返回 (清洗后的 DataFrame, Product 合并映射表)。
    """
    _require_not_empty(df, "clean_data")
    _require_columns(df, REQUIRED_COLUMNS, "clean_data")

    out = df.copy()
    n0 = len(out)

    # --- 2.1 只保留三个目标类别 ---
    dropped = out.loc[~out[TARGET].isin(TARGET_CLASSES), TARGET].value_counts()
    out = out[out[TARGET].isin(TARGET_CLASSES)].copy()
    print(f"  只保留三个目标类别：剔除 {n0 - len(out):,} 行")
    for name, cnt in dropped.items():
        print(f"      剔除 {name}: {cnt:,} 行")
    if out.empty:
        raise ValueError(
            f"clean_data：筛选目标类别后没有剩余数据；"
            f"请检查 {TARGET} 的取值是否包含 {TARGET_CLASSES}"
        )

    # --- 2.2 按 Complaint ID 去重 ---
    before = len(out)
    out = out.drop_duplicates(subset=["Complaint ID"], keep="first").reset_index(drop=True)
    print(f"  按 Complaint ID 去重：删除 {before - len(out):,} 行重复记录")

    # --- 2.3 Product 名称归一化 ---
    n_categories_before = out["Product"].nunique()
    normalized, mapping = _normalize_product(out["Product"])
    out["Product"] = normalized
    print(f"  Product 名称归一化：{len(mapping)} 条旧命名被合并，"
          f"类别数 {n_categories_before} → {out['Product'].nunique()}")

    # --- 2.4 空值填充 ---
    for col in FILL_UNKNOWN_COLUMNS:
        blank = out[col].isna() | (out[col].astype(str).str.strip() == "")
        n_blank = int(blank.sum())
        out[col] = out[col].astype(str).str.strip()
        out.loc[blank, col] = "Unknown"
        print(f"  {col} 空值填充为 \"Unknown\"：{n_blank:,} 行")

    print(f"  清洗完成：{len(out):,} 行 × {out.shape[1]} 列")
    return out, mapping


# --------------------------------------------------------------------------- #
# 3. 划分
# --------------------------------------------------------------------------- #

def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """按 seed 做 80/20 分层划分（以目标变量分层）。"""
    _require_not_empty(df, "split_data")
    _require_columns(df, FEATURES + [TARGET], "split_data")

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=test_size,
        random_state=seed,
        stratify=df[TARGET],
    )

    print(f"  随机种子 {seed}，测试集比例 {test_size:.0%}（按 {TARGET} 分层）")
    print(f"  训练集 {len(X_train):,} 行 | 测试集 {len(X_test):,} 行")
    dist = pd.DataFrame({
        "训练集": y_train.value_counts(),
        "测试集": y_test.value_counts(),
    }).reindex(TARGET_CLASSES)
    dist["训练集占比"] = (dist["训练集"] / len(y_train)).map("{:.2%}".format)
    dist["测试集占比"] = (dist["测试集"] / len(y_test)).map("{:.2%}".format)
    print(dist.to_string())
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# 4. 基线
# --------------------------------------------------------------------------- #

def _majority_class(y: pd.Series) -> str:
    """返回多数类；数量并列时按类别名排序取第一个，保证结果确定。"""
    counts = y.value_counts()
    top = counts[counts == counts.max()].index
    return sorted(top)[0]


def baseline_majority(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    """基线 1：永远预测训练集中的多数类。"""
    _require_not_empty(X_test, "baseline_majority")
    if y_train.empty:
        raise ValueError("baseline_majority：训练标签 y_train 为空")
    return np.full(len(X_test), _majority_class(y_train), dtype=object)


def baseline_lookup(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    key_columns: tuple[str, ...] = ("Product", "Issue"),
) -> np.ndarray:
    """基线 2：用训练集统计「每个 key 组合」最常见的处理结果作为预测。

    默认 key 为 Product × Issue 组合；训练集中没出现过的组合，
    以及公司的取值在训练集中未出现过而导致的未见组合，都回退到全局多数类。

    传入 key_columns=("Product", "Issue", "Company") 即可把公司维度也纳入组合键。
    """
    _require_not_empty(X_train, "baseline_lookup")
    _require_not_empty(X_test, "baseline_lookup")
    _require_columns(X_train, list(key_columns), "baseline_lookup(X_train)")
    _require_columns(X_test, list(key_columns), "baseline_lookup(X_test)")
    if y_train.empty:
        raise ValueError("baseline_lookup：训练标签 y_train 为空")

    global_majority = _majority_class(y_train)

    # 组合键里的空值统一占位，否则空值之间无法相互匹配。
    # 注意：必须重置索引！X_train 的索引在 train_test_split 后已被打乱，
    # 若直接与 y_train 拼接会按索引标签对齐，造成特征与标签错位。
    keys = list(key_columns)
    train_keys = X_train[keys].astype(str).replace({"nan": "<missing>"}).reset_index(drop=True)
    test_keys = X_test[keys].astype(str).replace({"nan": "<missing>"}).reset_index(drop=True)

    # 统计每个组合下各类别的出现次数，按次数降序（并列时按类别名排序）取第一个
    counts = (
        train_keys.assign(**{TARGET: y_train.reset_index(drop=True)})
        .groupby(keys + [TARGET], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["n", TARGET], ascending=[False, True])
    )
    best = counts.drop_duplicates(subset=keys).set_index(keys)[TARGET]

    # 用组合键对齐测试集：未命中的组合为 NaN，随后回退到全局多数类
    preds = best.reindex(pd.MultiIndex.from_frame(test_keys))
    n_fallback = int(preds.isna().sum())

    unseen_company = int((~X_test["Company"].isin(set(X_train["Company"]))).sum())
    print(f"  组合键 {keys}：训练集学到 {len(best):,} 个组合")
    print(f"  测试集命中组合 {len(preds) - n_fallback:,} 行"
          f"（{(len(preds) - n_fallback) / len(preds):.2%}），"
          f"未见组合回退到全局多数类 {global_majority!r}：{n_fallback:,} 行")
    print(f"    其中多数类不等于全局多数类的组合只有 {int((best != global_majority).sum()):,} 个，"
          f"其余组合的预测与多数类基线完全相同")
    print(f"    参考：测试集中公司未在训练集出现过的行 {unseen_company:,} 行"
          f"（Company {'已' if 'Company' in key_columns else '未'}纳入组合键）")

    return preds.fillna(global_majority).to_numpy(dtype=object)


# --------------------------------------------------------------------------- #
# 5. 评估
# --------------------------------------------------------------------------- #

def evaluate(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """打印单个基线的宏平均 F1、各类 precision/recall/F1 与混淆矩阵。"""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"evaluate：y_true 与 y_pred 长度不一致（{len(y_true)} vs {len(y_pred)}）"
        )

    report = classification_report(
        y_true, y_pred, labels=TARGET_CLASSES, zero_division=0, digits=4
    )
    cm = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES)
    macro_f1 = f1_score(y_true, y_pred, labels=TARGET_CLASSES, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        y_true, y_pred, labels=TARGET_CLASSES, average=None, zero_division=0
    )

    print(f"\n--- {name} ---")
    print(f"宏平均 F1 = {macro_f1:.4f}")
    print(report.rstrip())

    short = [SHORT_LABELS[c] for c in TARGET_CLASSES]
    cm_df = pd.DataFrame(cm, index=[f"真实:{s}" for s in short],
                         columns=[f"预测:{s}" for s in short])
    print("混淆矩阵（行 = 真实，列 = 预测）")
    print(cm_df.to_string())

    pred_counts = pd.Series(y_pred).value_counts(normalize=True)
    return {
        "baseline": name,
        "宏平均F1": round(macro_f1, 4),
        **{f"F1[{SHORT_LABELS[c]}]": round(f, 4) for c, f in zip(TARGET_CLASSES, per_class_f1)},
        "准确率": round(float((y_true.to_numpy() == y_pred).mean()), 4),
        "预测最集中": SHORT_LABELS.get(pred_counts.index[0], str(pred_counts.index[0])),
        "该类别占比": round(float(pred_counts.iloc[0]), 4),
    }


# --------------------------------------------------------------------------- #
# 6. 入口
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="CFPB 投诉处理结果三分类基线原型")
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"输入 CSV（默认 {DEFAULT_INPUT}）")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    line = "=" * 72
    print(line)
    print("CFPB 投诉处理结果三分类 —— 基线原型")
    print(line)

    print("\n[1/4] 加载数据")
    df = load_data(input_path)

    print("\n[2/4] 数据清洗")
    df_clean, mapping = clean_data(df)
    print("\n  Product 旧命名合并映射表：")
    if mapping.empty:
        print("      （本次数据中没有出现需要合并的旧命名）")
    else:
        print("  " + mapping.to_string(index=False).replace("\n", "\n  "))

    print(f"\n  清洗后 {TARGET} 分布：")
    dist = df_clean[TARGET].value_counts().reindex(TARGET_CLASSES)
    for name, cnt in dist.items():
        print(f"      {name:<34s} {cnt:>7,}  {cnt / len(df_clean):>7.2%}")

    print("\n[3/4] 划分训练集 / 测试集")
    X_train, X_test, y_train, y_test = split_data(df_clean)

    print("\n[4/4] 训练并评估基线")
    results = []

    pred = baseline_majority(X_train, y_train, X_test)
    results.append(evaluate("baseline_majority（永远预测多数类）", y_test, pred))

    pred = baseline_lookup(X_train, y_train, X_test, key_columns=("Product", "Issue"))
    results.append(evaluate("baseline_lookup（Product × Issue 组合）", y_test, pred))

    pred = baseline_lookup(
        X_train, y_train, X_test, key_columns=("Product", "Issue", "Company")
    )
    results.append(evaluate("baseline_lookup 变体（Product × Issue × Company）", y_test, pred))

    print("\n" + line)
    print("基线对比（宏平均 F1 越大越好；最后一行为把公司维度纳入组合键的变体）")
    print(line)
    comparison = pd.DataFrame(results).set_index("baseline")
    print(comparison.to_string())
    print("\n" + line)

    best = comparison["宏平均F1"].idxmax()
    print(f"宏平均 F1 最高：{best}（{comparison.loc[best, '宏平均F1']}）")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
