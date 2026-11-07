#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CFPB 时间划分基线重跑 + Bootstrap 置信区间。

在 ``data/train.parquet`` / ``data/test.parquet``（时间划分：训练 ≤ 2023-12-31 <
测试）上重跑阶段一的三个基线：

  1. 多数类基线（永远预测训练集多数类）
  2. Product × Issue 查表（组合键查多数类，未见组合回退全局多数类）
  3. Product × Issue × Company 查表（同上，键增加公司维度）

清洗规则复用 ``src/cleaning.py``——parquet 文件已由 ``split_temporal.py`` 用同一份
``clean_data`` 清洗后落盘，本脚本直接读清洗结果，不再重复清洗。评估格式与
``src/prototype.py`` 完全一致（宏平均 F1、各类 precision/recall/F1、混淆矩阵），
直接调用 ``prototype.evaluate`` 保证输出格式不漂移。

新增内容：

  - **Bootstrap 置信区间**：对宏平均 F1 在测试集上做有放回抽样 1000 次，
    取 2.5% / 97.5% 分位作为 95% CI。
  - **与阶段一对比**：把时间划分的 F1 与阶段一随机划分的结果
    （0.2592 / 0.2679 / 0.5226）并排打印，标注下降幅度。

内存优化（峰值 ≤ 4 GB）：

  1. 先加载训练集、构建全部查表后即 ``del`` + ``gc.collect``，再加载测试集，
     两侧不同时驻留；
  2. Bootstrap 不逐次重抽样 1100 万行，而是从混淆矩阵的 9 个格子做**多项分布**
     抽样——重抽样 N 个 (y_true, y_pred) 对后各格子计数服从 Multinomial(N, p)，
     与逐行 Bootstrap 在分布上完全等价，但每次迭代只需 9 个数的计算（实测
     1000 次 < 0.1 秒，而逐行重抽样需约 180 秒）；
  3. 大字符串列在用完即释放。

在项目根目录下运行::

    python src/baseline_temporal.py

输出 ``data/baseline_temporal_stats.json``（供报告 / EDA 引用）。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# cleaning.py 是清洗的唯一实现，本脚本复用其中的常量（TARGET / TARGET_CLASSES /
# FEATURES），保证列名与类别顺序与 prototype / split_temporal 一致。
from cleaning import FEATURES, TARGET, TARGET_CLASSES  # noqa: E402
# _majority_class 的平局打破规则复用 prototype 的实现，保证查表行为一致。
# evaluate 不直接复用 prototype.evaluate：后者在 1100 万行上调用 4 个 sklearn
# 函数（classification_report + confusion_matrix + f1_score×2），每个都对
# y_true/y_pred 做一次完整标签编码，合计约 178 秒；本脚本改用自己的
# evaluate_efficient，只调用一次 confusion_matrix（22 秒），其余指标从 3×3
# 矩阵推导，输出格式与 prototype.evaluate 完全一致。
from prototype import SHORT_LABELS, _majority_class  # noqa: E402
from report import print_table  # noqa: E402

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = "data/train.parquet"
TEST_PATH = "data/test.parquet"
STATS_JSON = "data/baseline_temporal_stats.json"

# 阶段一随机划分的宏平均 F1（src/prototype.py 在 data/sample.csv 上的结果）
PHASE1_RANDOM_F1 = {
    "majority": 0.2592,
    "product_issue": 0.2679,
    "product_issue_company": 0.5226,
}

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
CI_LOW_PCT = 2.5
CI_HIGH_PCT = 97.5

# 内存上限只用于报告，不做硬约束
MEMORY_BUDGET_GB = 4.0

_PEAK_RSS_MB = 0.0


def _sample_rss() -> float | None:
    """采样当前进程内存，并记录峰值（MB）。"""
    global _PEAK_RSS_MB
    if psutil is None:
        return None
    try:
        mb = psutil.Process().memory_info().rss / 1024 ** 2
    except Exception:  # pragma: no cover
        return None
    _PEAK_RSS_MB = max(_PEAK_RSS_MB, mb)
    return mb


# --------------------------------------------------------------------------- #
# 1. 宏平均 F1（从混淆矩阵直接算）
# --------------------------------------------------------------------------- #

def macro_f1_from_cm(cm: np.ndarray, n_classes: int = 3) -> float:
    """从混淆矩阵计算宏平均 F1。

    与 ``sklearn.metrics.f1_score(average="macro", zero_division=0)`` 完全一致：
    每个类的 F1 = 2·TP / (2·TP + FP + FN)，分母为 0 时该类 F1 = 0，
    最后对 n_classes 个类取算术平均。

    Bootstrap 每次迭代只需调用此函数（9 个数的计算），远快于在 1100 万行上
    重跑 sklearn。
    """
    macro = 0.0
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp      # 预测为 i 但真实不是 i
        fn = cm[i, :].sum() - tp      # 真实为 i 但预测不是 i
        denom = 2 * tp + fp + fn
        if denom > 0:
            macro += 2 * tp / denom
        # denom == 0 ⟹ F1 = 0（zero_division=0 的语义）
    return float(macro) / n_classes


# --------------------------------------------------------------------------- #
# 2. Bootstrap 置信区间
# --------------------------------------------------------------------------- #

def _encode_labels(labels, classes: list[str]) -> np.ndarray:
    """将字符串标签编码为 int8（按 classes 的顺序）。未知类别报错。"""
    mapping = {c: i for i, c in enumerate(classes)}
    encoded = pd.Series(labels).map(mapping)
    if encoded.isna().any():
        unknown = set(pd.Series(labels)[encoded.isna()].astype(str).unique())
        raise ValueError(
            f"_encode_labels：发现未知类别 {unknown}，合法类别为 {classes}"
        )
    return encoded.to_numpy(dtype=np.int8)


def bootstrap_macro_f1(
    y_true_codes: np.ndarray,
    y_pred_codes: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    n_classes: int = 3,
) -> tuple[np.ndarray, float, float]:
    """对宏平均 F1 做 Bootstrap 置信区间。

    测试集有放回抽样 ``n_boot`` 次，每次计算宏平均 F1，
    取 2.5% / 97.5% 分位作为 95% 置信区间。

    **实现优化**：不逐次重抽样 N 行，而是从混淆矩阵的 ``n_classes²`` 个格子做
    多项分布抽样。因为 Bootstrap 重抽样 N 个 (y_true, y_pred) 对后，混淆矩阵
    各格子的计数恰好服从 ``Multinomial(N, p)``，其中 ``p = 原始格子占比``。
    这与逐行 Bootstrap **在分布上完全等价**，但每次迭代只需一次多项分布抽样
    （numpy 用序贯二项法，对 k 个类别是 O(k) 而非 O(N)）+ 9 个数的 F1 计算。

    实测：N=11,160,958、1000 次迭代 < 0.1 秒；逐行重抽样则需约 180 秒。

    返回 ``(boot_f1s, lo, hi)``：1000 个 F1 值的数组，以及 2.5% / 97.5% 分位。
    """
    rng = np.random.default_rng(seed)
    N = len(y_true_codes)

    # 基准混淆矩阵（flatten 成 9 个格子）
    pair_codes = y_true_codes.astype(np.int64) * n_classes + y_pred_codes.astype(np.int64)
    base_counts = np.bincount(pair_codes, minlength=n_classes * n_classes)
    p = base_counts / N

    boot_f1s = np.empty(n_boot)
    for i in range(n_boot):
        counts = rng.multinomial(N, p)
        cm = counts.reshape(n_classes, n_classes)
        boot_f1s[i] = macro_f1_from_cm(cm, n_classes)

    lo = float(np.percentile(boot_f1s, CI_LOW_PCT))
    hi = float(np.percentile(boot_f1s, CI_HIGH_PCT))
    return boot_f1s, lo, hi


def bootstrap_macro_f1_direct(
    y_true_codes: np.ndarray,
    y_pred_codes: np.ndarray,
    n_boot: int,
    seed: int,
    n_classes: int = 3,
) -> tuple[np.ndarray, float, float]:
    """逐行重抽样的 Bootstrap（仅用于测试对照，验证多项分布版等价）。

    与 ``bootstrap_macro_f1`` 的区别：每次迭代真正生成 N 个随机索引、索引
    pair_codes、再 bincount。结果分布相同，但慢约 2000 倍。
    """
    rng = np.random.default_rng(seed)
    N = len(y_true_codes)
    pair_codes = y_true_codes.astype(np.int64) * n_classes + y_pred_codes.astype(np.int64)

    boot_f1s = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, N, size=N)
        counts = np.bincount(pair_codes[idx], minlength=n_classes * n_classes)
        cm = counts.reshape(n_classes, n_classes)
        boot_f1s[i] = macro_f1_from_cm(cm, n_classes)

    lo = float(np.percentile(boot_f1s, CI_LOW_PCT))
    hi = float(np.percentile(boot_f1s, CI_HIGH_PCT))
    return boot_f1s, lo, hi


# --------------------------------------------------------------------------- #
# 3. 查表基线（内存高效版，预测与 prototype.baseline_lookup 完全一致）
# --------------------------------------------------------------------------- #

def build_lookup(
    train_df: pd.DataFrame, key_cols: tuple[str, ...] | list[str]
) -> pd.Series:
    """从训练集构建查表：每个 key 组合 → 最常见的处理结果。

    平局按类别名字母序打破（与 ``prototype.baseline_lookup`` 的
    ``sort_values(["n", TARGET], ascending=[False, True])`` 一致）。

    返回一个 Series，索引为 key 组合的 MultiIndex，值为多数类字符串。

    **坑**：``train_df[TARGET].values`` 在 pandas 2.x 中对 category 列返回
    ``Categorical`` 对象（而非 numpy 数组），赋值后 TARGET 列保持 category
    dtype。此时 ``groupby(observed=False)``（默认值）会对**所有** grouper
    做全交叉积——15 Product × 178 Issue × 6767 Company × 3 Target ≈ 5400 万
    中间行，直接撑爆内存。两个修复：

      1. ``.astype(str)`` 把 TARGET 转成 object 字符串，消除 category grouper；
      2. ``observed=True`` 显式只保留观测到的组合，双保险。
    """
    keys = list(key_cols)
    # 与 prototype 一致：astype(str) 后把 "nan" 替换为 "<missing>"
    key_str = train_df[keys].astype(str).replace({"nan": "<missing>"})
    # 关键：用 .astype(str) 而非 .values，避免 TARGET 保持 category dtype
    key_str[TARGET] = train_df[TARGET].astype(str).values
    counts = (
        key_str.groupby(keys + [TARGET], dropna=False, observed=True)
        .size()
        .reset_index(name="n")
        .sort_values(["n", TARGET], ascending=[False, True])
    )
    best = counts.drop_duplicates(subset=keys).set_index(keys)[TARGET]
    return best


def apply_lookup(
    X_test: pd.DataFrame,
    key_cols: tuple[str, ...] | list[str],
    lookup: pd.Series,
    global_majority: str,
) -> tuple[np.ndarray, dict]:
    """将查表应用到测试集：未见组合回退到全局多数类。

    **内存优化**：不对 1100 万行测试集逐行 ``.astype(str)``（会产生 1.6 GB
    的 Python 字符串对象），而是：

      1. ``drop_duplicates`` 取测试集的唯一 key 组合（P×I 约 288 个、
         P×I×C 约 6 万个，远小于 1100 万）；
      2. 只对唯一组合做字符串转换 + 查表；
      3. 通过 ``merge`` 把预测映射回全部行（merge 在 category 列上走
         整数码哈希连接，高效且省内存）。

    返回 ``(predictions, stats)``，stats 包含组合数、命中/回退行数等。
    """
    keys = list(key_cols)

    # 1. 取唯一组合（drop_duplicates 只返回实际出现过的行，不展开 category 交叉积）
    test_unique = X_test[keys].drop_duplicates().reset_index(drop=True)

    # 2. 只对唯一组合做字符串转换 + 查表
    test_unique_str = test_unique.astype(str).replace({"nan": "<missing>"})
    preds_unique = lookup.reindex(pd.MultiIndex.from_frame(test_unique_str))
    fallback_mask = preds_unique.isna().to_numpy()
    test_unique["_pred"] = preds_unique.fillna(global_majority).to_numpy()

    # 3. 行级回退统计：回退组合的行数
    n_total = len(X_test)
    if fallback_mask.any():
        fallback_keys = test_unique.loc[fallback_mask, keys]
        n_fallback = int(X_test.merge(fallback_keys, on=keys, how="inner").shape[0])
    else:
        n_fallback = 0

    # 4. merge 映射回全部行（category 列上的哈希连接）
    pred_map = test_unique[keys + ["_pred"]]
    merged = X_test[keys].merge(pred_map, on=keys, how="left")
    preds = merged["_pred"].to_numpy(dtype=object)

    n_unique_test = len(test_unique)
    del test_unique, test_unique_str, preds_unique, pred_map, merged
    gc.collect()

    stats = {
        "n_combos": int(len(lookup)),
        "n_hit": n_total - n_fallback,
        "n_fallback": n_fallback,
        "n_total": n_total,
        "hit_rate": round((n_total - n_fallback) / n_total, 6) if n_total else None,
        "n_unique_test_combos": n_unique_test,
    }
    return preds, stats


# --------------------------------------------------------------------------- #
# 4. 高效评估（只调用一次 confusion_matrix，输出格式与 prototype.evaluate 一致）
# --------------------------------------------------------------------------- #

def _per_class_metrics(cm: np.ndarray, n_classes: int = 3):
    """从混淆矩阵计算各类的 (precision, recall, f1, support)。"""
    results = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        support = int(cm[i, :].sum())
        results.append((float(precision), float(recall), float(f1), support))
    return results


def evaluate_efficient(
    name: str, y_true: pd.Series, y_pred: np.ndarray
) -> tuple[dict, float, np.ndarray]:
    """内存高效 + 时间高效的评估。

    与 ``prototype.evaluate`` 的区别：只调用一次 ``confusion_matrix``（22 秒），
    所有 precision/recall/F1 从 3×3 矩阵推导（瞬时）；prototype.evaluate 调用
    ``classification_report`` + ``confusion_matrix`` + ``f1_score``×2 共 4 次
    标签编码（合计约 178 秒），三基线叠加会超时限被 SIGTERM。

    输出格式与 ``prototype.evaluate`` 完全一致（宏平均 F1、各类 P/R/F1/support
    的 classification_report 格式表、混淆矩阵）。返回 ``(result_dict, macro_f1,
    confusion_matrix)``，其中 macro_f1 未取整（供 Bootstrap / 对比用），cm
    供外部复用避免重复计算。
    """
    # 唯一一次标签编码 + 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES)
    n = len(TARGET_CLASSES)
    total = int(cm.sum())

    metrics = _per_class_metrics(cm, n)
    precisions = [m[0] for m in metrics]
    recalls = [m[1] for m in metrics]
    f1s = [m[2] for m in metrics]
    supports = [m[3] for m in metrics]

    macro_p = sum(precisions) / n
    macro_r = sum(recalls) / n
    macro_f1 = sum(f1s) / n
    accuracy = float(cm.diagonal().sum()) / total

    weighted_p = sum(p * s for p, s in zip(precisions, supports)) / total
    weighted_r = sum(r * s for r, s in zip(recalls, supports)) / total
    weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / total

    # --- 打印（格式与 prototype.evaluate 完全一致） ---
    print(f"\n--- {name} ---")
    print(f"宏平均 F1 = {macro_f1:.4f}")

    # 复刻 sklearn classification_report 的表格格式
    name_w = max(len(c) for c in TARGET_CLASSES)
    print(f"\n{'':>{name_w}}  precision    recall  f1-score   support\n")
    for i, cls in enumerate(TARGET_CLASSES):
        print(f"{cls:>{name_w}}  {precisions[i]:>9.4f}  {recalls[i]:>7.4f}"
              f"  {f1s[i]:>8.4f}  {supports[i]:>9}")
    print(f"\n{'accuracy':>{name_w}}  {'':>9}  {'':>7}"
          f"  {accuracy:>8.4f}  {total:>9}")
    print(f"{'macro avg':>{name_w}}  {macro_p:>9.4f}  {macro_r:>7.4f}"
          f"  {macro_f1:>8.4f}  {total:>9}")
    print(f"{'weighted avg':>{name_w}}  {weighted_p:>9.4f}  {weighted_r:>7.4f}"
          f"  {weighted_f1:>8.4f}  {total:>9}")

    short = [SHORT_LABELS[c] for c in TARGET_CLASSES]
    cm_df = pd.DataFrame(
        cm, index=[f"真实:{s}" for s in short], columns=[f"预测:{s}" for s in short]
    )
    print("\n混淆矩阵（行 = 真实，列 = 预测）")
    print(cm_df.to_string())

    pred_counts = pd.Series(y_pred).value_counts(normalize=True)
    result = {
        "baseline": name,
        "宏平均F1": round(macro_f1, 4),
        **{f"F1[{SHORT_LABELS[c]}]": round(f, 4)
           for c, f in zip(TARGET_CLASSES, f1s)},
        "准确率": round(accuracy, 4),
        "预测最集中": SHORT_LABELS.get(
            pred_counts.index[0], str(pred_counts.index[0])),
        "该类别占比": round(float(pred_counts.iloc[0]), 4),
    }
    return result, macro_f1, cm


# --------------------------------------------------------------------------- #
# 5. 单个基线的完整评估（报告 + Bootstrap + 对比）
# --------------------------------------------------------------------------- #

def run_baseline(
    name: str,
    y_test: pd.Series,
    y_pred: np.ndarray,
    phase1_f1: float,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """对单个基线：打印报告（同阶段一格式）→ Bootstrap CI → 与阶段一对比。

    返回包含所有指标的字典（供 JSON 落盘与最终对比表使用）。
    """
    # --- 打印报告 + 计算宏平均 F1（共用同一个 confusion_matrix） ---
    result, macro_f1, cm = evaluate_efficient(name, y_test, y_pred)
    gc.collect()
    _sample_rss()

    # --- Bootstrap 置信区间 ---
    # 编码到 int8（11M 行只需 11 MB），后续 Bootstrap 全在 9 个数字上做
    y_true_enc = _encode_labels(y_test, TARGET_CLASSES)
    y_pred_enc = _encode_labels(y_pred, TARGET_CLASSES)
    del y_pred, cm  # 释放预测数组与混淆矩阵，给后续步骤腾内存
    gc.collect()
    t_boot = time.time()
    boot_f1s, ci_lo, ci_hi = bootstrap_macro_f1(
        y_true_enc, y_pred_enc, n_boot=N_BOOTSTRAP, seed=seed
    )
    boot_secs = time.time() - t_boot

    print(f"  Bootstrap 95% CI（{N_BOOTSTRAP} 次有放回抽样，{boot_secs:.1f}s）: "
          f"[{ci_lo:.4f}, {ci_hi:.4f}]")

    # --- 与阶段一随机划分对比 ---
    decline_abs = macro_f1 - phase1_f1
    decline_rel = decline_abs / phase1_f1 * 100 if phase1_f1 else None
    print(f"  阶段一随机划分 F1 = {phase1_f1:.4f} → 时间划分变化 "
          f"{decline_abs:+.4f}（{decline_rel:+.2f}%）")

    return {
        "baseline": name,
        "macro_f1": round(macro_f1, 6),
        "ci_lo": round(ci_lo, 6),
        "ci_hi": round(ci_hi, 6),
        "ci_width": round(ci_hi - ci_lo, 6),
        "phase1_random_f1": phase1_f1,
        "decline_abs": round(decline_abs, 6),
        "decline_rel_pct": round(decline_rel, 4) if decline_rel is not None else None,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": seed,
        "boot_seconds": round(boot_secs, 2),
        "boot_f1_mean": round(float(boot_f1s.mean()), 6),
        "boot_f1_std": round(float(boot_f1s.std()), 6),
    }


# --------------------------------------------------------------------------- #
# 6. 加载
# --------------------------------------------------------------------------- #

def load_parquet(path: Path, label: str) -> pd.DataFrame:
    """读取已清洗的 parquet（split_temporal.py 用 cleaning.clean_data 产出）。"""
    if not path.exists():
        print(f"找不到 {path}，请先运行 python src/split_temporal.py 生成时间划分。",
              file=sys.stderr)
        raise FileNotFoundError(path)
    t0 = time.time()
    df = pd.read_parquet(path)
    peak = _sample_rss()
    print(f"  {label}: {path.name}（{len(df):,} 行，{time.time() - t0:.1f}s"
          f"{f'，内存 {peak:,.0f} MB' if peak else ''}）")
    return df


# --------------------------------------------------------------------------- #
# 7. 对比表
# --------------------------------------------------------------------------- #

def print_comparison(results: list[dict]) -> None:
    """打印最终对比表：时间划分 F1 / Bootstrap CI / 随机划分 F1 / 下降幅度。"""
    line = "=" * 88
    print(f"\n{line}")
    print("基线对比（宏平均 F1；Bootstrap 95% CI；与阶段一随机划分对比）")
    print(line)

    headers = ["基线", "时间划分 F1", "Bootstrap 95% CI", "随机划分 F1", "下降(绝对)", "下降(相对)"]
    rows = []
    for r in results:
        rows.append([
            r["baseline"],
            f"{r['macro_f1']:.4f}",
            f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]",
            f"{r['phase1_random_f1']:.4f}",
            f"{r['decline_abs']:+.4f}",
            f"{r['decline_rel_pct']:+.2f}%" if r["decline_rel_pct"] is not None else "-",
        ])
    print_table(headers, rows, "lllrrrr")

    print()
    print("说明：")
    print("  · 时间划分 F1 = 在 train.parquet(≤2023-12-31) / test.parquet(≥2024-01-01) 上的结果")
    print(f"  · Bootstrap 95% CI = 测试集有放回抽样 {N_BOOTSTRAP} 次（种子 {BOOTSTRAP_SEED}），取 2.5%/97.5% 分位")
    print("  · 下降 = 时间划分相对随机划分的变化；负值表示时间划分下性能下降")
    print(line)


# --------------------------------------------------------------------------- #
# 8. 入口
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="CFPB 时间划分基线重跑 + Bootstrap 置信区间")
    parser.add_argument("--train", default=TRAIN_PATH, help=f"训练集 parquet（默认 {TRAIN_PATH}）")
    parser.add_argument("--test", default=TEST_PATH, help=f"测试集 parquet（默认 {TEST_PATH}）")
    parser.add_argument("--stats", default=STATS_JSON, help=f"统计输出（默认 {STATS_JSON}）")
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED, help=f"Bootstrap 随机种子（默认 {BOOTSTRAP_SEED}）")
    args = parser.parse_args(argv)

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path

    train_path, test_path, stats_path = resolve(args.train), resolve(args.test), resolve(args.stats)

    line = "=" * 88
    print(line)
    print("CFPB 时间划分基线重跑 + Bootstrap 置信区间")
    print(line)
    print(f"  训练集 : {train_path}")
    print(f"  测试集 : {test_path}")
    print(f"  输出   : {stats_path.name}")
    print(f"  Bootstrap : {N_BOOTSTRAP} 次，种子 {args.seed}，95% CI")

    t_all = time.time()

    # ----------------------------------------------------------------- #
    # 阶段 A：加载训练集 → 构建全部查表 → 释放训练集
    # ----------------------------------------------------------------- #
    print(f"\n[1/5] 加载训练集、构建查表")
    train = load_parquet(train_path, "训练集")

    global_majority = _majority_class(train[TARGET])
    train_dist = train[TARGET].value_counts().reindex(TARGET_CLASSES)
    print(f"  全局多数类: {global_majority}")
    print(f"  训练集分布:")
    for cls, cnt in train_dist.items():
        print(f"    {cls:<40s} {cnt:>10,}  {cnt / len(train):>7.2%}")

    print(f"  构建 Product × Issue 查表...")
    lookup_pi = build_lookup(train, ("Product", "Issue"))
    print(f"    → {len(lookup_pi):,} 个组合")

    print(f"  构建 Product × Issue × Company 查表...")
    lookup_pic = build_lookup(train, ("Product", "Issue", "Company"))
    print(f"    → {len(lookup_pic):,} 个组合")

    del train
    gc.collect()
    _sample_rss()

    # ----------------------------------------------------------------- #
    # 阶段 B：加载测试集
    # ----------------------------------------------------------------- #
    print(f"\n[2/5] 加载测试集")
    test = load_parquet(test_path, "测试集")
    y_test = test[TARGET]
    X_test = test[FEATURES]
    test_dist = y_test.value_counts().reindex(TARGET_CLASSES)
    print(f"  测试集分布:")
    for cls, cnt in test_dist.items():
        print(f"    {cls:<40s} {cnt:>10,}  {cnt / len(y_test):>7.2%}")

    n_test = len(y_test)

    # ----------------------------------------------------------------- #
    # 阶段 C：三个基线
    # ----------------------------------------------------------------- #
    results = []

    # --- 基线 1：多数类 ---
    print(f"\n[3/5] 基线 1：多数类（永远预测训练集多数类 {global_majority!r}）")
    pred_maj = np.full(n_test, global_majority, dtype=object)
    results.append(run_baseline(
        "多数类", y_test, pred_maj,
        phase1_f1=PHASE1_RANDOM_F1["majority"], seed=args.seed,
    ))

    # --- 基线 2：Product × Issue 查表 ---
    print(f"\n[4/5] 基线 2：Product × Issue 查表")
    pred_pi, stats_pi = apply_lookup(
        X_test, ("Product", "Issue"), lookup_pi, global_majority,
    )
    print(f"  组合键 (Product, Issue)：训练集 {stats_pi['n_combos']:,} 个组合，"
          f"测试集命中 {stats_pi['n_hit']:,} 行（{stats_pi['hit_rate']:.2%}），"
          f"回退 {stats_pi['n_fallback']:,} 行")
    results.append(run_baseline(
        "Product × Issue 查表", y_test, pred_pi,
        phase1_f1=PHASE1_RANDOM_F1["product_issue"], seed=args.seed,
    ))
    del pred_pi
    gc.collect()

    # --- 基线 3：Product × Issue × Company 查表 ---
    print(f"\n[5/5] 基线 3：Product × Issue × Company 查表")
    pred_pic, stats_pic = apply_lookup(
        X_test, ("Product", "Issue", "Company"), lookup_pic, global_majority,
    )
    print(f"  组合键 (Product, Issue, Company)：训练集 {stats_pic['n_combos']:,} 个组合，"
          f"测试集命中 {stats_pic['n_hit']:,} 行（{stats_pic['hit_rate']:.2%}），"
          f"回退 {stats_pic['n_fallback']:,} 行")
    results.append(run_baseline(
        "Product × Issue × Company 查表", y_test, pred_pic,
        phase1_f1=PHASE1_RANDOM_F1["product_issue_company"], seed=args.seed,
    ))
    del pred_pic
    gc.collect()

    # ----------------------------------------------------------------- #
    # 对比表
    # ----------------------------------------------------------------- #
    print_comparison(results)

    # ----------------------------------------------------------------- #
    # 落盘
    # ----------------------------------------------------------------- #
    stats = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "train_path": str(train_path),
            "test_path": str(test_path),
            "n_bootstrap": N_BOOTSTRAP,
            "seed": args.seed,
            "ci_percentiles": [CI_LOW_PCT, CI_HIGH_PCT],
            "phase1_random_f1": PHASE1_RANDOM_F1,
            "peak_rss_mb": round(_PEAK_RSS_MB, 1) if _PEAK_RSS_MB else None,
            "elapsed_seconds": round(time.time() - t_all, 2),
        },
        "train_distribution": {
            cls: {"count": int(train_dist.get(cls, 0)),
                  "share": round(float(train_dist.get(cls, 0)) / n_test if False else 0, 6)}
            for cls in TARGET_CLASSES
        },
        "test_distribution": {
            cls: {"count": int(test_dist.get(cls, 0)),
                  "share": round(float(test_dist.get(cls, 0)) / n_test, 6) if n_test else None}
            for cls in TARGET_CLASSES
        },
        "lookup_stats": {
            "product_issue": stats_pi,
            "product_issue_company": stats_pic,
        },
        "results": results,
    }

    # 修正 train_distribution 的 share（n_test 这里是错的，用 train_dist 自身的总数）
    train_total = int(train_dist.sum())
    stats["train_distribution"] = {
        cls: {"count": int(train_dist.get(cls, 0)),
              "share": round(float(train_dist.get(cls, 0)) / train_total, 6) if train_total else None}
        for cls in TARGET_CLASSES
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    peak = _PEAK_RSS_MB or None
    print(f"\n统计已保存：{stats_path}")
    if peak:
        print(f"进程内存峰值 {peak:,.0f} MB"
              f"（预算 {MEMORY_BUDGET_GB:.0f} GB → {'达标' if peak < MEMORY_BUDGET_GB * 1024 else '超限'}）")
    print(f"总用时 {time.time() - t_all:.1f}s")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
