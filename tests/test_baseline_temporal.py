#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""src/baseline_temporal.py 的自动化测试。

测试覆盖：

  1. ``macro_f1_from_cm`` 与 sklearn ``f1_score`` 完全一致（含 zero_division=0）；
  2. Bootstrap 区间覆盖率——合成数据上 95% CI 应覆盖真实 F1 约 95%；
  3. 随机种子可复现——相同种子结果完全一致，换种子则不同；
  4. 多项分布 Bootstrap 与逐行重抽样 Bootstrap 在分布上等价；
  5. 高效查表（``build_lookup`` + ``apply_lookup``）的预测与
     ``prototype.baseline_lookup`` 逐条一致（含平局打破、未见组合回退）；
  6. ``_encode_labels`` 对未知类别报错。

全部测试数据在本文件内手工构造，不依赖 data/ 下的真实 parquet。

运行::

    pytest -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, f1_score

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import baseline_temporal as BT  # noqa: E402
import prototype as P  # noqa: E402

TARGET = P.TARGET
EXP = "Closed with explanation"
NONMON = "Closed with non-monetary relief"
MON = "Closed with monetary relief"
CLASSES = P.TARGET_CLASSES

DEFAULT_ROW = {
    "Complaint ID": "id-0",
    TARGET: EXP,
    "Product": "Credit reporting or other personal consumer reports",
    "Issue": "Incorrect information on your report",
    "Company": "EQUIFAX, INC.",
    "State": "CA",
    "Submitted via": "Web",
}


def build(rows: list[dict]) -> pd.DataFrame:
    """按 DEFAULT_ROW 补齐字段构造测试 DataFrame。"""
    records = []
    for i, row in enumerate(rows):
        merged = {**DEFAULT_ROW, "Complaint ID": f"id-{i}", **row}
        records.append(merged)
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# 1. macro_f1_from_cm 与 sklearn 一致
# --------------------------------------------------------------------------- #

def test_01_macro_f1_from_cm_matches_sklearn():
    """在 20 组随机 (y_true, y_pred) 上，macro_f1_from_cm 与 sklearn f1_score 完全一致。"""
    rng = np.random.default_rng(99)
    classes = ["A", "B", "C"]
    for _ in range(20):
        y_true = rng.choice(classes, size=100)
        y_pred = rng.choice(classes, size=100)
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        f1_mine = BT.macro_f1_from_cm(cm, 3)
        f1_sklearn = f1_score(y_true, y_pred, labels=classes,
                              average="macro", zero_division=0)
        assert abs(f1_mine - f1_sklearn) < 1e-10, (
            f"不一致：mine={f1_mine:.10f}, sklearn={f1_sklearn:.10f}"
        )


def test_02_macro_f1_from_cm_zero_division():
    """某类完全缺席（TP=FP=FN=0）时 F1=0，不报错。"""
    # 类 C 完全不出现：真实和预测里都没有 C
    cm = np.array([[10, 0, 0],
                   [0, 5, 0],
                   [0, 0, 0]])
    # 类 C 的 TP=0, FP=0, FN=0 → denom=0 → F1=0
    # 类 A: TP=10, FP=0, FN=0 → F1=1
    # 类 B: TP=5, FP=0, FN=0 → F1=1
    # macro = (1 + 1 + 0) / 3 = 0.6667
    assert abs(BT.macro_f1_from_cm(cm, 3) - 2 / 3) < 1e-10

    # 用 sklearn 交叉验证
    y_true = ["A"] * 10 + ["B"] * 5
    y_pred = ["A"] * 10 + ["B"] * 5
    f1_sklearn = f1_score(y_true, y_pred, labels=["A", "B", "C"],
                          average="macro", zero_division=0)
    assert abs(BT.macro_f1_from_cm(cm, 3) - f1_sklearn) < 1e-10


def test_03_macro_f1_from_cm_perfect_and_worst():
    """完美预测 F1=1，全错（对角线全 0）F1=0。"""
    cm_perfect = np.array([[10, 0, 0],
                           [0, 5, 0],
                           [0, 0, 3]])
    assert BT.macro_f1_from_cm(cm_perfect, 3) == 1.0

    cm_worst = np.array([[0, 10, 0],
                         [5, 0, 0],
                         [0, 0, 0]])  # C 缺席
    # A: TP=0, FP=5, FN=10 → F1=0
    # B: TP=0, FP=10, FN=5 → F1=0
    # C: denom=0 → F1=0
    assert BT.macro_f1_from_cm(cm_worst, 3) == 0.0


# --------------------------------------------------------------------------- #
# 2. Bootstrap 区间覆盖率
# --------------------------------------------------------------------------- #

def test_04_bootstrap_coverage():
    """95% Bootstrap CI 应覆盖总体真实 F1 约 95% 的次数。

    用"总体 vs 样本"设计：先构造一个 5 万行的大总体，其 F1 视为"真值"；
    再从总体中反复抽 500 行小样本，每个样本构建 95% Bootstrap CI，
    验证 CI 覆盖真值的比例接近 95%（允许 ±10 pp 蒙特卡洛误差）。
    """
    rng = np.random.default_rng(12345)
    n_classes = 3
    N_POP = 50_000

    # 大"总体"：60% / 30% / 10%，预测正确约 65%
    pop_true = rng.choice(n_classes, size=N_POP, p=[0.6, 0.3, 0.1])
    pop_pred = pop_true.copy()
    flip = rng.random(N_POP) < 0.35
    pop_pred[flip] = rng.integers(0, n_classes, size=flip.sum())

    # 总体 F1 = "真值"
    cm = confusion_matrix(pop_true, pop_pred, labels=list(range(n_classes)))
    true_f1 = BT.macro_f1_from_cm(cm, n_classes)

    # 从总体中抽 200 个小样本（各 500 行），每个样本构建 95% CI
    n_trials = 200
    n_covered = 0
    for trial in range(n_trials):
        idx = rng.choice(N_POP, size=500, replace=False)
        y_t, y_p = pop_true[idx], pop_pred[idx]
        _, lo, hi = BT.bootstrap_macro_f1(
            y_t, y_p, n_boot=300, seed=trial * 1000 + 1, n_classes=n_classes,
        )
        if lo <= true_f1 <= hi:
            n_covered += 1

    coverage = n_covered / n_trials
    # 95% CI 的覆盖率应接近 0.95；bootstrap 百分位 CI 在小样本上偏低，允许 ±10 pp
    assert 0.82 <= coverage <= 0.99, (
        f"覆盖率 {coverage:.2%}，95% CI 应覆盖约 95%（允许 ±10 pp 蒙特卡洛误差）"
    )


# --------------------------------------------------------------------------- #
# 3. 随机种子可复现
# --------------------------------------------------------------------------- #

def test_05_bootstrap_seed_reproducible():
    """相同随机种子两次 Bootstrap 结果完全一致；换种子则不同。"""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 3, size=500)
    y_pred = rng.integers(0, 3, size=500)

    f1s_a, lo_a, hi_a = BT.bootstrap_macro_f1(y_true, y_pred, n_boot=200, seed=42)
    f1s_b, lo_b, hi_b = BT.bootstrap_macro_f1(y_true, y_pred, n_boot=200, seed=42)

    # 完全一致
    assert np.array_equal(f1s_a, f1s_b)
    assert lo_a == lo_b
    assert hi_a == hi_b

    # 换种子后不同
    f1s_c, lo_c, hi_c = BT.bootstrap_macro_f1(y_true, y_pred, n_boot=200, seed=7)
    assert not np.array_equal(f1s_a, f1s_c), "换种子后 Bootstrap 结果不应相同"


# --------------------------------------------------------------------------- #
# 4. 多项分布 Bootstrap 与逐行重抽样等价
# --------------------------------------------------------------------------- #

def test_06_multinomial_matches_direct_bootstrap():
    """多项分布版与逐行重抽样版的 CI 在蒙特卡洛误差范围内一致。

    两种方法用不同的随机数流，不会逐次相同，但在同一数据上各自运行多次后，
    CI 的均值应非常接近（差异 < 0.02）。
    """
    rng = np.random.default_rng(77)
    n_classes = 3
    N = 2000
    y_true = rng.choice(n_classes, size=N, p=[0.5, 0.35, 0.15])
    y_pred = y_true.copy()
    flip = rng.random(N) < 0.4
    y_pred[flip] = rng.integers(0, n_classes, size=flip.sum())

    # 各跑 30 次，取 CI 中点的均值
    multinomial_mids = []
    direct_mids = []
    for trial in range(30):
        _, lo_m, hi_m = BT.bootstrap_macro_f1(
            y_true, y_pred, n_boot=300, seed=trial * 100 + 10, n_classes=n_classes)
        multinomial_mids.append((lo_m + hi_m) / 2)

        _, lo_d, hi_d = BT.bootstrap_macro_f1_direct(
            y_true, y_pred, n_boot=300, seed=trial * 100 + 10, n_classes=n_classes)
        direct_mids.append((lo_d + hi_d) / 2)

    mean_diff = abs(np.mean(multinomial_mids) - np.mean(direct_mids))
    assert mean_diff < 0.02, (
        f"多项分布版与逐行版 CI 中点均值差异 {mean_diff:.4f} 过大"
    )


# --------------------------------------------------------------------------- #
# 5. 高效查表与 prototype.baseline_lookup 预测完全一致
# --------------------------------------------------------------------------- #

def test_07_lookup_matches_prototype_predictions():
    """高效查表（build_lookup + apply_lookup）的预测与 prototype.baseline_lookup 逐条一致。

    包含平局打破、未见组合回退、索引打乱等场景。
    """
    # 训练集：各组合的多数类已知（含平局）
    train = build([
        {"Product": "A", "Issue": "X", TARGET: EXP},
        {"Product": "A", "Issue": "X", TARGET: EXP},
        {"Product": "A", "Issue": "X", TARGET: NONMON},   # A/X -> EXP (2:1)
        {"Product": "B", "Issue": "Y", TARGET: NONMON},
        {"Product": "B", "Issue": "Y", TARGET: NONMON},
        {"Product": "B", "Issue": "Y", TARGET: MON},      # B/Y -> NONMON (2:1)
        {"Product": "C", "Issue": "Z", TARGET: MON},
        {"Product": "C", "Issue": "Z", TARGET: MON},
        {"Product": "C", "Issue": "Z", TARGET: EXP},      # C/Z -> MON (2:1)
        {"Product": "D", "Issue": "W", TARGET: NONMON},   # D/W -> NONMON (1:0)
        {"Product": "E", "Issue": "V", TARGET: EXP},      # E/V -> EXP (1:0)
        {"Product": "E", "Issue": "V", TARGET: NONMON},   # E/V 平局 1:1 -> 字母序 EXP
    ])
    # 故意打乱行序，模拟 train_test_split 后索引被打乱
    train = train.sample(frac=1.0, random_state=7).reset_index(drop=True)

    test = build([
        {"Product": "A", "Issue": "X"},        # -> EXP
        {"Product": "B", "Issue": "Y"},        # -> NONMON
        {"Product": "C", "Issue": "Z"},        # -> MON
        {"Product": "D", "Issue": "W"},        # -> NONMON
        {"Product": "E", "Issue": "V"},        # -> EXP（平局字母序）
        {"Product": "F", "Issue": "U"},        # 未见 -> 全局多数类
        {"Product": "A", "Issue": "Y"},        # 未见组合 -> 全局多数类
    ])

    # prototype 版
    pred_proto = P.baseline_lookup(
        train[P.FEATURES], train[TARGET], test[P.FEATURES],
        key_columns=("Product", "Issue"),
    )

    # 高效版
    lookup = BT.build_lookup(train, ("Product", "Issue"))
    global_majority = P._majority_class(train[TARGET])
    pred_mine, stats = BT.apply_lookup(
        test[P.FEATURES], ("Product", "Issue"), lookup, global_majority,
    )

    assert np.array_equal(pred_proto, pred_mine), (
        f"预测不一致：\nproto={pred_proto}\nmine ={pred_mine}"
    )
    # 未见组合数 = 2（F/U 和 A/Y）
    assert stats["n_fallback"] == 2
    assert stats["n_hit"] == 5


def test_08_lookup_with_company_matches_prototype():
    """三列键（Product × Issue × Company）查表也与 prototype 一致。"""
    train = build([
        {"Product": "A", "Issue": "X", "Company": "C1", TARGET: EXP},
        {"Product": "A", "Issue": "X", "Company": "C1", TARGET: EXP},
        {"Product": "A", "Issue": "X", "Company": "C1", TARGET: NONMON},  # A/X/C1 -> EXP
        {"Product": "A", "Issue": "X", "Company": "C2", TARGET: NONMON},
        {"Product": "A", "Issue": "X", "Company": "C2", TARGET: MON},     # A/X/C2 -> MON(平局)->字母序 MON
        {"Product": "B", "Issue": "Y", "Company": "C1", TARGET: MON},
        {"Product": "B", "Issue": "Y", "Company": "C1", TARGET: MON},     # B/Y/C1 -> MON
    ])
    train = train.sample(frac=1.0, random_state=3).reset_index(drop=True)

    test = build([
        {"Product": "A", "Issue": "X", "Company": "C1"},  # -> EXP
        {"Product": "A", "Issue": "X", "Company": "C2"},  # -> MON
        {"Product": "B", "Issue": "Y", "Company": "C1"},  # -> MON
        {"Product": "A", "Issue": "X", "Company": "C3"},  # 未见公司 -> 全局多数类
        {"Product": "Z", "Issue": "Z", "Company": "C9"},  # 全未见 -> 全局多数类
    ])

    pred_proto = P.baseline_lookup(
        train[P.FEATURES], train[TARGET], test[P.FEATURES],
        key_columns=("Product", "Issue", "Company"),
    )

    lookup = BT.build_lookup(train, ("Product", "Issue", "Company"))
    global_majority = P._majority_class(train[TARGET])
    pred_mine, stats = BT.apply_lookup(
        test[P.FEATURES], ("Product", "Issue", "Company"), lookup, global_majority,
    )

    assert np.array_equal(pred_proto, pred_mine), (
        f"三列键预测不一致：\nproto={pred_proto}\nmine ={pred_mine}"
    )


def test_09_lookup_tie_break_alphabetical():
    """平局时按类别名字母序打破（与 prototype 一致）。"""
    # EXP vs MON vs NONMON 平局（各 1 票）
    # 字母序：EXP < MON < NONMON → 选 EXP
    train = build([
        {"Product": "P", "Issue": "I", TARGET: EXP},
        {"Product": "P", "Issue": "I", TARGET: MON},
        {"Product": "P", "Issue": "I", TARGET: NONMON},
    ])
    test = build([{"Product": "P", "Issue": "I"}])

    lookup = BT.build_lookup(train, ("Product", "Issue"))
    global_majority = P._majority_class(train[TARGET])
    pred, _ = BT.apply_lookup(
        test[P.FEATURES], ("Product", "Issue"), lookup, global_majority,
    )
    assert pred[0] == EXP  # 字母序最小

    # MON vs NONMON 平局（各 1 票），EXP 0 票
    # 字母序：MON < NONMON → 选 MON
    train2 = build([
        {"Product": "P", "Issue": "I", TARGET: MON},
        {"Product": "P", "Issue": "I", TARGET: NONMON},
    ])
    lookup2 = BT.build_lookup(train2, ("Product", "Issue"))
    global_majority2 = P._majority_class(train2[TARGET])
    pred2, _ = BT.apply_lookup(
        test[P.FEATURES], ("Product", "Issue"), lookup2, global_majority2,
    )
    assert pred2[0] == MON


# --------------------------------------------------------------------------- #
# 6. 编码与边界
# --------------------------------------------------------------------------- #

def test_10_encode_labels_roundtrip():
    """_encode_labels 按 TARGET_CLASSES 顺序编码，可往返还原。"""
    # TARGET_CLASSES = [EXP, NONMON, MON] → EXP=0, NONMON=1, MON=2
    labels = pd.Series([EXP, NONMON, MON, EXP, MON])
    codes = BT._encode_labels(labels, CLASSES)
    expected = np.array([0, 1, 2, 0, 2], dtype=np.int8)
    assert np.array_equal(codes, expected)


def test_11_encode_labels_rejects_unknown():
    """_encode_labels 对未知类别报错。"""
    labels = pd.Series([EXP, "Closed", NONMON])  # "Closed" 不在 TARGET_CLASSES
    with pytest.raises(ValueError, match="未知类别"):
        BT._encode_labels(labels, CLASSES)


def test_12_bootstrap_returns_valid_ci():
    """Bootstrap 返回的 CI 满足 lo <= mean <= hi，且 lo/hi 在 [0,1] 内。"""
    rng = np.random.default_rng(55)
    y_true = rng.integers(0, 3, size=1000)
    y_pred = rng.integers(0, 3, size=1000)

    boot_f1s, lo, hi = BT.bootstrap_macro_f1(y_true, y_pred, n_boot=200, seed=42)

    assert len(boot_f1s) == 200
    assert 0 <= lo <= hi <= 1.0
    assert lo <= boot_f1s.mean() <= hi
    assert lo <= np.median(boot_f1s) <= hi
