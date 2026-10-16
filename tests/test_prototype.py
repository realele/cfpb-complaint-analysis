#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""src/prototype.py 的自动化测试。

全部测试数据都在本文件内手工构造，不依赖 data/sample.csv。

运行::

    pytest -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 让测试可以直接 import src/prototype.py（无论从哪个目录执行 pytest）
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import prototype as P  # noqa: E402

TARGET = P.TARGET
EXP = "Closed with explanation"
NONMON = "Closed with non-monetary relief"
MON = "Closed with monetary relief"
CREDIT = "Credit reporting or other personal consumer reports"

# 一行「合法」记录的默认值，各测试只需覆盖自己关心的字段
DEFAULT_ROW = {
    "Complaint ID": "id-0",
    TARGET: EXP,
    "Product": CREDIT,
    "Issue": "Incorrect information on your report",
    "Company": "EQUIFAX, INC.",
    "State": "CA",
    "Submitted via": "Web",
}


def build(rows: list[dict]) -> pd.DataFrame:
    """按 DEFAULT_ROW 补齐字段构造测试 DataFrame；Complaint ID 默认逐行唯一。"""
    records = []
    for i, row in enumerate(rows):
        merged = {**DEFAULT_ROW, "Complaint ID": f"id-{i}", **row}
        records.append(merged)
    return pd.DataFrame(records)


def synthetic_frame(n_exp: int = 180, n_nonmon: int = 90, n_mon: int = 30) -> pd.DataFrame:
    """构造类别比例 60/30/10 的合成数据（共 300 行），用于划分相关测试。"""
    rows = []
    for resp, n in ((EXP, n_exp), (NONMON, n_nonmon), (MON, n_mon)):
        for k in range(n):
            rows.append({
                TARGET: resp,
                "Product": [CREDIT, "Debt collection", "Credit card"][k % 3],
                "Issue": ["Issue A", "Issue B"][k % 2],
                "Company": f"Company {k % 7}",
                "State": ["CA", "NY", "TX"][k % 3],
                "Submitted via": "Web",
            })
    return build(rows)


# --------------------------------------------------------------------------- #
# 1. 类别筛选
# --------------------------------------------------------------------------- #

def test_01_clean_data_drops_non_target_classes():
    """非目标类别（In progress 等）必须被剔除，只留下三个目标类别。"""
    df = build([
        {"Complaint ID": "1", TARGET: EXP},
        {"Complaint ID": "2", TARGET: NONMON},
        {"Complaint ID": "3", TARGET: MON},
        {"Complaint ID": "4", TARGET: "In progress"},
        {"Complaint ID": "5", TARGET: "Untimely response"},
        {"Complaint ID": "6", TARGET: "Closed"},
        {"Complaint ID": "7", TARGET: "Closed without relief"},
        {"Complaint ID": "8", TARGET: "Closed with relief"},
    ])

    clean, mapping = P.clean_data(df)

    assert isinstance(mapping, pd.DataFrame)          # 第二个返回值是合并映射表
    assert len(clean) == 3
    assert set(clean[TARGET]) == set(P.TARGET_CLASSES)
    for bad in ("In progress", "Untimely response", "Closed",
                "Closed without relief", "Closed with relief"):
        assert bad not in set(clean[TARGET])


# --------------------------------------------------------------------------- #
# 2. 按 Complaint ID 去重
# --------------------------------------------------------------------------- #

def test_02_clean_data_deduplicates_by_complaint_id():
    """同一个 Complaint ID 只保留第一次出现的记录。"""
    df = build([
        {"Complaint ID": "C-1", TARGET: EXP},
        {"Complaint ID": "C-1", TARGET: NONMON},   # 重复 ID，应被删除
        {"Complaint ID": "C-2", TARGET: MON},
        {"Complaint ID": "C-2", TARGET: MON},      # 重复 ID，应被删除
        {"Complaint ID": "C-3", TARGET: EXP},
    ])

    clean, _ = P.clean_data(df)

    assert len(clean) == 3
    assert clean["Complaint ID"].is_unique
    assert set(clean["Complaint ID"]) == {"C-1", "C-2", "C-3"}
    # keep="first"：C-1 保留的是第一条（explanation）而不是第二条（non-monetary）
    assert clean.loc[clean["Complaint ID"] == "C-1", TARGET].iloc[0] == EXP


# --------------------------------------------------------------------------- #
# 3. Product 名称归一化
# --------------------------------------------------------------------------- #

def test_03_product_names_are_normalized():
    """含 credit repair services 的旧命名、Credit reporting 等应合并为统一名称。"""
    legacy_long = "Credit reporting, credit repair services, or other personal consumer reports"
    df = build([
        {"Product": "Credit reporting"},
        {"Product": legacy_long},
        {"Product": CREDIT},                       # 已经是新命名，保持不变
        {"Product": "Bank account or service"},
        {"Product": "Money transfers"},
    ])

    clean, mapping = P.clean_data(df)

    assert len(clean) == 5                                   # 归一化不改变行数
    assert clean["Product"].nunique() == 3
    assert set(clean["Product"]) == {
        CREDIT, "Checking or savings account", "Money transfer, virtual currency, or money service"
    }

    m = mapping.set_index("原名称")
    assert len(mapping) == 4                                 # 4 条旧命名被合并
    assert m.loc["Credit reporting", "归一化后"] == CREDIT
    assert m.loc[legacy_long, "归一化后"] == CREDIT
    assert m.loc["Bank account or service", "归一化后"] == "Checking or savings account"
    assert m.loc["Money transfers", "归一化后"] == "Money transfer, virtual currency, or money service"
    # 映射表里的行数统计要准确
    assert m.loc["Credit reporting", "影响行数"] == 1
    assert m.loc["Bank account or service", "影响行数"] == 1


# --------------------------------------------------------------------------- #
# 4. State 空值填充
# --------------------------------------------------------------------------- #

def test_04_state_blanks_are_filled_with_unknown():
    """State 的空值（NaN / 空串）填成 "Unknown"，Submitted via 同理。"""
    df = build([
        {"Complaint ID": "1", "State": np.nan},
        {"Complaint ID": "2", "State": ""},
        {"Complaint ID": "3", "State": "   "},     # 纯空格也算空值
        {"Complaint ID": "4", "State": "CA"},
        {"Complaint ID": "5", "State": " NY "},    # 正常值应被去空格保留
        {"Complaint ID": "6", "State": "TX", "Submitted via": np.nan},
    ])

    clean, _ = P.clean_data(df)

    assert clean["State"].isna().sum() == 0
    assert clean["Submitted via"].isna().sum() == 0
    assert int((clean["State"] == "Unknown").sum()) == 3
    assert int((clean["Submitted via"] == "Unknown").sum()) == 1
    assert set(clean["State"]) == {"Unknown", "CA", "NY", "TX"}


# --------------------------------------------------------------------------- #
# 5. 划分比例与分层
# --------------------------------------------------------------------------- #

def test_05_split_is_80_20_and_stratified():
    """测试集约占 20%，且三个类别在测试集中的占比与整体一致（分层）。"""
    df = synthetic_frame()                       # 300 行，60% / 30% / 10%

    X_train, X_test, y_train, y_test = P.split_data(df, test_size=0.2, seed=42)

    assert len(X_train) + len(X_test) == len(df)
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)

    ratio = len(X_test) / len(df)
    assert 0.18 <= ratio <= 0.22, f"测试集占比 {ratio:.2%} 偏离 20% 太多"

    overall = df[TARGET].value_counts(normalize=True).reindex(P.TARGET_CLASSES)
    in_test = y_test.value_counts(normalize=True).reindex(P.TARGET_CLASSES)
    assert np.allclose(overall.to_numpy(), in_test.to_numpy(), atol=0.02), (
        f"分层失败：整体 {overall.round(4).to_dict()} vs 测试集 {in_test.round(4).to_dict()}"
    )
    # 少数类不能整个掉出测试集
    assert (y_test.value_counts().reindex(P.TARGET_CLASSES) > 0).all()


# --------------------------------------------------------------------------- #
# 6. 可复现性
# --------------------------------------------------------------------------- #

def test_06_same_seed_gives_identical_split():
    """相同随机种子两次划分结果完全相同；换种子则会不同。"""
    df = synthetic_frame()

    a = P.split_data(df, test_size=0.2, seed=42)
    b = P.split_data(df, test_size=0.2, seed=42)
    c = P.split_data(df, test_size=0.2, seed=7)

    assert a[1].equals(b[1])                                    # X_test 完全一致
    assert a[3].equals(b[3])                                    # y_test 完全一致
    assert list(a[0].index) == list(b[0].index)                 # 训练集索引一致

    assert list(a[1].index) != list(c[1].index)                 # 换种子后划分不同


# --------------------------------------------------------------------------- #
# 7. 多数类基线
# --------------------------------------------------------------------------- #

def test_07_baseline_majority_always_predicts_majority_class():
    """无论测试集长什么样，多数类基线只输出一个类别，且就是训练集的多数类。"""
    df = build([
        {"Complaint ID": "1", TARGET: EXP},
        {"Complaint ID": "2", TARGET: EXP},
        {"Complaint ID": "3", TARGET: EXP},
        {"Complaint ID": "4", TARGET: NONMON},
        {"Complaint ID": "5", TARGET: MON},
    ])
    X_test = build([{"Complaint ID": f"t{i}"} for i in range(4)])[P.FEATURES]

    pred = P.baseline_majority(df[P.FEATURES], df[TARGET], X_test)
    assert len(pred) == len(X_test)
    assert set(pred) == {EXP}

    # 训练集多数类变成 non-monetary 后，预测也跟着变
    df2 = build([
        {"Complaint ID": "1", TARGET: NONMON},
        {"Complaint ID": "2", TARGET: NONMON},
        {"Complaint ID": "3", TARGET: MON},
    ])
    pred2 = P.baseline_majority(df2[P.FEATURES], df2[TARGET], X_test)
    assert set(pred2) == {NONMON}


# --------------------------------------------------------------------------- #
# 8. 查表基线的回退
# --------------------------------------------------------------------------- #

def test_08_baseline_lookup_falls_back_to_global_majority(capsys):
    """训练集没见过的 Product × Issue 组合，回退到全局多数类。"""
    train = build([
        {"Product": "A", "Issue": "X", TARGET: EXP},
        {"Product": "A", "Issue": "X", TARGET: EXP},
        {"Product": "B", "Issue": "Y", TARGET: NONMON},
    ])   # 全局多数类 = EXP

    test = build([
        {"Complaint ID": "t1", "Product": "A", "Issue": "X"},          # 见过 -> EXP
        {"Complaint ID": "t2", "Product": "B", "Issue": "Y"},          # 见过 -> NONMON
        {"Complaint ID": "t3", "Product": "C", "Issue": "Z"},          # 没见过 -> 全局多数类
        {"Complaint ID": "t4", "Product": "C", "Issue": "Z"},          # 没见过 -> 全局多数类
    ])

    pred = P.baseline_lookup(train[P.FEATURES], train[TARGET], test[P.FEATURES])
    out = capsys.readouterr().out

    assert list(pred) == [EXP, NONMON, EXP, EXP]
    assert "未见组合回退到全局多数类" in out
    assert "：2 行" in out, "回退行数统计应为 2 行"


# --------------------------------------------------------------------------- #
# 9. 无效输入要有清晰报错
# --------------------------------------------------------------------------- #

def test_09_invalid_inputs_raise_clear_errors():
    """空 DataFrame、非 DataFrame、缺列、长度不一致等都必须显式报错。"""
    # 空 DataFrame
    with pytest.raises(ValueError, match="为空"):
        P.clean_data(pd.DataFrame())
    with pytest.raises(ValueError, match="为空"):
        P.split_data(pd.DataFrame())
    with pytest.raises(ValueError, match="为空"):
        P.baseline_majority(pd.DataFrame(), pd.Series([EXP]), pd.DataFrame())

    # 不是 DataFrame
    with pytest.raises(TypeError, match="DataFrame"):
        P.clean_data([1, 2, 3])

    # 缺列（错误信息里要指出缺了哪些列）
    missing = build([{"Complaint ID": "1"}]).drop(columns=["Issue", "Company"])
    with pytest.raises(ValueError, match="缺少必需列"):
        P.clean_data(missing)

    ok = build([{"Complaint ID": "1"}])
    with pytest.raises(ValueError, match="缺少必需列"):
        P.baseline_lookup(ok[P.FEATURES], ok[TARGET], ok[P.FEATURES].drop(columns=["Issue"]))

    # 筛选后没有任何数据
    with pytest.raises(ValueError, match="没有剩余数据"):
        P.clean_data(build([{"Complaint ID": "1", TARGET: "In progress"}]))

    # 预测与真实标签长度不一致
    with pytest.raises(ValueError, match="长度不一致"):
        P.evaluate("t", pd.Series([EXP, EXP]), np.array([EXP], dtype=object))


# --------------------------------------------------------------------------- #
# 10. 与手工计算逐条核对
# --------------------------------------------------------------------------- #

def test_10_lookup_matches_hand_computed_predictions():
    """10 行小数据上，查表基线的预测与手工计算逐条一致。

    训练集（10 行）各组合的多数类：
        A/X: explanation, explanation, non-monetary  ->  explanation
        B/Y: non-monetary, non-monetary, monetary    ->  non-monetary
        C/Z: monetary, monetary, explanation         ->  monetary
        D/W: non-monetary                            ->  non-monetary
        全局计数：explanation 3 / non-monetary 4 / monetary 3 -> 全局多数类 non-monetary
    """
    train = build([
        {"Product": "A", "Issue": "X", TARGET: EXP},
        {"Product": "A", "Issue": "X", TARGET: EXP},
        {"Product": "A", "Issue": "X", TARGET: NONMON},
        {"Product": "B", "Issue": "Y", TARGET: NONMON},
        {"Product": "B", "Issue": "Y", TARGET: NONMON},
        {"Product": "B", "Issue": "Y", TARGET: MON},
        {"Product": "C", "Issue": "Z", TARGET: MON},
        {"Product": "C", "Issue": "Z", TARGET: MON},
        {"Product": "C", "Issue": "Z", TARGET: EXP},
        {"Product": "D", "Issue": "W", TARGET: NONMON},
    ])
    # 故意打乱行序且不重置索引，模拟 train_test_split 之后"索引被打乱"的状态：
    # 若实现里用 pd.concat(axis=1) 按索引标签对齐特征与标签，这里就会算错
    train = train.sample(frac=1.0, random_state=7)

    test = build([
        {"Complaint ID": "t1", "Product": "A", "Issue": "X"},        # -> explanation
        {"Complaint ID": "t2", "Product": "B", "Issue": "Y"},        # -> non-monetary
        {"Complaint ID": "t3", "Product": "C", "Issue": "Z"},        # -> monetary
        {"Complaint ID": "t4", "Product": "D", "Issue": "W"},        # -> non-monetary
        {"Complaint ID": "t5", "Product": "E", "Issue": "从未出现过"},  # 未见组合 -> 全局多数类
    ])

    pred = P.baseline_lookup(train[P.FEATURES], train[TARGET], test[P.FEATURES])

    assert list(pred) == [EXP, NONMON, MON, NONMON, NONMON]
