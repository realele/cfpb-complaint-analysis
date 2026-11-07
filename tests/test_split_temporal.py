#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""src/split_temporal.py 的自动化测试。

用一份 7 行、手工构造的 parquet 缓存跑通整条划分流程，再核对落盘结果与统计。
数据里刻意塞了这些边界情况：

  * ``2023-12-31T23:59:59``（带时间戳）必须落在训练集；
  * ``2024-01-01T00:00:00`` 必须落在测试集；
  * 同一个 Complaint ID 出现两次（要按 ID 去重）；
  * 一行 Product 为空、一行是 ``In progress``（都要被剔除）；
  * ``State`` 为纯空白、Product 是旧命名（要走 cleaning 的填充与归一化）。

运行::

    pytest -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cleaning as C  # noqa: E402
import prototype as P  # noqa: E402
import split_temporal as S  # noqa: E402

EXP = "Closed with explanation"
NONMON = "Closed with non-monetary relief"
MON = "Closed with monetary relief"

BOUNDARY = pd.Timestamp("2024-01-01 00:00:00")

# (date_received, product, issue, company, state, submitted_via, target, complaint_id)
ROWS = [
    ("2023-12-31T23:59:59.000Z", "Credit reporting", "Issue A", "Company A", "CA",
     "Web", EXP, "1"),
    ("2023-12-31T23:59:59.000Z", "Credit reporting", "Issue A", "Company A", "CA",
     "Web", EXP, "1"),                                  # 同 ID 重复，应被去重
    ("2024-01-01T00:00:00.000Z", "Mortgage", "Issue B", "Company A", "",
     "Phone", NONMON, "2"),                             # 空 State → Unknown
    ("2024-06-01T10:00:00.000Z", "Credit card or prepaid card", "Issue C", "Company B",
     "TX", "Web", MON, "3"),                            # 旧命名 → Credit card
    ("2023-05-05", "Bank account or service", "Issue D", "Company C", "   ",
     "Web", EXP, "4"),                                  # 空白 State、旧命名
    ("2022-01-01", "", "Issue E", "Company D", "NY", "Web", EXP, "5"),        # 无 Product
    ("2022-02-02", "Debt collection", "Issue F", "Company E", "NY", "Web",
     "In progress", "6"),                               # 非三主类
]


def write_cache(path: Path) -> Path:
    con = duckdb.connect()
    values = ",\n".join(
        "(" + ", ".join("NULL" if v == "" else "'" + v.replace("'", "''") + "'" for v in row) + ")"
        for row in ROWS
    )
    con.execute(f"""
        COPY (
            SELECT * FROM (VALUES {values}) AS t(
                date_received, product, issue, company, state,
                submitted_via, target, complaint_id)
        ) TO '{path.as_posix()}' (FORMAT parquet)
    """)
    con.close()
    return path


@pytest.fixture(scope="module")
def split_outputs(tmp_path_factory) -> dict:
    """跑一遍完整划分，返回路径与统计。"""
    tmp = tmp_path_factory.mktemp("cfpb_split")
    cache = write_cache(tmp / "cache.parquet")
    train, test, stats = tmp / "train.parquet", tmp / "test.parquet", tmp / "stats.json"

    code = S.main([
        "--cache", str(cache),
        "--train", str(train),
        "--test", str(test),
        "--stats", str(stats),
        "--memory-limit", "512MB",
    ])
    assert code == 0

    return {
        "train": pd.read_parquet(train),
        "test": pd.read_parquet(test),
        "stats": json.loads(stats.read_text(encoding="utf-8")),
        "train_path": train,
        "test_path": test,
    }


# --------------------------------------------------------------------------- #
# 1. 边界：两侧不交叉，日期全部落在正确区间
# --------------------------------------------------------------------------- #

def test_01_no_boundary_overlap(split_outputs):
    train, test = split_outputs["train"], split_outputs["test"]
    assert len(train) and len(test)

    # 训练集最晚一天必须早于测试集最早一天，且不共享任何 Complaint ID
    assert train["Date received"].max() < BOUNDARY
    assert test["Date received"].min() >= BOUNDARY
    assert train["Date received"].max() < test["Date received"].min()

    ids_train = set(train["Complaint ID"])
    ids_test = set(test["Complaint ID"])
    assert not (ids_train & ids_test)


def test_02_all_dates_in_range(split_outputs):
    train, test = split_outputs["train"], split_outputs["test"]
    assert (train["Date received"] <= pd.Timestamp("2023-12-31 23:59:59")).all()
    assert (test["Date received"] >= BOUNDARY).all()


def test_03_timestamp_on_last_day_stays_in_train(split_outputs):
    """2023-12-31T23:59:59 属于训练集，2024-01-01T00:00:00 属于测试集。"""
    train, test = split_outputs["train"], split_outputs["test"]
    assert 1 in set(train["Complaint ID"])
    assert 2 in set(test["Complaint ID"])


# --------------------------------------------------------------------------- #
# 2. 清洗规则
# --------------------------------------------------------------------------- #

def test_04_only_three_classes_and_no_dirty_values(split_outputs):
    for side in ("train", "test"):
        df = split_outputs[side]
        assert set(df["Company response to consumer"].unique()) <= set(C.TARGET_CLASSES)
        # State / Submitted via 不允许有空值或纯空白
        assert df["State"].isna().sum() == 0
        assert (df["State"].astype(str).str.strip() == "").sum() == 0
        assert df["Submitted via"].isna().sum() == 0


def test_05_product_is_normalized(split_outputs):
    """旧命名必须被合并，输出里不能再出现 PRODUCT_NORMALIZATION 的键。"""
    old_names = set(C.PRODUCT_NORMALIZATION)
    for side in ("train", "test"):
        products = set(split_outputs[side]["Product"].astype(str))
        assert not (products & old_names), f"{side} 里还有旧命名：{products & old_names}"

    train = split_outputs["train"]
    assert "Checking or savings account" in set(train["Product"].astype(str))
    assert set(split_outputs["test"]["Product"].astype(str)) >= {"Credit card"}


def test_06_duplicate_complaint_id_deduplicated(split_outputs):
    train = split_outputs["train"]
    assert train["Complaint ID"].is_unique
    assert len(train) == 2                     # id=1 出现了两次，去重后剩一条


def test_07_blank_state_becomes_unknown(split_outputs):
    train = split_outputs["train"]
    row = train[train["Complaint ID"] == 4].iloc[0]
    assert row["State"] == "Unknown"           # 原值为纯空格


# --------------------------------------------------------------------------- #
# 3. 与 prototype / cleaning 的一致性（防止两份实现漂移）
# --------------------------------------------------------------------------- #

def test_08_split_and_prototype_share_one_cleaning_implementation():
    """split_temporal、prototype 都用 cleaning.clean_data，且没有自己的副本。"""
    assert S.clean_data is C.clean_data
    assert P.clean_data is C.clean_data
    assert S.TARGET == C.TARGET and P.TARGET == C.TARGET
    assert S.TARGET_CLASSES == C.TARGET_CLASSES == P.TARGET_CLASSES
    assert S.FEATURES == C.FEATURES == P.FEATURES

    # prototype 里不应再残留自己的清洗实现或归一化表
    assert "PRODUCT_NORMALIZATION" not in vars(P)
    assert "normalize_product" not in vars(P)


def test_09_matches_clean_data_on_the_same_rows(split_outputs):
    """同样的原始行交给 cleaning.clean_data，结果应与落盘内容逐值一致。"""
    columns = ["Date received", "Product", "Issue", "Company", "State",
               "Submitted via", "Company response to consumer", "Complaint ID"]
    raw = pd.DataFrame([list(r) for r in ROWS], columns=columns)
    raw["Date received"] = (
        pd.to_datetime(raw["Date received"], format="mixed", utc=True).dt.tz_localize(None)
    )
    # 与 load_raw 的下推过滤保持一致：Product 为空的行在进 clean_data 之前就被丢掉
    raw = raw[raw["Product"].astype(str).str.strip() != ""]

    expected, _ = C.clean_data(raw)
    expected = expected[expected["Date received"] < BOUNDARY]

    got = split_outputs["train"]
    assert len(expected) == len(got) == 2

    expected = expected.sort_values("Complaint ID").reset_index(drop=True)
    got = got.sort_values("Complaint ID").reset_index(drop=True)
    for col in ("Product", "State", "Submitted via", "Company response to consumer"):
        pd.testing.assert_series_equal(
            expected[col].astype(str), got[col].astype(str), check_names=False
        )


# --------------------------------------------------------------------------- #
# 4. 统计口径
# --------------------------------------------------------------------------- #

def test_10_totals_account_for_every_row(split_outputs):
    t = split_outputs["stats"]["totals"]
    assert t["cache_rows"] == len(ROWS)
    assert t["rows"] == len(split_outputs["train"]) + len(split_outputs["test"])
    assert t["no_product"] == 1
    # 7 行 = 4 行有效 + 1 无 Product + 1 非三主类 + 1 同 ID 去重
    assert t["other_classes"] == 2


def test_11_class_shares_sum_to_one(split_outputs):
    for side in ("train", "test"):
        classes = split_outputs["stats"][side]["classes"]
        assert sum(v["count"] for v in classes.values()) == split_outputs["stats"][side]["rows"]
        assert sum(v["share"] for v in classes.values()) == pytest.approx(1.0, abs=1e-6)


def test_12_company_overlap_numbers():
    train_counts = pd.Series({"A": 3, "B": 1})
    test_counts = pd.Series({"A": 2, "C": 5})
    ov = S.company_overlap(train_counts, test_counts)

    assert ov["n_train_companies"] == 2
    assert ov["n_test_companies"] == 2
    assert ov["n_both"] == 1
    assert ov["share_of_test"] == 0.5
    assert ov["jaccard"] == pytest.approx(1 / 3, abs=1e-6)
    assert ov["test_rows_from_seen_company"] == 2
    assert ov["test_rows_from_seen_company_share"] == pytest.approx(2 / 7, abs=1e-6)


def test_13_yearly_table_marks_split_and_computes_rate():
    train = {"year_counts": pd.DataFrame({EXP: [9, 0], NONMON: [0, 0], MON: [1, 0]},
                                         index=pd.Index([2022, 2023], name="y"))}
    test = {"year_counts": pd.DataFrame({EXP: [0], NONMON: [2], MON: [2]},
                                        index=pd.Index([2024], name="y"))}
    rows = S.yearly_table(train, test)

    assert [r["year"] for r in rows] == [2022, 2023, 2024]
    assert [r["split"] for r in rows] == ["train", "train", "test"]
    assert rows[0]["complaints"] == 10 and rows[0]["monetary_count"] == 1
    assert rows[0]["monetary_share"] == pytest.approx(0.1, abs=1e-6)
    assert rows[2]["monetary_share"] == pytest.approx(0.5, abs=1e-6)
    assert rows[2]["explanation_share"] == 0.0
