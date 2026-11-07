#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""src/eda.py 的自动化测试。

测试覆盖：

  1. ``truncate`` 截断逻辑（不截断短串、截断长串加 …）；
  2. ``normalize_product_name`` 与 ``cleaning.PRODUCT_NORMALIZATION`` 一致；
  3. 归一化后的聚合计数正确（模拟 DuckDB 查询结果）；
  4. 三主类顺序与 ``cleaning.TARGET_CLASSES`` 一致（防止顺序漂移）；
  5. ``load_stats`` 能正确加载 JSON 并返回预期字段。

全部测试数据在本文件内手工构造，不依赖 data/ 下的真实文件。

运行::

    pytest -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import eda  # noqa: E402
from cleaning import PRODUCT_NORMALIZATION, TARGET_CLASSES  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. truncate
# --------------------------------------------------------------------------- #

class TestTruncate:
    def test_short_string_unchanged(self):
        """短于阈值的字符串原样返回。"""
        assert eda.truncate("短", 28) == "短"
        assert eda.truncate("hello world", 28) == "hello world"
        assert eda.truncate("", 28) == ""

    def test_long_string_truncated(self):
        """超过阈值的字符串截断并加 …。"""
        long = "A" * 30
        result = eda.truncate(long, 28)
        assert len(result) == 29  # 28 chars + …
        assert result.endswith("…")
        assert result[:-1] == "A" * 28

    def test_exact_boundary(self):
        """恰好等于 n 或 n+1 长度时不截断（只有 > n+1 才截断）。"""
        # len == n → 不截断
        s_n = "B" * 28
        assert eda.truncate(s_n, 28) == "B" * 28
        # len == n+1 → 不截断（边界恰好不触发）
        s_n1 = "B" * 29
        assert eda.truncate(s_n1, 28) == "B" * 29
        # len == n+2 → 截断
        s_n2 = "B" * 30
        result = eda.truncate(s_n2, 28)
        assert result == "B" * 28 + "…"

    def test_unicode_truncation(self):
        """中文字符按字符数截断，不按字节。"""
        s = "测试" * 20  # 40 chars
        result = eda.truncate(s, 10)
        assert result.endswith("…")
        assert result[:-1] == "测试" * 5  # 10 chars


# --------------------------------------------------------------------------- #
# 2. normalize_product_name
# --------------------------------------------------------------------------- #

class TestNormalizeProductName:
    def test_known_old_names_normalized(self):
        """PRODUCT_NORMALIZATION 中的旧命名全部归一化。"""
        for old_key, (new_name, reason) in PRODUCT_NORMALIZATION.items():
            # old_key 是小写键，实际数据里可能是原始大小写
            # normalize_product_name 内部做 .strip().lower() 再查表
            result = eda.normalize_product_name(old_key.upper())
            assert result == new_name, (
                f"归一化 '{old_key}' 应得到 '{new_name}'，实际得到 '{result}'"
            )

    def test_unknown_name_unchanged(self):
        """不在归一化表中的产品名原样返回（仅 strip）。"""
        assert eda.normalize_product_name("Mortgage") == "Mortgage"
        assert eda.normalize_product_name("Debt collection") == "Debt collection"
        # 带首尾空白
        assert eda.normalize_product_name("  Mortgage  ") == "Mortgage"

    def test_case_insensitive(self):
        """大小写不影响归一化结果。"""
        assert eda.normalize_product_name("Credit reporting") == \
            eda.normalize_product_name("CREDIT REPORTING") == \
            eda.normalize_product_name("credit reporting")

    def test_consistent_with_cleaning(self):
        """与 cleaning.normalize_product 的映射目标完全一致。"""
        # 逐一核对：eda 的归一化结果应等于 cleaning 里对应的目标名称
        for old_key, (new_name, _) in PRODUCT_NORMALIZATION.items():
            assert eda.normalize_product_name(old_key) == new_name


# --------------------------------------------------------------------------- #
# 3. 归一化后的聚合计数
# --------------------------------------------------------------------------- #

class TestProductAggregation:
    def test_normalization_then_aggregation(self):
        """模拟 DuckDB 查询结果：旧命名 + 新命名混合 → 归一化后正确合并。"""
        # 模拟 product × target 计数
        mock_df = pd.DataFrame([
            {"product": "Credit reporting", "target": "Closed with explanation", "n": 100},
            {"product": "Credit reporting", "target": "Closed with monetary relief", "n": 1},
            {"product": "Credit reporting or other personal consumer reports", "target": "Closed with explanation", "n": 500},
            {"product": "Credit reporting, credit repair services, or other personal consumer reports",
             "target": "Closed with explanation", "n": 200},
            {"product": "Mortgage", "target": "Closed with explanation", "n": 50},
            {"product": "Mortgage", "target": "Closed with monetary relief", "n": 10},
        ])

        mock_df["product_norm"] = mock_df["product"].apply(eda.normalize_product_name)
        agg = mock_df.groupby(["product_norm", "target"], observed=True)["n"].sum().reset_index()
        pivot = agg.pivot(index="product_norm", columns="target", values="n").fillna(0)

        # 三种征信类旧名应合并为一个
        cr_name = "Credit reporting or other personal consumer reports"
        assert cr_name in pivot.index
        cr_total = pivot.loc[cr_name].sum()
        assert cr_total == 801  # 100 + 1 + 500 + 200

        # Mortgage 不合并
        assert "Mortgage" in pivot.index
        assert pivot.loc["Mortgage"].sum() == 60  # 50 + 10

    def test_monetary_rate_calculation(self):
        """金钱救济率 = monetary_count / total * 100。"""
        mock_df = pd.DataFrame([
            {"product": "Credit card", "target": "Closed with explanation", "n": 80},
            {"product": "Credit card", "target": "Closed with non-monetary relief", "n": 10},
            {"product": "Credit card", "target": "Closed with monetary relief", "n": 10},
            {"product": "Credit card or prepaid card", "target": "Closed with explanation", "n": 90},
            {"product": "Credit card or prepaid card", "target": "Closed with monetary relief", "n": 10},
        ])
        mock_df["product_norm"] = mock_df["product"].apply(eda.normalize_product_name)
        agg = mock_df.groupby(["product_norm", "target"], observed=True)["n"].sum().reset_index()
        pivot = agg.pivot(index="product_norm", columns="target", values="n").fillna(0)
        pivot["total"] = pivot[TARGET_CLASSES].sum(axis=1)
        pivot["monetary_rate"] = pivot["Closed with monetary relief"] / pivot["total"] * 100

        # "Credit card" 和 "Credit card or prepaid card" 都归一化为 "Credit card"
        assert "Credit card" in pivot.index
        # 80 + 10 + 10 (Credit card) + 90 + 10 (Credit card or prepaid card) = 200
        assert pivot.loc["Credit card", "total"] == 200
        assert pivot.loc["Credit card", "Closed with monetary relief"] == 20
        assert abs(pivot.loc["Credit card", "monetary_rate"] - 20 / 200 * 100) < 0.01


# --------------------------------------------------------------------------- #
# 4. 三主类顺序一致性
# --------------------------------------------------------------------------- #

class TestClassOrder:
    def test_target_classes_order(self):
        """eda 模块导入的 TARGET_CLASSES 与 cleaning 一致。"""
        assert eda.TARGET_CLASSES == TARGET_CLASSES
        assert eda.TARGET_CLASSES == [
            "Closed with explanation",
            "Closed with non-monetary relief",
            "Closed with monetary relief",
        ]

    def test_short_labels_cover_all_classes(self):
        """SHORT 映射覆盖所有三个目标类别。"""
        for c in TARGET_CLASSES:
            assert c in eda.SHORT
            assert isinstance(eda.SHORT[c], str)


# --------------------------------------------------------------------------- #
# 5. load_stats（用临时 JSON 验证）
# --------------------------------------------------------------------------- #

class TestLoadStats:
    def test_load_stats_with_mock_files(self, tmp_path):
        """load_stats 能正确读取 JSON 并返回预期字段。"""
        mock_full = {
            "totals": {"valid_rows": 100, "raw_rows": 110},
            "top_companies": [
                {"company": "Test Co", "complaints": 50,
                 "closed_with_monetary_relief": {"share_of_company": 0.1}},
            ],
            "missing_rates": [
                {"column": "narrative", "missing": 77, "missing_rate": 0.77},
            ],
        }
        mock_split = {
            "yearly": [{"year": 2020, "complaints": 100, "monetary_share": 0.05}],
            "train": {"classes": {
                "Closed with explanation": {"share": 0.7},
            }},
            "test": {"classes": {
                "Closed with explanation": {"share": 0.6},
            }},
        }

        full_path = tmp_path / "full_stats.json"
        split_path = tmp_path / "split_stats.json"
        full_path.write_text(json.dumps(mock_full), encoding="utf-8")
        split_path.write_text(json.dumps(mock_split), encoding="utf-8")

        # 临时替换路径
        orig_full, orig_split = eda.FULL_STATS, eda.SPLIT_STATS
        eda.FULL_STATS = full_path
        eda.SPLIT_STATS = split_path
        try:
            full, split = eda.load_stats()
            assert full["totals"]["valid_rows"] == 100
            assert full["top_companies"][0]["company"] == "Test Co"
            assert split["yearly"][0]["year"] == 2020
            assert split["train"]["classes"]["Closed with explanation"]["share"] == 0.7
        finally:
            eda.FULL_STATS = orig_full
            eda.SPLIT_STATS = orig_split
