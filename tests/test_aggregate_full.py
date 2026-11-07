#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""src/aggregate_full.py 的自动化测试。

用一份 7 行、手工构造的迷你 CSV 跑通整条流水线（含 parquet 缓存），
再逐项核对 JSON 里的统计结果。数据里刻意塞了几个坑：

  * ``Consumer complaint narrative`` 含换行与逗号（跨行记录）；
  * ``Date received`` 混用纯日期与带时间戳两种写法；
  * 一行目标变量为空、一行 Product 为空（都算无效行）；
  * ``State`` 为纯空白串。

运行::

    pytest -v
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import aggregate_full as A  # noqa: E402

EXP = "Closed with explanation"
NONMON = "Closed with non-monetary relief"
MON = "Closed with monetary relief"

HEADER = [
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
    "Consumer complaint narrative", "Company public response", "Company", "State",
    "ZIP code", "Tags", "Submitted via", "Date sent to company",
    "Company response to consumer", "Timely response?", "Complaint ID",
]

# (日期, Product, Company, State, 响应, Complaint ID, 叙述)
FULL_ROWS = [
    ("2019-03-04", "Credit reporting", "Company A", "CA", EXP, "1", "太短的账户信息"),
    ("2020-05-06T10:00:00.000Z", "Credit reporting", "Company A", "TX", NONMON, "2", ""),
    ("2021-01-01", "Mortgage", "Company B", "CA", MON, "3", "叙述里既有逗号, 也有换行\n第二段"),
    ("2021-06-01", "Mortgage", "Company B", "", EXP, "4", ""),
    ("2022-01-01", "Debt collection", "Company C", "NY", "", "5", ""),
    ("2022-02-01", "", "Company C", "NY", EXP, "6", ""),
    ("2023-01-01", "Debt collection", "Company C", "NY", "In progress", "7", ""),
]

# 样本只保留原数据里的有效行，并且刻意少放一行（不含 In progress），
# 好在第 6 个测试里验证「样本 vs 全量」确实算得出偏差
SAMPLE_ROWS = FULL_ROWS[:4]


def write_csv(path: Path, rows: list[tuple], columns: list[str]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for date, product, company, state, response, cid, narrative in rows:
            writer.writerow([
                date, product, "Sub", "Issue", "Sub-issue", narrative,
                "Public response" if cid == "1" else "", company, state, "38141",
                "", "Web", date, response, "Yes", cid,
            ])
    return path


@pytest.fixture(scope="module")
def stats(tmp_path_factory) -> dict:
    """跑一遍完整流水线，返回解析后的 full_stats.json。"""
    tmp = tmp_path_factory.mktemp("cfpb")
    source = write_csv(tmp / "complaints.csv", FULL_ROWS, HEADER)
    sample = write_csv(tmp / "sample.csv", SAMPLE_ROWS, HEADER)
    out = tmp / "full_stats.json"

    code = A.main([
        "--input", str(source),
        "--sample", str(sample),
        "--cache", str(tmp / "cache" / "slim.parquet"),
        "--output", str(out),
        "--memory-limit", "512MB",
    ])
    assert code == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_01_totals_and_valid_rows(stats):
    """7 行里 2 行无效（目标变量为空 / Product 为空），有效 5 行。"""
    t = stats["totals"]
    assert t["raw_rows"] == 7
    assert t["valid_rows"] == 5
    assert t["invalid_rows"] == 2
    assert t["n_target"] == 4          # explanation / non-monetary / monetary / In progress
    assert t["n_product"] == 3


def test_02_date_formats_are_both_parsed(stats):
    """两种日期写法都要被识别，且没有解析失败的行。"""
    d = stats["date_format_check"]
    assert d["n_rows"] == 7
    assert d["n_with_time"] == 1
    assert d["n_date_only"] == 6
    assert d["n_unparsable"] == 0

    years = {row["year"]: row["count"] for row in stats["year_distribution"]}
    assert years == {2019: 1, 2020: 1, 2021: 2, 2023: 1}   # 无效行的年份不计入


def test_03_target_and_product_distribution(stats):
    """分布计数与手算一致，占比分母是有效行（5 行）。"""
    target = {row["category"]: row["count"] for row in stats["target_distribution"]}
    assert target == {EXP: 2, NONMON: 1, MON: 1, "In progress": 1}

    explanation = next(r for r in stats["target_distribution"] if r["category"] == EXP)
    assert explanation["share_of_valid"] == 0.4
    assert explanation["share_of_raw"] == round(2 / 7, 6)

    product = {row["product"]: row["count"] for row in stats["product_distribution"]}
    assert product == {"Credit reporting": 2, "Mortgage": 2, "Debt collection": 1}
    assert [r["product"] for r in stats["product_distribution"]][:1] == ["Credit reporting"]


def test_04_company_shares_sum_to_one(stats):
    """三主类 + 其他 = 100%，且按投诉量降序、并列时按公司名升序。"""
    companies = stats["top_companies"]
    assert len(companies) == 3
    assert [c["company"] for c in companies] == ["Company A", "Company B", "Company C"]

    company_a = companies[0]
    assert company_a["complaints"] == 2
    assert company_a["closed_with_explanation"]["share_of_company"] == 0.5
    assert company_a["closed_with_non_monetary_relief"]["share_of_company"] == 0.5

    for c in companies:
        total = (
            c["closed_with_explanation"]["share_of_company"]
            + c["closed_with_non_monetary_relief"]["share_of_company"]
            + c["closed_with_monetary_relief"]["share_of_company"]
            + c["other_responses"]["share_of_company"]
        )
        assert total == pytest.approx(1.0, abs=1e-6)


def test_05_missing_rates_and_narrative(stats):
    """缺失率按全量行算；叙述列非空 2 行（含跨行那条）。"""
    missing = {row["column"]: row["missing"] for row in stats["missing_rates"]}
    assert set(missing) == set(HEADER)          # 16 列全都要有
    assert missing["Company response to consumer"] == 1
    assert missing["Product"] == 1
    assert missing["State"] == 1                # 纯空白算缺失
    assert missing["Consumer complaint narrative"] == 5

    assert stats["narrative"]["non_empty"] == 2
    assert stats["narrative"]["share_of_raw"] == round(2 / 7, 6)

    assert [row["state"] for row in stats["state_top15"]] == ["CA", "NY", "TX"]


def test_06_sample_vs_full_deviation(stats):
    """样本 4 行（2/1/1），全量有效 5 行（2/1/1/1），偏差按百分点给。"""
    comp = stats["sample_vs_full"]
    assert comp["sample"]["valid_rows"] == 4

    rows = {r["category"]: r for r in comp["target_main_classes"]}
    assert rows[EXP]["sample_share"] == 0.5
    assert rows[EXP]["full_share"] == 0.4
    assert rows[EXP]["diff_pp"] == pytest.approx(10.0, abs=1e-4)

    product = {r["product"]: r for r in comp["product_top10"]}
    # 全量：Credit reporting 40% / Mortgage 40% / Debt collection 20%
    # 样本：Credit reporting 50% / Mortgage 50% / 没有 Debt collection
    assert product["Credit reporting"]["diff_pp"] == pytest.approx(10.0, abs=1e-4)
    assert product["Debt collection"]["diff_pp"] == pytest.approx(-20.0, abs=1e-4)

    assert comp["summary"]["target_max_abs_diff_pp"] == pytest.approx(10.0, abs=1e-4)
    assert comp["summary"]["product_top10_max_abs_diff_pp"] == pytest.approx(20.0, abs=1e-4)
