#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CFPB 消费者投诉数据 —— 全量聚合统计。

对 ``data/complaints.csv``（约 8.4 GB）做一次全量扫描，产出 ``data/full_stats.json``：
总行数 / 有效行数、目标变量 8 类分布、按年份投诉量、Product 全类分布、
前 30 家公司及三主类占比、各列缺失率、公开叙述行数、State 前 15，
并与 ``data/sample.csv``（20 万行）对比三主类与 Product 前 10 的占比偏差。

在项目根目录下运行::

    python src/aggregate_full.py              # 首次会建立 parquet 缓存，之后复用
    python src/aggregate_full.py --refresh    # 强制重建缓存（原始 CSV 变了才需要）

--------------------------------------------------------------------------
为什么换用 DuckDB（而不是继续用 pandas 分块）
--------------------------------------------------------------------------
1. **单次扫描出全部指标**。要算的东西有 9 组互不相同的分组聚合，pandas 分块
   必须为每一组在内存里维护一份 dict/Series 累加器，代码长且容易写错；DuckDB
   把原始 CSV 一次性转成列式 parquet 缓存后，每组聚合都只是一条 SQL 的
   ``GROUP BY``，逻辑与结果都更可核对。
2. **内存可控 + 溢写**。``SET memory_limit`` 硬上限 + ``temp_directory`` 允许落盘，
   峰值内存稳定在 GB 级，不会因为某个大分组（如 Company × 响应类型）炸掉。
3. **正确处理跨行记录（关键）**。本文件的 ``Consumer complaint narrative`` 里含有换行，
   16,545,376 条记录在磁盘上摊成了 27,656,645 个物理行，即大量「一条记录跨多行」。
   DuckDB 的并行 CSV 读取器不支持这种文件（会直接报 NotImplemented），所以这里显式
   ``parallel=false`` 让单线程 RFC4180 解析器来读——单线程实测约 440 MB/s，
   16.5M 行统计一遍只要 19 秒，完全可接受；而并行读取器则会把跨行记录切碎成
   错误的行（这类文件上用「按行切分 + 分块」的写法必然出错）。DuckDB 只在
   扫 CSV 时是单线程，之后所有聚合都跑在 parquet 上，仍然是 20 线程并行。
4. **缓存可复用**。parquet 缓存（去掉长文本，只留一个布尔位）后续做全量建模
   时可直接 ``read_parquet``，不必再扫一遍 8.4 GB。

--------------------------------------------------------------------------
关于 Date received 的两种格式
--------------------------------------------------------------------------
该列混有 ``YYYY-MM-DD``（约 93.6%）与 ``YYYY-MM-DDTHH:MM:SS.mmmZ``（约 6.4%）两种写法。
这里不走 pandas 的 ``format="mixed"``，而是用 DuckDB 的
``try_cast(date_received AS TIMESTAMP)``：它对两种格式都能正确解析，且解析失败返回
NULL 而不是抛错。脚本会额外统计「带时间戳 / 纯日期 / 无法解析」三者的行数，
把这条数据坑的实际情况直接打成表，便于验收。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from report import fmt_gb, fmt_int, fmt_pct, print_table

try:                      # 只是用来报告实测内存峰值，缺了不影响统计
    import psutil
except ImportError:       # pragma: no cover
    psutil = None

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = "data/complaints.csv"
SAMPLE_CSV = "data/sample.csv"
CACHE_PARQUET = "data/cache/complaints_slim.parquet"   # CSV 的单遍瘦身缓存
TMP_DIR = "data/cache/duckdb_tmp"                      # 内存不足时的溢写目录
OUTPUT_JSON = "data/full_stats.json"

TARGET = "Company response to consumer"
TARGET_CLASSES = [
    "Closed with explanation",
    "Closed with non-monetary relief",
    "Closed with monetary relief",
]

TOP_N_COMPANIES = 30
TOP_N_STATES = 15
TOP_N_PRODUCTS = 10

# 原始列名 -> parquet 缓存里的列别名（别名让后面的 SQL 可以不用到处写双引号）
COLUMN_ALIASES = {
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "has_narrative",   # 长文本压成一个布尔位
    "Company public response": "company_public_response",
    "Company": "company",
    "State": "state",
    "ZIP code": "zip_code",
    "Tags": "tags",
    "Submitted via": "submitted_via",
    "Date sent to company": "date_sent_to_company",
    "Company response to consumer": "target",
    "Timely response?": "timely_response",
    "Complaint ID": "complaint_id",
}

# 「有效行」定义：目标变量与 Product 均非空（空串视同空值）
VALID = (
    "target IS NOT NULL AND trim(target) <> '' "
    "AND product IS NOT NULL AND trim(product) <> ''"
)

MEMORY_LIMIT = "2GB"
MAX_TEMP_SIZE = "20GB"


# --------------------------------------------------------------------------- #
# 内存峰值采样（表格打印与数字格式化见 report.py）
# --------------------------------------------------------------------------- #

# 进程内存峰值（MB）：心跳线程与各阶段结束时各采样一次，用来看内存是否真的可控
_PEAK_RSS_MB: float = 0.0


def sample_rss() -> float | None:
    global _PEAK_RSS_MB
    if psutil is None:
        return None
    try:
        mb = psutil.Process().memory_info().rss / 1024 ** 2
    except Exception:      # pragma: no cover
        return None
    _PEAK_RSS_MB = max(_PEAK_RSS_MB, mb)
    return mb


class Heartbeat:
    """后台心跳线程：长时间只读扫描时定期打印已用时间（和已写出的文件大小）。"""

    def __init__(self, label: str, watch: Path | None = None, interval: float = 10.0):
        self.label = label
        self.watch = watch
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start = 0.0

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            elapsed = time.time() - self._start
            rss = sample_rss()
            extra = ""
            if self.watch is not None and self.watch.exists():
                extra = f"，已写出 {self.watch.stat().st_size / 1024 ** 2:,.0f} MB"
            if rss is not None:
                extra += f"，进程内存 {rss:,.0f} MB"
            print(f"      进行中 … 已用 {elapsed:5.1f}s{extra}", flush=True)

    def __enter__(self) -> "Heartbeat":
        self._start = time.time()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=1)


# --------------------------------------------------------------------------- #
# 1. 建立列式缓存（唯一一次全量 CSV 扫描）
# --------------------------------------------------------------------------- #

def build_cache(con: duckdb.DuckDBPyConnection, csv_path: Path, parquet_path: Path) -> tuple[float, bool]:
    """把 8.4 GB CSV 扫一遍写成 parquet：16 列全留，长文本压成布尔位。

    返回 (耗时秒数, 是否复用了已有缓存)。已存在缓存时直接跳过。
    """
    if parquet_path.exists():
        age_h = (time.time() - parquet_path.stat().st_mtime) / 3600
        print(f"  复用已有缓存：{parquet_path.name}"
              f"（{parquet_path.stat().st_size / 1024 ** 2:,.0f} MB，{age_h:.1f} 小时前生成）")
        print("  如需重建请加 --refresh")
        return 0.0, True

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists():
        parquet_path.unlink()

    select_cols = ",\n           ".join(
        ("(CASE WHEN \"Consumer complaint narrative\" IS NULL "
         "OR trim(\"Consumer complaint narrative\") = '' THEN FALSE ELSE TRUE END) AS has_narrative")
        if alias == "has_narrative" else f'"{raw}" AS {alias}'
        for raw, alias in COLUMN_ALIASES.items()
    )

    sql = f"""
    COPY (
        SELECT {select_cols}
        FROM read_csv('{csv_path.as_posix()}', header=true, all_varchar=true, parallel=false)
    ) TO '{parquet_path.as_posix()}'
      (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 122880)
    """

    print(f"  源文件 {fmt_gb(csv_path.stat().st_size)}，单线程解析 CSV 并写 parquet 缓存"
          f"（该文件有跨行记录，并行读会切错行）...")
    t0 = time.time()
    with Heartbeat("CSV → parquet", watch=parquet_path):
        con.execute(sql)
    elapsed = time.time() - t0
    print(f"  缓存完成：{parquet_path.name}"
          f"（{parquet_path.stat().st_size / 1024 ** 3:.2f} GB，耗时 {elapsed:.1f}s）")
    return elapsed, False


# --------------------------------------------------------------------------- #
# 2. 各指标查询
# --------------------------------------------------------------------------- #

def q_totals(con: duckdb.DuckDBPyConnection) -> dict:
    row = con.execute(f"""
        SELECT
            count(*)                                              AS raw_rows,
            count(*) FILTER (WHERE {VALID})                       AS valid_rows,
            count(DISTINCT complaint_id)                          AS n_complaint_id,
            count(DISTINCT company)                               AS n_company,
            count(DISTINCT trim(company))                         AS n_company_trimmed,
            count(DISTINCT product)                               AS n_product,
            count(DISTINCT trim(product))                         AS n_product_trimmed,
            count(DISTINCT target)                                AS n_target,
            count(DISTINCT trim(target))                          AS n_target_trimmed
        FROM c
    """).fetchone()
    keys = ["raw_rows", "valid_rows", "n_complaint_id", "n_company", "n_company_trimmed",
            "n_product", "n_product_trimmed", "n_target", "n_target_trimmed"]
    return dict(zip(keys, row))


def q_target_distribution(con, raw_rows: int, valid_rows: int) -> list[dict]:
    rows = con.execute(f"""
        SELECT target, count(*) AS n
        FROM c WHERE {VALID}
        GROUP BY 1
        ORDER BY n DESC, target
    """).fetchall()
    return [
        {"category": cat, "count": n,
         "share_of_valid": round(n / valid_rows, 6) if valid_rows else None,
         "share_of_raw": round(n / raw_rows, 6) if raw_rows else None}
        for cat, n in rows
    ]


def q_date_format(con) -> dict:
    """验证两种日期格式的实际占比，顺带确认没有解析不了的脏值。"""
    row = con.execute("""
        SELECT
            count(*)                                                   AS n_rows,
            count(*) FILTER (WHERE date_received LIKE '%T%')            AS n_with_time,
            count(*) FILTER (WHERE date_received IS NOT NULL
                               AND date_received NOT LIKE '%T%')        AS n_date_only,
            count(*) FILTER (WHERE date_received IS NULL)               AS n_null,
            count(*) FILTER (WHERE date_received IS NOT NULL
                               AND try_cast(date_received AS TIMESTAMP) IS NULL) AS n_unparsable
        FROM c
    """).fetchone()
    keys = ["n_rows", "n_with_time", "n_date_only", "n_null", "n_unparsable"]
    out = dict(zip(keys, row))
    n = out["n_rows"] or 1
    out["with_time_share"] = round(out["n_with_time"] / n, 6)
    out["date_only_share"] = round(out["n_date_only"] / n, 6)
    return out


def q_year_distribution(con) -> list[dict]:
    rows = con.execute(f"""
        SELECT year(try_cast(date_received AS TIMESTAMP)) AS yr,
               count(*)                                   AS n
        FROM c WHERE {VALID}
        GROUP BY 1
        ORDER BY 1 NULLS LAST
    """).fetchall()
    return [{"year": int(yr) if yr is not None else None, "count": n} for yr, n in rows]


def q_product_distribution(con, raw_rows: int, valid_rows: int) -> list[dict]:
    rows = con.execute(f"""
        SELECT product, count(*) AS n
        FROM c WHERE {VALID}
        GROUP BY 1
        ORDER BY n DESC, product
    """).fetchall()
    return [
        {"product": p, "count": n,
         "share_of_valid": round(n / valid_rows, 6) if valid_rows else None,
         "share_of_raw": round(n / raw_rows, 6) if raw_rows else None}
        for p, n in rows
    ]


def q_top_companies(con, valid_rows: int) -> list[dict]:
    """有效投诉量前 30 的公司，以及三主类在该公司有效投诉中的占比。

    占比分母 = 该公司全部有效投诉行数，所以三主类 + 其他回应类型合计为 100%。
    """
    rows = con.execute(f"""
        WITH agg AS (
            SELECT
                trim(company)                                       AS company,
                count(*)                                            AS n,
                count(*) FILTER (WHERE target = ?)                   AS n_exp,
                count(*) FILTER (WHERE target = ?)                   AS n_nonmon,
                count(*) FILTER (WHERE target = ?)                   AS n_mon
            FROM c WHERE {VALID}
            GROUP BY 1
        )
        SELECT company, n, n_exp, n_nonmon, n_mon,
               n - n_exp - n_nonmon - n_mon                         AS n_other
        FROM agg
        ORDER BY n DESC, company
        LIMIT {TOP_N_COMPANIES}
    """, TARGET_CLASSES).fetchall()

    out = []
    for rank, (company, n, n_exp, n_nonmon, n_mon, n_other) in enumerate(rows, start=1):
        out.append({
            "rank": rank,
            "company": company,
            "complaints": n,
            "share_of_valid": round(n / valid_rows, 6) if valid_rows else None,
            "closed_with_explanation": {"count": n_exp, "share_of_company": round(n_exp / n, 6)},
            "closed_with_non_monetary_relief": {"count": n_nonmon, "share_of_company": round(n_nonmon / n, 6)},
            "closed_with_monetary_relief": {"count": n_mon, "share_of_company": round(n_mon / n, 6)},
            "other_responses": {"count": n_other, "share_of_company": round(n_other / n, 6)},
        })
    return out


def q_missing_rates(con, raw_rows: int) -> list[dict]:
    """全部 16 列的缺失率（NULL 或纯空白串都算缺失，分母为全量行数）。"""
    selects = []
    for alias in COLUMN_ALIASES.values():
        if alias == "has_narrative":
            selects.append("count(*) FILTER (WHERE NOT has_narrative) AS m_has_narrative")
        else:
            selects.append(f"count(*) FILTER (WHERE {alias} IS NULL OR trim({alias}) = '') AS m_{alias}")
    row = con.execute("SELECT " + ",\n               ".join(selects) + " FROM c").fetchone()

    out = []
    for (raw, alias), n_missing in zip(COLUMN_ALIASES.items(), row):
        out.append({
            "column": raw,
            "missing": n_missing,
            "missing_rate": round(n_missing / raw_rows, 6) if raw_rows else None,
        })
    return out


def q_narrative(con, raw_rows: int, valid_rows: int) -> dict:
    n_non_empty = con.execute(
        "SELECT count(*) FILTER (WHERE has_narrative) FROM c"
    ).fetchone()[0]
    return {
        "non_empty": n_non_empty,
        "share_of_raw": round(n_non_empty / raw_rows, 6) if raw_rows else None,
        "share_of_valid": round(n_non_empty / valid_rows, 6) if valid_rows else None,
    }


def q_state_top(con, valid_rows: int) -> list[dict]:
    rows = con.execute(f"""
        SELECT trim(state) AS st, count(*) AS n
        FROM c
        WHERE {VALID} AND state IS NOT NULL AND trim(state) <> ''
        GROUP BY 1
        ORDER BY n DESC, st
        LIMIT {TOP_N_STATES}
    """).fetchall()
    return [
        {"state": st, "count": n,
         "share_of_valid": round(n / valid_rows, 6) if valid_rows else None}
        for st, n in rows
    ]


# --------------------------------------------------------------------------- #
# 3. 样本 vs 全量
# --------------------------------------------------------------------------- #

def load_sample(con: duckdb.DuckDBPyConnection, sample_path: Path) -> dict:
    """用同一套口径统计样本：总行数、有效行数、目标分布、Product 分布。"""
    con.execute(f"""
        CREATE OR REPLACE VIEW s AS
        SELECT "Date received"                  AS date_received,
               "Product"                        AS product,
               "Issue"                          AS issue,
               "Company"                        AS company,
               "State"                          AS state,
               "Submitted via"                  AS submitted_via,
               "Company response to consumer"   AS target,
               "Timely response?"               AS timely_response,
               "Complaint ID"                   AS complaint_id
        FROM read_csv('{sample_path.as_posix()}', header=true, all_varchar=true)
    """)

    raw_rows, valid_rows = con.execute(
        f"SELECT count(*), count(*) FILTER (WHERE {VALID}) FROM s"
    ).fetchone()

    target = dict(con.execute(
        f"SELECT target, count(*) FROM s WHERE {VALID} GROUP BY 1"
    ).fetchall())
    product = dict(con.execute(
        f"SELECT product, count(*) FROM s WHERE {VALID} GROUP BY 1"
    ).fetchall())

    return {
        "path": str(sample_path),
        "raw_rows": raw_rows,
        "valid_rows": valid_rows,
        "target_counts": target,
        "product_counts": product,
    }


def compare(sample: dict, full_target: list[dict], full_product: list[dict]) -> dict:
    """三主类 + Product 前 10 的占比偏差（绝对百分点）。

    两侧分母统一为「有效行」：样本是用 `dropna(target, product)` 抽出来的，
    拿它和全量有效行比才是同一人群。偏差 = 样本占比 - 全量占比。
    """
    s_n = sample["valid_rows"] or 1

    target_rows = []
    for cat in TARGET_CLASSES:
        s = sample["target_counts"].get(cat, 0) / s_n
        f = next((d["share_of_valid"] for d in full_target if d["category"] == cat), 0.0)
        target_rows.append({
            "category": cat,
            "sample_share": round(s, 6),
            "full_share": round(f, 6),
            "diff_pp": round((s - f) * 100, 4),
        })

    product_rows = []
    for d in full_product[:TOP_N_PRODUCTS]:
        prod = d["product"]
        s = sample["product_counts"].get(prod, 0) / s_n
        f = d["share_of_valid"] or 0.0
        product_rows.append({
            "product": prod,
            "sample_share": round(s, 6),
            "full_share": round(f, 6),
            "diff_pp": round((s - f) * 100, 4),
        })

    return {
        "sample": {"path": sample["path"], "raw_rows": sample["raw_rows"],
                   "valid_rows": sample["valid_rows"]},
        "target_main_classes": target_rows,
        "product_top10": product_rows,
        "summary": {
            "target_max_abs_diff_pp": max(abs(r["diff_pp"]) for r in target_rows),
            "product_top10_max_abs_diff_pp": max(abs(r["diff_pp"]) for r in product_rows),
        },
    }


# --------------------------------------------------------------------------- #
# 4. 报告
# --------------------------------------------------------------------------- #

def print_report(stats: dict, comparison: dict) -> None:
    t = stats["totals"]
    line = "=" * 78

    print(f"\n{line}\n[2/9] 总行数 / 有效行数\n{line}")
    print_table(
        ["指标", "数值"],
        [
            ["原始总行数", fmt_int(t["raw_rows"])],
            ["有效行数（目标变量 & Product 非空）", fmt_int(t["valid_rows"])],
            ["无效行数", fmt_int(t["raw_rows"] - t["valid_rows"])],
            ["无效行占比", fmt_pct((t["raw_rows"] - t["valid_rows"]) / t["raw_rows"] if t["raw_rows"] else 0)],
            ["不重复 Complaint ID 数", fmt_int(t["n_complaint_id"])],
            ["公司数", fmt_int(t["n_company"])],
            ["Product 类别数（全量，含目标变量缺失行）", fmt_int(t["n_product"])],
            ["目标变量取值数", fmt_int(t["n_target"])],
        ],
        "lr",
    )
    if t["n_company"] != t["n_company_trimmed"]:
        print(f"  注意：公司名去首尾空白后类别数 {t['n_company']:,} → {t['n_company_trimmed']:,}，"
              f"下表按去空白后的名称聚合。")
    if t["n_target"] != t["n_target_trimmed"]:
        print(f"  注意：目标变量取值存在首尾空白差异（{t['n_target']} → {t['n_target_trimmed']}），"
              f"请人工确认。")
    if t["n_complaint_id"] < t["raw_rows"]:
        gap = t["raw_rows"] - t["n_complaint_id"]
        print(f"  注意：有 {fmt_int(gap)} 行 Complaint ID 缺失，"
              f"本统计按行计、未按 ID 去重（仅占 {fmt_pct(gap / t['raw_rows'], 3)}，影响可忽略）。")

    print(f"\n{line}\n[3/9] 目标变量类别分布（共 {len(stats['target_distribution'])} 类，"
          f"分母 = 有效行）\n{line}")
    print_table(
        ["Company response to consumer", "数量", "占有效行", "占全量"],
        [[d["category"], fmt_int(d["count"]), fmt_pct(d["share_of_valid"]), fmt_pct(d["share_of_raw"])]
         for d in stats["target_distribution"]],
        "lrrr",
    )
    main3 = [d for d in stats["target_distribution"] if d["category"] in TARGET_CLASSES]
    main3_share = sum(d["share_of_valid"] or 0 for d in main3)
    print(f"  三主类合计：{fmt_int(sum(d['count'] for d in main3))} 行，占有效行 {fmt_pct(main3_share)}")

    d = stats["date_format_check"]
    print(f"\n{line}\n[4/9] Date received 格式核对与按年份投诉量\n{line}")
    print_table(
        ["写法", "行数", "占比"],
        [
            ["带时间戳（YYYY-MM-DDTHH:MM:SS.mmmZ）", fmt_int(d["n_with_time"]), fmt_pct(d["with_time_share"])],
            ["纯日期（YYYY-MM-DD）", fmt_int(d["n_date_only"]), fmt_pct(d["date_only_share"])],
            ["空值", fmt_int(d["n_null"]), fmt_pct(d["n_null"] / d["n_rows"] if d["n_rows"] else 0)],
            ["用 try_cast 仍无法解析", fmt_int(d["n_unparsable"]), fmt_pct(d["n_unparsable"] / d["n_rows"] if d["n_rows"] else 0)],
        ],
        "lrr",
    )
    print("  （两种写法都能被 try_cast 解析 → 逐行解析不丢数；换成 pandas 默认的单格式推断，"
          f"少数派那 {fmt_int(min(d['n_with_time'], d['n_date_only']))} 行会整列变成 NaT）")
    print()
    print_table(
        ["年份", "投诉量（有效行）"],
        [[("无法解析" if y["year"] is None else y["year"]), fmt_int(y["count"])]
         for y in stats["year_distribution"]],
        "lr",
    )

    print(f"\n{line}\n[5/9] Product 全量类别分布（原始值，未做旧命名归一化，"
          f"有效行内 {len(stats['product_distribution'])} 类）\n{line}")
    print_table(
        ["Product", "数量", "占有效行"],
        [[p["product"], fmt_int(p["count"]), fmt_pct(p["share_of_valid"])]
         for p in stats["product_distribution"]],
        "lrr",
    )

    print(f"\n{line}\n[6/9] 投诉量前 {TOP_N_COMPANIES} 的公司（占比分母 = 该公司有效投诉量）\n{line}")
    print_table(
        ["#", "Company", "投诉量", "占全量", "explanation", "non-monetary", "monetary", "其他"],
        [[c["rank"], c["company"], fmt_int(c["complaints"]), fmt_pct(c["share_of_valid"]),
          fmt_pct(c["closed_with_explanation"]["share_of_company"]),
          fmt_pct(c["closed_with_non_monetary_relief"]["share_of_company"]),
          fmt_pct(c["closed_with_monetary_relief"]["share_of_company"]),
          fmt_pct(c["other_responses"]["share_of_company"])]
         for c in stats["top_companies"]],
        "rlrrrrrr",
    )

    print(f"\n{line}\n[7/9] 各列缺失率（NULL 或纯空白，分母 = 全量 {fmt_int(t['raw_rows'])} 行）\n{line}")
    print_table(
        ["列名", "缺失数", "缺失率"],
        [[m["column"], fmt_int(m["missing"]), fmt_pct(m["missing_rate"])]
         for m in stats["missing_rates"]],
        "lrr",
    )

    nar = stats["narrative"]
    print(f"\n{line}\n[8/9] Consumer complaint narrative 非空（公开同意）\n{line}")
    print_table(
        ["指标", "数值"],
        [
            ["非空（有公开叙述）行数", fmt_int(nar["non_empty"])],
            ["占全量", fmt_pct(nar["share_of_raw"])],
            ["占有效行", fmt_pct(nar["share_of_valid"])],
        ],
        "lr",
    )

    print(f"\n{line}\n[9/9] State 全量分布（前 {TOP_N_STATES}，占有效行）\n{line}")
    print_table(
        ["State", "数量", "占有效行"],
        [[s["state"], fmt_int(s["count"]), fmt_pct(s["share_of_valid"])]
         for s in stats["state_top15"]],
        "lrr",
    )

    # ---------------- 样本 vs 全量 ----------------
    print(f"\n{line}\n样本 vs 全量：占比偏差（绝对百分点，正 = 样本偏高）\n{line}")
    print(f"  样本 {comparison['sample']['path']}\n"
          f"  样本行数 {fmt_int(comparison['sample']['raw_rows'])}，"
          f"有效行 {fmt_int(comparison['sample']['valid_rows'])}")
    print()
    print_table(
        ["三主类", "样本占比", "全量占比", "偏差(pp)"],
        [[r["category"], fmt_pct(r["sample_share"]), fmt_pct(r["full_share"]), f"{r['diff_pp']:+.3f}"]
         for r in comparison["target_main_classes"]],
        "lrrr",
    )
    print()
    print_table(
        [f"Product 前 {TOP_N_PRODUCTS}", "样本占比", "全量占比", "偏差(pp)"],
        [[r["product"], fmt_pct(r["sample_share"]), fmt_pct(r["full_share"]), f"{r['diff_pp']:+.3f}"]
         for r in comparison["product_top10"]],
        "lrrr",
    )
    s = comparison["summary"]
    print(f"\n  三主类最大绝对偏差   : {s['target_max_abs_diff_pp']:.3f} pp")
    print(f"  Product 前 10 最大绝对偏差 : {s['product_top10_max_abs_diff_pp']:.3f} pp")

    # ---------------- 一致性自检 ----------------
    problems = []
    if t["valid_rows"] > t["raw_rows"]:
        problems.append("有效行数大于总行数")
    if sum(d["count"] for d in stats["target_distribution"]) != t["valid_rows"]:
        problems.append("目标变量计数之和 ≠ 有效行数")
    if sum(p["count"] for p in stats["product_distribution"]) != t["valid_rows"]:
        problems.append("Product 计数之和 ≠ 有效行数")
    print(f"\n{line}")
    print("一致性自检：" + ("通过（目标分布、Product 分布计数之和均等于有效行数）"
                        if not problems else "；".join(problems)))
    print(line)


# --------------------------------------------------------------------------- #
# 5. 入口
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="CFPB 全量投诉数据聚合统计")
    parser.add_argument("--input", default=INPUT_CSV, help=f"原始 CSV（默认 {INPUT_CSV}）")
    parser.add_argument("--sample", default=SAMPLE_CSV, help=f"样本 CSV（默认 {SAMPLE_CSV}）")
    parser.add_argument("--output", default=OUTPUT_JSON, help=f"输出 JSON（默认 {OUTPUT_JSON}）")
    parser.add_argument("--cache", default=CACHE_PARQUET, help=f"parquet 缓存（默认 {CACHE_PARQUET}）")
    parser.add_argument("--refresh", action="store_true", help="强制重建 parquet 缓存")
    parser.add_argument("--memory-limit", default=MEMORY_LIMIT, help=f"DuckDB 内存上限（默认 {MEMORY_LIMIT}）")
    args = parser.parse_args(argv)

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path

    csv_path, sample_path = resolve(args.input), resolve(args.sample)
    parquet_path, output_path = resolve(args.cache), resolve(args.output)
    tmp_dir = PROJECT_ROOT / TMP_DIR

    if not csv_path.exists():
        print(f"找不到原始数据：{csv_path}", file=sys.stderr)
        return 1
    if not sample_path.exists():
        print(f"找不到样本文件：{sample_path}（先运行 src/sample_data.py）", file=sys.stderr)
        return 1

    tmp_dir.mkdir(parents=True, exist_ok=True)
    if args.refresh and parquet_path.exists():
        print(f"按 --refresh 删除旧缓存：{parquet_path}")
        parquet_path.unlink()

    con = duckdb.connect()
    threads = min(20, (con.execute("SELECT current_setting('threads')").fetchone()[0] or 4))
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute(f"SET temp_directory='{tmp_dir.as_posix()}'")
    con.execute(f"SET max_temp_directory_size='{MAX_TEMP_SIZE}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")   # DuckDB 自带进度条会刷屏，改用下面的心跳

    line = "=" * 78
    print(line)
    print("CFPB 消费者投诉数据 —— 全量聚合")
    print(line)
    print(f"  输入   : {csv_path}（{fmt_gb(csv_path.stat().st_size)}）")
    print(f"  样本   : {sample_path}")
    print(f"  输出   : {output_path}")
    print(f"  引擎   : DuckDB {duckdb.__version__}，{threads} 线程，"
          f"内存上限 {args.memory_limit}，溢写目录 {tmp_dir}")

    print("\n[1/9] 建立列式缓存（本流程唯一一次全量 CSV 扫描）")
    cache_seconds, cache_reused = build_cache(con, csv_path, parquet_path)
    con.execute(f"CREATE OR REPLACE VIEW c AS SELECT * FROM read_parquet('{parquet_path.as_posix()}')")

    timings: dict[str, float] = {"build_cache": round(cache_seconds, 2)}
    t_start = time.time()

    t0 = time.time()
    totals = q_totals(con)
    timings["totals"] = round(time.time() - t0, 2)
    raw_rows, valid_rows = totals["raw_rows"], totals["valid_rows"]
    if valid_rows == 0:
        print("有效行数为 0，请检查 VALID 条件或数据内容。", file=sys.stderr)
        return 1

    t0 = time.time()
    target_dist = q_target_distribution(con, raw_rows, valid_rows)
    timings["target"] = round(time.time() - t0, 2)

    t0 = time.time()
    date_check = q_date_format(con)
    year_dist = q_year_distribution(con)
    timings["year"] = round(time.time() - t0, 2)

    t0 = time.time()
    product_dist = q_product_distribution(con, raw_rows, valid_rows)
    timings["product"] = round(time.time() - t0, 2)

    t0 = time.time()
    top_companies = q_top_companies(con, valid_rows)
    timings["companies"] = round(time.time() - t0, 2)

    t0 = time.time()
    missing = q_missing_rates(con, raw_rows)
    timings["missing"] = round(time.time() - t0, 2)

    t0 = time.time()
    narrative = q_narrative(con, raw_rows, valid_rows)
    timings["narrative"] = round(time.time() - t0, 2)

    t0 = time.time()
    state_top = q_state_top(con, valid_rows)
    timings["state"] = round(time.time() - t0, 2)

    print(f"\n  聚合完成，用时 {time.time() - t_start:.1f}s")

    sample_rss()   # 采一次峰值，写进 JSON 的 meta
    sample = load_sample(con, sample_path)
    comparison = compare(sample, target_dist, product_dist)

    stats = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "source_file": str(csv_path),
            "source_size_bytes": csv_path.stat().st_size,
            "source_size_gb": round(csv_path.stat().st_size / 1024 ** 3, 3),
            "engine": {"name": "duckdb", "version": duckdb.__version__,
                       "threads": threads, "memory_limit": args.memory_limit},
            "cache_parquet": str(parquet_path),
            "cache_reused": cache_reused,
            "peak_rss_mb": round(_PEAK_RSS_MB, 1) if _PEAK_RSS_MB else None,
            "definitions": {
                "valid_row": "Company response to consumer 与 Product 均非空（纯空白视同空值）",
                "missing": "NULL 或纯空白串",
                "target_share_denominator": "valid_rows",
            },
            "timings_seconds": timings,
        },
        "totals": {
            **totals,
            "invalid_rows": raw_rows - valid_rows,
            "valid_rate": round(valid_rows / raw_rows, 6) if raw_rows else None,
        },
        "target_distribution": target_dist,
        "date_format_check": date_check,
        "year_distribution": year_dist,
        "product_distribution": product_dist,
        "top_companies": top_companies,
        "missing_rates": missing,
        "narrative": narrative,
        "state_top15": state_top,
        "sample_vs_full": comparison,
    }

    # date_format_check 里有些字段是辅助用的，留在 JSON 里便于复核，已知无害
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print_report(stats, comparison)

    peak = _PEAK_RSS_MB or None
    if peak:
        print(f"\n进程内存峰值 {peak:,.0f} MB（DuckDB 内存上限 {args.memory_limit}）")
    print(f"输出：{output_path}"
          f"（{output_path.stat().st_size / 1024:.1f} KB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
