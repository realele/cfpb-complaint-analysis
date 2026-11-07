#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CFPB 投诉数据 —— 时间划分（替代随机 80/20）。

阶段一用 ``train_test_split`` 随机划分，同一家公司的历史投诉与未来投诉会被分到
两侧，模型等于"见过这家公司未来会怎么处理"——典型的**时间泄漏**。这里改成按
``Date received`` 切：训练集 ≡ 2023-12-31 及以前，测试集 ≡ 2024-01-01 及以后，
测试集在时间上完全晚于训练集，更接近上线时的真实场景。

数据来源是 ``data/cache/complaints_slim.parquet``（``src/aggregate_full.py`` 生成的
全量缓存，1627 万有效行），不再回头解析 8.4 GB 的 CSV。

清洗规则全部来自 ``src/cleaning.py``（与 ``prototype.py`` 同一份实现）：
三主类筛选、Complaint ID 去重、Product 旧命名归一化、State/Submitted via 空值填充。
本文件只负责"按时间分两侧 + 统计 + 落盘"。

在项目根目录下运行::

    python src/split_temporal.py

输出：
    data/train.parquet          训练集（2011-01 ~ 2023-12-31）
    data/test.parquet           测试集（2024-01-01 ~ 至今）
    data/split_stats.json       划分统计（供报告 / EDA 引用）

内存：两侧**依次**处理——训练集算完统计、写完盘就释放，再读测试集，因此不会
两份全量数据同时驻留。测试集约 1100 万行，实测峰值约 3 GB（读入时把字符串列
转成字典编码，否则光 pandas 的 object 列就要 7 GB）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa

from cleaning import FEATURES, TARGET, TARGET_CLASSES, clean_data
from report import fmt_gb, fmt_int, fmt_pct, print_table

try:                      # 只用于报告内存峰值，缺了不影响划分
    import psutil
except ImportError:       # pragma: no cover
    psutil = None

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE_PARQUET = "data/cache/complaints_slim.parquet"
TRAIN_PATH = "data/train.parquet"
TEST_PATH = "data/test.parquet"
STATS_JSON = "data/split_stats.json"

# 划分边界：训练集含 2023-12-31 全天（含带时间戳的 23:59），测试集从 2024-01-01 00:00 起
BOUNDARY = pd.Timestamp("2024-01-01 00:00:00")
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TRAIN_LAST_YEAR = 2023

EXP, NONMON, MON = TARGET_CLASSES

# 落盘保留的列：沿用原始列名，后续可直接按 prototype.FEATURES 取用
MODEL_COLUMNS = ["Complaint ID", "Date received"] + FEATURES + [TARGET]

MEMORY_LIMIT = "3GB"

_PEAK_RSS_MB = 0.0


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #

def sample_rss() -> float | None:
    """采样当前进程内存，并记录峰值（MB）。"""
    global _PEAK_RSS_MB
    if psutil is None:
        return None
    try:
        mb = psutil.Process().memory_info().rss / 1024 ** 2
    except Exception:      # pragma: no cover
        return None
    _PEAK_RSS_MB = max(_PEAK_RSS_MB, mb)
    return mb


def _to_pandas(tbl: pa.Table) -> pd.DataFrame:
    """Arrow 表 → pandas，同时把字符串列转字典编码。

    这一步是内存命门：1600 万行、6 个字符串列若按 object 落 pandas 要 7 GB 以上；
    转成字典编码（pandas 里即 category）后只要 1 GB 左右，后续 groupby 也更快。
    """
    for i, name in enumerate(tbl.column_names):
        if pa.types.is_string(tbl.schema.field(i).type):
            tbl = tbl.set_column(i, name, tbl.column(i).dictionary_encode())
    df = tbl.to_pandas()
    # Complaint ID 有 2000 多行缺失，arrow 的 int64+null 落到 pandas 会变成 float64，
    # 这里保持可空整型，免得 ID 被写成 8223945.0。
    df["Complaint ID"] = df["Complaint ID"].astype("Int64")
    return df


def _shrink(df: pd.DataFrame) -> pd.DataFrame:
    """内存收尾：object 列收回 category，并清掉筛选后残留的空类别。

    不清空类别的话，落盘后的 parquet 字典里还带着被剔除的取值
    （例如训练集里已经没有的 "In progress"），后续画图/取值会看着很怪。
    """
    for col in df.columns:
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            if col in ("Product", "State", "Submitted via", TARGET):
                df[col] = df[col].astype("category")
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].cat.remove_unused_categories()
    return df


# --------------------------------------------------------------------------- #
# 1. 读缓存
# --------------------------------------------------------------------------- #

def profile_cache(con: duckdb.DuckDBPyConnection, cache_path: Path) -> dict:
    """一次扫描拿到"读之前"的各项行数，用来说明哪些行没进划分。"""
    row = con.sql(f"""
        SELECT
            count(*)                                                         AS n_rows,
            count(*) FILTER (WHERE try_cast(date_received AS TIMESTAMP) IS NULL) AS n_bad_date,
            count(*) FILTER (WHERE product IS NULL OR trim(product) = '')    AS n_no_product
        FROM read_parquet('{cache_path.as_posix()}')
    """).fetchall()[0]
    return {"cache_rows": row[0], "bad_date": row[1], "no_product": row[2]}


def load_raw(
    con: duckdb.DuckDBPyConnection,
    cache_path: Path,
    side: str,
) -> pd.DataFrame:
    """从 parquet 缓存读出一侧数据（尚未清洗）。

    只下推两个 clean_data 不负责的过滤条件：日期必须能解析、Product 必须非空。
    三主类筛选、去重、Product 归一化、空值填充都留给 ``cleaning.clean_data``，
    保证与 prototype.py 用的是同一套规则。
    """
    if side == "train":
        date_cond = "ts < TIMESTAMP '2024-01-01 00:00:00'"
    elif side == "test":
        date_cond = "ts >= TIMESTAMP '2024-01-01 00:00:00'"
    else:
        raise ValueError(f"load_raw：side 只能是 'train' 或 'test'，收到 {side!r}")

    sql = f"""
        WITH src AS (
            SELECT
                try_cast(complaint_id AS BIGINT)         AS "Complaint ID",
                try_cast(date_received AS TIMESTAMP)     AS ts,
                product                                  AS "Product",
                issue                                    AS "Issue",
                company                                  AS "Company",
                state                                    AS "State",
                submitted_via                            AS "Submitted via",
                target                                   AS "Company response to consumer"
            FROM read_parquet('{cache_path.as_posix()}')
        )
        SELECT "Complaint ID", ts AS "Date received", "Product", "Issue", "Company",
               "State", "Submitted via", "Company response to consumer"
        FROM src
        WHERE ts IS NOT NULL
          AND "Product" IS NOT NULL AND trim("Product") <> ''
          AND {date_cond}
    """
    t0 = time.time()
    tbl = con.sql(sql).to_arrow_table()
    df = _to_pandas(tbl)
    del tbl
    peak = sample_rss()
    print(f"    读入 {side}: {len(df):,} 行（{time.time() - t0:.1f}s"
          f"{f'，内存 {peak:,.0f} MB' if peak else ''}）")
    return df


# --------------------------------------------------------------------------- #
# 2. 单侧处理（清洗 → 统计 → 落盘 → 释放）
# --------------------------------------------------------------------------- #

def summarize(df: pd.DataFrame, mapping: pd.DataFrame, read_seconds: float) -> dict:
    """从清洗后的一侧数据里取出统计量。只保留小额聚合结果，不保留 DataFrame。"""
    n = len(df)
    counts = df[TARGET].value_counts()
    companies = df["Company"].astype(str).value_counts()

    years = df["Date received"].dt.year
    by_year = df.groupby([years, df[TARGET]], observed=True).size().unstack(fill_value=0)
    for cls in TARGET_CLASSES:                 # 某些年份可能整类缺失
        if cls not in by_year.columns:
            by_year[cls] = 0
    by_year = by_year[TARGET_CLASSES]

    return {
        "rows": n,
        "date_min": str(df["Date received"].min().date()),
        "date_max": str(df["Date received"].max().date()),
        "read_seconds": read_seconds,
        "classes": {
            cls: {"count": int(counts.get(cls, 0)),
                  "share": round(float(counts.get(cls, 0)) / n, 6) if n else None}
            for cls in TARGET_CLASSES
        },
        "company_counts": companies,           # Series：公司 → 投诉量
        "year_counts": by_year,                # DataFrame：年份 × 三主类
        "product_mapping": mapping,
    }


def run_side(
    con: duckdb.DuckDBPyConnection,
    cache_path: Path,
    side: str,
    out_path: Path | None = None,
) -> dict:
    """读一侧 → 清洗（cleaning.clean_data）→ 汇总 → 可选落盘。返回汇总字典。"""
    label = "训练集" if side == "train" else "测试集"
    t0 = time.time()
    raw = load_raw(con, cache_path, side)
    read_seconds = time.time() - t0

    cleaned, mapping = clean_data(raw, label=label)
    del raw
    cleaned = _shrink(cleaned)
    sample_rss()

    summary = summarize(cleaned, mapping, read_seconds)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned[MODEL_COLUMNS].to_parquet(out_path, index=False, compression="zstd")
        summary["output"] = str(out_path)
        summary["output_mb"] = round(out_path.stat().st_size / 1024 ** 2, 1)

    del cleaned
    return summary


# --------------------------------------------------------------------------- #
# 3. 统计
# --------------------------------------------------------------------------- #

def company_overlap(train_counts: pd.Series, test_counts: pd.Series) -> dict:
    """公司集合重叠率 + 行级"已见公司"覆盖率。

    集合重叠率高，说明公司维度几乎没有泛化压力（测试集的公司训练集基本都见过）；
    行级覆盖率更贴近泄漏程度——它等于"测试集里有多少行来自训练集见过的公司"。
    """
    train_set = set(train_counts.index)
    test_set = set(test_counts.index)
    both = train_set & test_set
    union = train_set | test_set

    test_hit_rows = int(test_counts[test_counts.index.isin(both)].sum())
    train_hit_rows = int(train_counts[train_counts.index.isin(both)].sum())

    top_n = min(10, len(test_counts))
    top_test = test_counts.head(top_n)
    top_seen = int(sum(1 for c in top_test.index if c in train_set))

    return {
        "n_train_companies": len(train_set),
        "n_test_companies": len(test_set),
        "n_both": len(both),
        "share_of_train": round(len(both) / len(train_set), 6) if train_set else None,
        "share_of_test": round(len(both) / len(test_set), 6) if test_set else None,
        "jaccard": round(len(both) / len(union), 6) if union else None,
        "test_rows_from_seen_company": test_hit_rows,
        "test_rows_from_seen_company_share":
            round(test_hit_rows / int(test_counts.sum()), 6) if test_counts.sum() else None,
        "train_rows_from_shared_company_share":
            round(train_hit_rows / int(train_counts.sum()), 6) if train_counts.sum() else None,
        "test_top_companies_checked": top_n,
        "test_top_companies_seen_in_train": top_seen,
    }


def yearly_table(train: dict, test: dict) -> list[dict]:
    """合并两侧的「年份 × 三主类」计数，算出投诉量与三主类占比（EDA 用）。"""
    counts = pd.concat([train["year_counts"], test["year_counts"]]).sort_index()
    total = counts.sum(axis=1)

    rows = []
    for year, row in counts.iterrows():
        n = int(total.loc[year])
        rows.append({
            "year": int(year),
            "split": "train" if int(year) <= TRAIN_LAST_YEAR else "test",
            "complaints": n,
            "monetary_count": int(row[MON]),
            "explanation_share": round(float(row[EXP]) / n, 6) if n else None,
            "non_monetary_share": round(float(row[NONMON]) / n, 6) if n else None,
            "monetary_share": round(float(row[MON]) / n, 6) if n else None,
        })
    return rows


# --------------------------------------------------------------------------- #
# 4. 报告
# --------------------------------------------------------------------------- #

def print_report(stats: dict) -> None:
    line = "=" * 78
    tr, te = stats["train"], stats["test"]
    total = stats["totals"]["rows"]

    print(f"\n{line}\n[1/4] 划分规模\n{line}")
    print_table(
        ["集合", "时间范围", "行数", "占比"],
        [
            ["训练集", f"{tr['date_min']} ~ {tr['date_max']}", fmt_int(tr["rows"]),
             fmt_pct(tr["rows"] / total)],
            ["测试集", f"{te['date_min']} ~ {te['date_max']}", fmt_int(te["rows"]),
             fmt_pct(te["rows"] / total)],
            ["合计", "", fmt_int(total), "100.00%"],
        ],
        "llrr",
    )
    print(f"  缓存 {fmt_int(stats['totals']['cache_rows'])} 行 → 三主类 "
          f"{fmt_int(total)} 行；日期无法解析 {fmt_int(stats['totals']['bad_date'])} 行、"
          f"Product 为空 {fmt_int(stats['totals']['no_product'])} 行、"
          f"其余类别（含同 ID 去重）{fmt_int(stats['totals']['other_classes'])} 行")

    print(f"\n{line}\n[2/4] 三主类占比与分布漂移\n{line}")
    rows = []
    for cls in TARGET_CLASSES:
        a, b = tr["classes"][cls], te["classes"][cls]
        rows.append([cls, fmt_int(a["count"]), fmt_pct(a["share"]),
                     fmt_int(b["count"]), fmt_pct(b["share"]),
                     f"{(b['share'] - a['share']) * 100:+.2f}"])
    print_table(["类别", "训练行数", "训练占比", "测试行数", "测试占比", "漂移(pp)"],
                rows, "lrrrrr")

    ov = stats["company_overlap"]
    print(f"\n{line}\n[3/4] 公司集合重叠（时间划分与随机划分的关键差别）\n{line}")
    print_table(
        ["指标", "数值"],
        [
            ["训练集公司数", fmt_int(ov["n_train_companies"])],
            ["测试集公司数", fmt_int(ov["n_test_companies"])],
            ["两侧都出现的公司数", fmt_int(ov["n_both"])],
            ["占训练集公司数", fmt_pct(ov["share_of_train"])],
            ["占测试集公司数", fmt_pct(ov["share_of_test"])],
            ["Jaccard 重叠度", fmt_pct(ov["jaccard"])],
            ["测试集中来自「训练集已见公司」的行", fmt_int(ov["test_rows_from_seen_company"])],
            ["  该部分占测试集", fmt_pct(ov["test_rows_from_seen_company_share"])],
            ["测试集投诉量前 {} 的公司中训练集见过的".format(ov["test_top_companies_checked"]),
             f"{ov['test_top_companies_seen_in_train']} / {ov['test_top_companies_checked']}"],
        ],
        "lr",
    )

    print(f"\n{line}\n[4/4] 各年度金钱救济率（EDA 用）\n{line}")
    print_table(
        ["年份", "划分", "投诉量", "explanation", "non-monetary", "monetary", "金钱救济笔数"],
        [[str(r["year"]), "训练" if r["split"] == "train" else "测试",
          fmt_int(r["complaints"]), fmt_pct(r["explanation_share"]),
          fmt_pct(r["non_monetary_share"]), fmt_pct(r["monetary_share"]),
          fmt_int(r["monetary_count"])]
         for r in stats["yearly"]],
        "llrrrrr",
    )


# --------------------------------------------------------------------------- #
# 5. 入口
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="CFPB 全量数据的时间划分")
    parser.add_argument("--cache", default=CACHE_PARQUET,
                        help=f"parquet 缓存（默认 {CACHE_PARQUET}）")
    parser.add_argument("--train", default=TRAIN_PATH, help=f"训练集输出（默认 {TRAIN_PATH}）")
    parser.add_argument("--test", default=TEST_PATH, help=f"测试集输出（默认 {TEST_PATH}）")
    parser.add_argument("--stats", default=STATS_JSON, help=f"统计输出（默认 {STATS_JSON}）")
    parser.add_argument("--memory-limit", default=MEMORY_LIMIT,
                        help=f"DuckDB 内存上限（默认 {MEMORY_LIMIT}）")
    args = parser.parse_args(argv)

    def resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path

    cache_path, train_path = resolve(args.cache), resolve(args.train)
    test_path, stats_path = resolve(args.test), resolve(args.stats)

    if not cache_path.exists():
        print(f"找不到 parquet 缓存：{cache_path}\n"
              f"先运行 python src/aggregate_full.py 生成。", file=sys.stderr)
        return 1

    con = duckdb.connect()
    con.execute("SET threads=20")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute("SET enable_progress_bar=false")

    line = "=" * 78
    print(line)
    print("CFPB 投诉数据 —— 时间划分（训练 ≤ 2023-12-31 < 测试）")
    print(line)
    print(f"  缓存   : {cache_path}（{fmt_gb(cache_path.stat().st_size)}）")
    print(f"  输出   : {train_path.name} / {test_path.name} / {stats_path.name}")
    print(f"  引擎   : DuckDB {duckdb.__version__} + pandas {pd.__version__}")

    t_all = time.time()
    print(f"\n[1/4] 扫缓存、按时间切分（训练 < {BOUNDARY} ≤ 测试）")
    prof = profile_cache(con, cache_path)
    print(f"  缓存 {fmt_int(prof['cache_rows'])} 行；日期无法解析 "
          f"{fmt_int(prof['bad_date'])} 行；Product 为空 {fmt_int(prof['no_product'])} 行")

    print("  训练集：")
    train = run_side(con, cache_path, "train", train_path)
    print("  测试集：")
    test = run_side(con, cache_path, "test", test_path)
    sample_rss()

    # --- 划分自检：边界不能交叉，ID 不能串侧 ---
    if train["rows"] and test["rows"]:
        if not train["date_max"] < TEST_START <= test["date_min"]:
            raise ValueError(
                f"划分边界交叉：训练集最晚 {train['date_max']}，测试集最早 {test['date_min']}"
            )

    print("\n[2/4] 落盘")
    for s in (train, test):
        print(f"  {Path(s['output']).name}: {s['rows']:,} 行 × {len(MODEL_COLUMNS)} 列，"
              f"{s['output_mb']:,.0f} MB")

    print("\n[3/4] 统计")
    total_rows = train["rows"] + test["rows"]
    stats = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "cache_parquet": str(cache_path),
            "boundary": str(BOUNDARY),
            "train_ends": TRAIN_END,
            "test_starts": TEST_START,
            "engine": {"duckdb": duckdb.__version__, "pandas": pd.__version__},
            "peak_rss_mb": round(_PEAK_RSS_MB, 1) if _PEAK_RSS_MB else None,
            "elapsed_seconds": {"read_train": round(train["read_seconds"], 2),
                                "read_test": round(test["read_seconds"], 2),
                                "total": round(time.time() - t_all, 2)},
        },
        "totals": {
            "cache_rows": prof["cache_rows"],
            "rows": total_rows,
            "bad_date": prof["bad_date"],
            "no_product": prof["no_product"],
            "other_classes": prof["cache_rows"] - prof["bad_date"] - prof["no_product"] - total_rows,
        },
        "train": {k: train[k] for k in ("rows", "date_min", "date_max", "classes", "output")},
        "test": {k: test[k] for k in ("rows", "date_min", "date_max", "classes", "output")},
        "company_overlap": company_overlap(train["company_counts"], test["company_counts"]),
        "yearly": yearly_table(train, test),
        "product_mapping": train["product_mapping"].to_dict(orient="records"),
    }

    print_report(stats)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    peak = _PEAK_RSS_MB or None
    print(f"\n{line}")
    if peak:
        print(f"进程内存峰值 {peak:,.0f} MB")
    print(f"统计：{stats_path}（总用时 {time.time() - t_all:.1f}s）")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
