#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CFPB 投诉数据探索性数据分析（EDA）—— 6 张图回答 6 个问题。

每张图都对应一个明确的业务问题，图表标题即问题本身，并在生成后打印 2-3 句
解读供报告直接引用。

数据来源：
- ``data/full_stats.json``：全量聚合统计（``aggregate_full.py`` 产出）
- ``data/split_stats.json``：时间划分统计（``split_temporal.py`` 产出）
- ``data/cache/complaints_slim.parquet``：全量列式缓存（DuckDB 直接查询，不解析 CSV）

运行::

    python src/eda.py

产出（``figures/`` 下，PNG，dpi=150）::

    fig3_yearly_volume_relief.png   年度投诉量 + 金钱救济率双轴
    fig4_train_test_class.png       训练 vs 测试三主类占比
    fig5_top_companies.png          前 15 公司投诉量 + 金钱救济率
    fig6_product_relief.png         归一化后各 Product 投诉量 + 金钱救济率
    fig7_missing_rates.png          各列缺失率
    fig8_narrative_consent.png      叙述文本公开同意率随年份变化
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，直接存文件
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# cleaning.py 的 PRODUCT_NORMALIZATION 用于图 6 的产品名归一化，
# TARGET_CLASSES 保证三主类顺序与全局一致。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cleaning import PRODUCT_NORMALIZATION, TARGET_CLASSES  # noqa: E402

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
FULL_STATS = PROJECT_ROOT / "data" / "full_stats.json"
SPLIT_STATS = PROJECT_ROOT / "data" / "split_stats.json"
# DuckDB 的 read_parquet 接受正斜杠路径，Windows 下也兼容
CACHE_PARQUET = str(PROJECT_ROOT / "data" / "cache" / "complaints_slim.parquet").replace("\\", "/")

DPI = 150

# 配色方案：蓝=量/训练集，红=比率/重点，橙=测试集，灰=参考线
C_VOLUME = "#2E86AB"
C_VOLUME_LIGHT = "#A8C8D8"
C_RATE = "#C0392B"
C_TRAIN = "#2E86AB"
C_TEST = "#E67E22"
C_HIGHLIGHT = "#C0392B"
C_NORMAL = "#2E86AB"
C_REF = "#BDC3C7"
C_GREEN = "#27AE60"

# matplotlib 全局字体与样式设置（中文）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
plt.rcParams["figure.dpi"] = DPI
plt.rcParams["savefig.dpi"] = DPI
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# 三主类短名（图例 / 标签用）
SHORT = {
    "Closed with explanation": "仅解释",
    "Closed with non-monetary relief": "非金钱救济",
    "Closed with monetary relief": "金钱救济",
}


# --------------------------------------------------------------------------- #
# 辅助函数
# --------------------------------------------------------------------------- #

def load_stats() -> tuple[dict, dict]:
    """加载 full_stats.json 与 split_stats.json。"""
    with open(FULL_STATS, encoding="utf-8") as f:
        full = json.load(f)
    with open(SPLIT_STATS, encoding="utf-8") as f:
        split = json.load(f)
    return full, split


def query_parquet(sql: str) -> pd.DataFrame:
    """用 DuckDB 查询 parquet 缓存，返回 DataFrame。

    DuckDB 的 read_parquet 直接读列式缓存（158 MB），GROUP BY 聚合在 16M 行上
    只需 2-3 秒，内存峰值远低于 pandas 全量加载。
    """
    con = duckdb.connect()
    try:
        df = con.execute(sql).fetchdf()
    finally:
        con.close()
    return df


def truncate(s: str, n: int = 28) -> str:
    """截断长字符串，末尾加 …。"""
    s = str(s)
    return s[:n] + "…" if len(s) > n + 1 else s


def normalize_product_name(name: str) -> str:
    """按 cleaning.PRODUCT_NORMALIZATION 归一化产品名。

    与 ``cleaning.normalize_product`` 的映射逻辑一致，但这里只返回归一化后的
    名称（不含合并映射表），供 EDA 聚合使用。
    """
    key = str(name).strip().lower()
    if key in PRODUCT_NORMALIZATION:
        return PRODUCT_NORMALIZATION[key][0]
    return str(name).strip()


def save_fig(fig: plt.Figure, name: str) -> Path:
    """保存图片到 figures/ 目录。"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def print_interp(n: int, title: str, lines: list[str]) -> None:
    """打印每张图的 2-3 句解读。"""
    print(f"\n[图 {n}] {title}")
    for line in lines:
        print(f"  {line}")


# --------------------------------------------------------------------------- #
# 图 3：年度投诉量 + 金钱救济率双轴折线图（2011–2026）
# 问题：数据的时间漂移有多严重？
# --------------------------------------------------------------------------- #

def fig3_yearly_volume_relief(split: dict) -> None:
    yearly = split["yearly"]
    years = [y["year"] for y in yearly]
    volumes = [y["complaints"] for y in yearly]
    rates = [y["monetary_share"] * 100 for y in yearly]
    is_partial = [y["year"] == 2026 for y in yearly]

    fig, ax1 = plt.subplots(figsize=(13, 6.5))

    # 左轴：投诉量（柱状），2026 年用浅色标注部分年份
    bar_colors = [C_VOLUME_LIGHT if p else C_VOLUME for p in is_partial]
    bars = ax1.bar(years, volumes, color=bar_colors, edgecolor="white",
                   linewidth=0.5, zorder=2, width=0.7)
    ax1.set_xlabel("年份", fontsize=11)
    ax1.set_ylabel("投诉量（万条）", color=C_VOLUME, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=C_VOLUME, labelsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1e4:.0f}"))

    # 右轴：金钱救济率（折线）
    ax2 = ax1.twinx()
    ax2.plot(years, rates, color=C_RATE, marker="o", linewidth=2.2,
             markersize=7, zorder=3, markeredgecolor="white", markeredgewidth=0.8)
    ax2.set_ylabel("金钱救济率（%）", color=C_RATE, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=C_RATE, labelsize=9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.set_ylim(0, max(rates) * 1.25)

    # 标注 2026 为部分年份（文字放在图表上方，箭头指向柱顶）
    idx_2026 = years.index(2026)
    ax1.annotate(
        "2026 为部分年份\n（截至 7/3）",
        xy=(2026, volumes[idx_2026]),
        xytext=(2026, max(volumes) * 1.08),
        fontsize=8.5, color="#555", ha="center", va="bottom",
        arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
    )

    # 训练 / 测试划分边界（虚线 + 顶部标签）
    ax1.axvline(x=2023.5, color="#7F8C8D", linestyle=":", linewidth=1.2,
                alpha=0.7, zorder=1)
    ax1.text(2023.5, max(volumes) * 1.08, "训练│测试\n边界",
             fontsize=8, color="#7F8C8D", ha="center", va="bottom")

    ax1.set_xticks(years)
    ax1.set_xticklabels(years, fontsize=9)
    ax1.set_ylim(0, max(volumes) * 1.15)

    ax1.set_title(
        "图 3  年度投诉量与金钱救济率（2011–2026）\n"
        "问题：数据的时间漂移有多严重？",
        fontsize=13, fontweight="bold", pad=14,
    )

    # 右轴网格线关掉，避免与左轴交叉
    ax2.grid(False)
    fig.tight_layout()
    save_fig(fig, "fig3_yearly_volume_relief.png")

    # 解读
    yd = {y["year"]: y for y in yearly}
    v_2012, v_2025 = yd[2012]["complaints"], yd[2025]["complaints"]
    r_2012, r_2025 = yd[2012]["monetary_share"] * 100, yd[2025]["monetary_share"] * 100
    print_interp(3, "年度投诉量与金钱救济率", [
        f"投诉量从 2012 年 {v_2012:,} 条激增至 2025 年 {v_2025:,} 条（约 {v_2025 / v_2012:.0f} 倍），"
        f"2024 年起量级远超历史——时间漂移在规模层面极其显著。",
        f"金钱救济率从 2012 年 {r_2012:.1f}% 持续下滑至 2025 年 {r_2025:.2f}%，降幅超 90%，"
        f"说明近年来金融机构越来越少以金钱补偿方式结案。",
        "2026 年数据截至 7 月 3 日（浅色柱），绝对量不宜与完整年份直接比较；"
        "训练/测试边界（2023↔2024）两侧的量级与救济率差异均很大。",
    ])


# --------------------------------------------------------------------------- #
# 图 4：训练集 vs 测试集三主类占比对比条形图
# 问题：时间划分两侧的分布差异
# --------------------------------------------------------------------------- #

def fig4_train_test_class(split: dict) -> None:
    train_c = split["train"]["classes"]
    test_c = split["test"]["classes"]

    classes = TARGET_CLASSES
    labels = [SHORT[c] for c in classes]
    train_shares = [train_c[c]["share"] * 100 for c in classes]
    test_shares = [test_c[c]["share"] * 100 for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, train_shares, width, label="训练集（2011–2023）",
                   color=C_TRAIN, edgecolor="white", linewidth=0.5, zorder=2)
    bars2 = ax.bar(x + width / 2, test_shares, width, label="测试集（2024–2026）",
                   color=C_TEST, edgecolor="white", linewidth=0.5, zorder=2)

    # 数值标注
    for bar, val in zip(bars1, train_shares):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, color=C_TRAIN)
    for bar, val in zip(bars2, test_shares):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, color=C_TEST)

    # 标注变化箭头
    for i, (t, e) in enumerate(zip(train_shares, test_shares)):
        diff = e - t
        sign = "+" if diff > 0 else ""
        color = C_RATE if abs(diff) > 5 else "#555"
        ax.annotate(
            f"{sign}{diff:.1f}pp",
            xy=(i, max(t, e) + 6), ha="center", fontsize=9, color=color,
            fontweight="bold" if abs(diff) > 5 else "normal",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("占比（%）", fontsize=11)
    ax.set_ylim(0, max(max(train_shares), max(test_shares)) * 1.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    ax.set_title(
        "图 4  训练集 vs 测试集三主类占比对比\n"
        "问题：时间划分两侧的分布差异",
        fontsize=13, fontweight="bold", pad=14,
    )

    fig.tight_layout()
    save_fig(fig, "fig4_train_test_class.png")

    # 解读
    exp_d = test_c["Closed with explanation"]["share"] - train_c["Closed with explanation"]["share"]
    non_d = test_c["Closed with non-monetary relief"]["share"] - train_c["Closed with non-monetary relief"]["share"]
    mon_d = test_c["Closed with monetary relief"]["share"] - train_c["Closed with monetary relief"]["share"]
    print_interp(4, "训练 vs 测试三主类占比", [
        f"仅解释类从训练集 {train_shares[0]:.1f}% 降到测试集 {test_shares[0]:.1f}%（{exp_d * 100:+.1f}pp），"
        f"非金钱救济从 {train_shares[1]:.1f}% 升至 {test_shares[1]:.1f}%（{non_d * 100:+.1f}pp）——"
        f"两个大类发生了显著漂移。",
        f"金钱救济从 {train_shares[2]:.1f}% 降到 {test_shares[2]:.1f}%（{mon_d * 100:+.1f}pp），"
        f"在测试集中仅剩 0.6%，几乎退化为罕见类。",
        "这意味着训练阶段学到的「金钱救济」模式在测试集上几乎无法命中，"
        "是 P×I×C 查表基线 F1 下降 8.2% 的直接原因。",
    ])


# --------------------------------------------------------------------------- #
# 图 5：投诉量前 15 公司的投诉量（条形）+ 各自金钱救济率（散点次坐标）
# 问题：头部公司的处理策略差异有多大？
# --------------------------------------------------------------------------- #

def fig5_top_companies(full: dict) -> None:
    top15 = full["top_companies"][:15]
    # 从下往上画（最大公司在顶部）
    companies = [truncate(c["company"], 30) for c in reversed(top15)]
    volumes = [c["complaints"] for c in reversed(top15)]
    rates = [c["closed_with_monetary_relief"]["share_of_company"] * 100
             for c in reversed(top15)]
    y_pos = np.arange(len(top15))

    # 双面板布局：上面板=投诉量，下面板=金钱救济率。共享 y 轴（公司名），
    # 避免双轴散点与柱标签重叠的问题。
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 10), sharey=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
    )

    # --- 上面板：投诉量（水平柱，对数刻度） ---
    ax_top.barh(y_pos, volumes, color=C_VOLUME, edgecolor="white",
                linewidth=0.5, height=0.6, zorder=2)
    ax_top.set_yticks(y_pos)
    ax_top.set_yticklabels(companies, fontsize=9)
    ax_top.set_xscale("log")
    ax_top.set_xlabel("投诉量（对数刻度）", color=C_VOLUME, fontsize=11)
    ax_top.tick_params(axis="x", labelcolor=C_VOLUME, labelsize=9)
    ax_top.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{x / 1e4:.0f}万" if x >= 1e4 else f"{x:.0f}"))
    for i, v in enumerate(volumes):
        ax_top.text(v * 1.15, i, f"{v:,}", va="center", fontsize=7.5, color=C_VOLUME)
    ax_top.grid(axis="x", alpha=0.2)

    # --- 下面板：金钱救济率（水平柱，线性 %） ---
    ax_bot.barh(y_pos, rates, color=C_RATE, edgecolor="white",
                linewidth=0.5, height=0.6, zorder=2)
    ax_bot.set_xlabel("金钱救济率（%）", color=C_RATE, fontsize=11)
    ax_bot.tick_params(axis="x", labelcolor=C_RATE, labelsize=9)
    ax_bot.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    max_rate = max(rates) if rates else 1
    ax_bot.set_xlim(0, max_rate * 1.25)
    for i, r in enumerate(rates):
        ax_bot.text(r + max_rate * 0.02, i, f"{r:.2f}%", va="center",
                    fontsize=8, color=C_RATE)
    # 隐藏下面板的 y 轴标签（与上面板共享）
    ax_bot.tick_params(axis="y", length=0)
    ax_bot.grid(axis="x", alpha=0.2)

    fig.suptitle(
        "图 5  投诉量前 15 公司：投诉量 vs 金钱救济率\n"
        "问题：头部公司的处理策略差异有多大？",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(top=0.88, bottom=0.08)
    save_fig(fig, "fig5_top_companies.png")

    # 解读
    top3 = top15[:3]
    top3_vol = sum(c["complaints"] for c in top3)
    total_valid = full["totals"]["valid_rows"]
    banks = [c for c in top15 if c["closed_with_monetary_relief"]["share_of_company"] > 0.1]
    print_interp(5, "前 15 公司投诉量与金钱救济率", [
        f"前三名（TransUnion、Equifax、Experian）合计 {top3_vol:,} 条投诉，"
        f"占全量 {top3_vol / total_valid * 100:.1f}%——征信三巨头垄断了投诉来源。",
        f"这三家的金钱救济率几乎为零（{top3[0]['closed_with_monetary_relief']['share_of_company'] * 100:.4f}% / "
        f"{top3[1]['closed_with_monetary_relief']['share_of_company'] * 100:.3f}% / "
        f"{top3[2]['closed_with_monetary_relief']['share_of_company'] * 100:.2f}%），"
        f"而银行类公司（如 BoA {next(c['closed_with_monetary_relief']['share_of_company'] * 100 for c in top15 if 'BANK OF AMERICA' in c['company']):.1f}%、"
        f"Citibank {next(c['closed_with_monetary_relief']['share_of_company'] * 100 for c in top15 if 'CITIBANK' in c['company']):.1f}%）"
        f"则有 10%+ 的金钱救济率。",
        "头部公司的处理策略截然不同：征信机构倾向于「仅解释」或「非金钱救济」，"
        "银行则有更高比例的金钱补偿——这解释了为何按公司查表（P×I×C）能大幅提升 F1。",
    ])


# --------------------------------------------------------------------------- #
# 图 6：归一化后各 Product 的投诉量与金钱救济率
# 问题：产品间差异
# --------------------------------------------------------------------------- #

def fig6_product_relief() -> None:
    # 用 DuckDB 查询 parquet 缓存：product × target 计数（仅三主类）
    sql = f"""
        SELECT product, target, COUNT(*) AS n
        FROM read_parquet('{CACHE_PARQUET}')
        WHERE target IN ('Closed with explanation',
                         'Closed with non-monetary relief',
                         'Closed with monetary relief')
          AND product IS NOT NULL
          AND trim(product) <> ''
        GROUP BY product, target
    """
    df = query_parquet(sql)

    # 归一化产品名（与 cleaning.py 的 PRODUCT_NORMALIZATION 一致）
    df["product_norm"] = df["product"].apply(normalize_product_name)

    # 按归一化后的产品名 + target 聚合
    agg = df.groupby(["product_norm", "target"], observed=True)["n"].sum().reset_index()
    pivot = agg.pivot(index="product_norm", columns="target", values="n").fillna(0)

    # 确保三列都在
    for c in TARGET_CLASSES:
        if c not in pivot.columns:
            pivot[c] = 0

    pivot["total"] = pivot[TARGET_CLASSES].sum(axis=1)
    pivot["monetary_rate"] = pivot["Closed with monetary relief"] / pivot["total"] * 100
    pivot = pivot.sort_values("total", ascending=True)  # 升序 → 最大在顶部

    products = [truncate(p, 35) for p in pivot.index]
    volumes = [int(v) for v in pivot["total"].tolist()]
    rates = pivot["monetary_rate"].tolist()
    y_pos = np.arange(len(products))

    # 双面板布局：上面板=投诉量，下面板=金钱救济率。
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 10), sharey=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
    )

    # --- 上面板：投诉量（水平柱，对数） ---
    ax_top.barh(y_pos, volumes, color=C_VOLUME, edgecolor="white",
                linewidth=0.5, height=0.6, zorder=2)
    ax_top.set_yticks(y_pos)
    ax_top.set_yticklabels(products, fontsize=9)
    ax_top.set_xscale("log")
    ax_top.set_xlabel("投诉量（对数刻度）", color=C_VOLUME, fontsize=11)
    ax_top.tick_params(axis="x", labelcolor=C_VOLUME, labelsize=9)
    ax_top.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{x / 1e4:.0f}万" if x >= 1e4 else f"{x:.0f}"))
    for i, v in enumerate(volumes):
        ax_top.text(v * 1.15, i, f"{v:,}", va="center", fontsize=7.5, color=C_VOLUME)
    ax_top.grid(axis="x", alpha=0.2)

    # --- 下面板：金钱救济率（水平柱，线性 %） ---
    ax_bot.barh(y_pos, rates, color=C_RATE, edgecolor="white",
                linewidth=0.5, height=0.6, zorder=2)
    ax_bot.set_xlabel("金钱救济率（%）", color=C_RATE, fontsize=11)
    ax_bot.tick_params(axis="x", labelcolor=C_RATE, labelsize=9)
    ax_bot.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    max_rate = max(rates) if rates else 1
    ax_bot.set_xlim(0, max_rate * 1.3)
    for i, r in enumerate(rates):
        ax_bot.text(r + max_rate * 0.02, i, f"{r:.2f}%", va="center",
                    fontsize=8, color=C_RATE)
    ax_bot.tick_params(axis="y", length=0)
    ax_bot.grid(axis="x", alpha=0.2)

    fig.suptitle(
        "图 6  归一化后各 Product 的投诉量与金钱救济率\n"
        "问题：产品间差异",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(top=0.88, bottom=0.08)
    save_fig(fig, "fig6_product_relief.png")

    # 解读
    cr_total = int(pivot.loc["Credit reporting or other personal consumer reports", "total"]) \
        if "Credit reporting or other personal consumer reports" in pivot.index else 0
    cr_rate = pivot.loc["Credit reporting or other personal consumer reports", "monetary_rate"] \
        if "Credit reporting or other personal consumer reports" in pivot.index else 0
    grand_total = pivot["total"].sum()
    # 找最高金钱救济率的产品
    max_rate_product = pivot["monetary_rate"].idxmax()
    max_rate_val = pivot["monetary_rate"].max()
    print_interp(6, "各 Product 投诉量与金钱救济率", [
        f"归一化后征信类产品（Credit reporting…）合计 {cr_total:,} 条，"
        f"占三主类的 {cr_total / grand_total * 100:.1f}%，但其金钱救济率仅 {cr_rate:.2f}%——"
        f"量级最大但几乎无金钱补偿。",
        f"金钱救济率最高的产品是「{truncate(max_rate_product, 25)}」（{max_rate_val:.1f}%），"
        f"与征信类形成鲜明对比——产品维度决定了救济策略的根本差异。",
        "这意味着 Product 是三分类任务中最强的高基数特征之一，"
        "也是 P×I 查表基线在时间划分下反升 25% 的根源。",
    ])


# --------------------------------------------------------------------------- #
# 图 7：各列缺失率横向条形图
# 问题：哪些字段不可用（重点标出 narrative 76.94%）
# --------------------------------------------------------------------------- #

def fig7_missing_rates(full: dict) -> None:
    missing = full["missing_rates"]
    # 按缺失率升序排（最小的在底部，最大的在顶部）
    missing_sorted = sorted(missing, key=lambda x: x["missing_rate"])
    columns = [m["column"] for m in missing_sorted]
    rates = [m["missing_rate"] * 100 for m in missing_sorted]

    # 高亮：narrative 和 tags（>50%）用红色，其余蓝色
    colors = [C_HIGHLIGHT if r > 50 else C_NORMAL for r in rates]

    y_pos = np.arange(len(columns))
    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(y_pos, rates, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.65, zorder=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(columns, fontsize=9)
    ax.set_xlabel("缺失率（%）", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_xlim(0, max(rates) * 1.18)

    # 数值标注
    for i, (r, m) in enumerate(zip(rates, missing_sorted)):
        count = m["missing"]
        label = f"{r:.2f}%" if r < 1 else f"{r:.1f}%"
        if count > 0:
            label += f"（{count:,}）"
        ax.text(r + max(rates) * 0.01, i, label, va="center", fontsize=8,
                color=C_HIGHLIGHT if r > 50 else "#333")

    # 重点标注 narrative
    for i, col in enumerate(columns):
        if col == "Consumer complaint narrative":
            ax.annotate(
                "叙述文本字段\n仅 23% 可用",
                xy=(rates[i], i), xytext=(rates[i] + 15, i + 1.5),
                fontsize=8.5, color=C_HIGHLIGHT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_HIGHLIGHT, lw=1),
            )

    ax.set_title(
        "图 7  各列缺失率\n"
        "问题：哪些字段不可用？",
        fontsize=13, fontweight="bold", pad=14,
    )

    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save_fig(fig, "fig7_missing_rates.png")

    # 解读
    narrative_rate = next(m["missing_rate"] for m in missing if m["column"] == "Consumer complaint narrative")
    tags_rate = next(m["missing_rate"] for m in missing if m["column"] == "Tags")
    core_cols = ["Product", "Issue", "Company", "State", "Submitted via"]
    core_max = max(m["missing_rate"] for m in missing if m["column"] in core_cols)
    print_interp(7, "各列缺失率", [
        f"叙述文本（Consumer complaint narrative）缺失率 {narrative_rate * 100:.2f}%，"
        f"仅约 23% 的投诉有叙述内容——需消费者同意才公开，无法作为常规特征使用。",
        f"Tags 字段缺失率更高（{tags_rate * 100:.1f}%），几乎完全不可用；"
        f"而核心建模特征（Product / Issue / Company / State / Submitted via）"
        f"缺失率均低于 {core_max * 100:.2f}%，数据质量可靠。",
        "结构化特征完整度高，文本特征不可用——这进一步验证了阶段一选择"
        "「结构化特征预测救济结果」而非「叙述文本分类」的判断。",
    ])


# --------------------------------------------------------------------------- #
# 图 8：叙述文本公开同意率随年份的变化
# 问题：文本字段未来是否可用
# --------------------------------------------------------------------------- #

def fig8_narrative_consent() -> None:
    sql = f"""
        SELECT EXTRACT(YEAR FROM try_cast(date_received AS TIMESTAMP)) AS yr,
               AVG(CAST(has_narrative AS INTEGER)) AS consent_rate,
               COUNT(*) AS n
        FROM read_parquet('{CACHE_PARQUET}')
        WHERE try_cast(date_received AS TIMESTAMP) IS NOT NULL
        GROUP BY yr
        ORDER BY yr
    """
    df = query_parquet(sql)
    years = df["yr"].astype(int).tolist()
    rates = (df["consent_rate"] * 100).tolist()
    counts = df["n"].tolist()
    overall_rate = (df["consent_rate"] * df["n"]).sum() / df["n"].sum() * 100

    fig, ax = plt.subplots(figsize=(13, 6.5))

    # 2011-2014 同意率为 0（CFPB 尚未开放叙述公开），用虚线连接
    ax.plot(years, rates, color=C_VOLUME, marker="o", linewidth=2.2,
            markersize=7, markeredgecolor="white", markeredgewidth=0.8, zorder=3)

    # 2026 用空心标记表示部分年份
    idx_2026 = years.index(2026)
    ax.plot(years[idx_2026], rates[idx_2026], marker="o", markersize=10,
            markerfacecolor="white", markeredgecolor=C_RATE, markeredgewidth=2, zorder=4)

    # 总体平均参考线
    ax.axhline(y=overall_rate, color=C_REF, linestyle="--", linewidth=1.2, zorder=1)
    ax.text(years[-1], overall_rate + 0.8, f"总体平均 {overall_rate:.1f}%",
            fontsize=8.5, color="#888", ha="right")

    # 标注峰值
    peak_idx = rates.index(max(rates))
    ax.annotate(
        f"峰值 {rates[peak_idx]:.1f}%\n（{years[peak_idx]} 年）",
        xy=(years[peak_idx], rates[peak_idx]),
        xytext=(years[peak_idx] - 2, rates[peak_idx] + 6),
        fontsize=8.5, color=C_RATE, ha="center",
        arrowprops=dict(arrowstyle="->", color=C_RATE, lw=0.8),
    )

    # 标注 2015 年突变
    idx_2015 = years.index(2015)
    ax.annotate(
        "2015 年起\nCFPB 开放叙述公开",
        xy=(years[idx_2015], rates[idx_2015]),
        xytext=(years[idx_2015] - 2.5, rates[idx_2015] - 12),
        fontsize=8.5, color="#555", ha="center",
        arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
    )

    # 标注 2026 部分年份
    ax.annotate(
        "2026 部分年份\n（截至 7/3，\n近期投诉尚未获得同意）",
        xy=(years[idx_2026], rates[idx_2026]),
        xytext=(years[idx_2026] - 2.5, rates[idx_2026] + 10),
        fontsize=8, color=C_RATE, ha="center",
        arrowprops=dict(arrowstyle="->", color=C_RATE, lw=0.8),
    )

    ax.set_xlabel("年份", fontsize=11)
    ax.set_ylabel("叙述公开同意率（%）", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=9)
    ax.set_ylim(-3, max(rates) * 1.25)

    ax.set_title(
        "图 8  叙述文本公开同意率随年份变化\n"
        "问题：文本字段未来是否可用？",
        fontsize=13, fontweight="bold", pad=14,
    )

    fig.tight_layout()
    save_fig(fig, "fig8_narrative_consent.png")

    # 解读
    rate_2015 = rates[idx_2015]
    rate_peak = rates[peak_idx]
    rate_2025 = rates[years.index(2025)]
    print_interp(8, "叙述文本公开同意率随年份变化", [
        f"2011–2014 年同意率为 0%（CFPB 尚未开放叙述公开），2015 年突然跳至 {rate_2015:.1f}%，"
        f"2016–2018 年达到峰值约 {rate_peak:.1f}%。",
        f"此后逐年下降，2025 年降至 {rate_2025:.1f}%——投诉量激增但同意率下降，"
        f"说明近期消费者更不愿公开叙述内容，或数据存在处理延迟。",
        "总体平均仅 23%，文本字段作为特征的未来可用性受限；"
        "即使使用叙述文本，也需按年份分层采样以避免偏差。",
    ])


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("CFPB EDA —— 6 张图回答 6 个问题")
    print("=" * 60)

    full, split = load_stats()
    print(f"已加载 full_stats.json + split_stats.json")

    fig3_yearly_volume_relief(split)
    fig4_train_test_class(split)
    fig5_top_companies(full)
    fig6_product_relief()
    fig7_missing_rates(full)
    fig8_narrative_consent()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"EDA 完成，6 张图已保存到 figures/（用时 {elapsed:.1f} 秒）")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
