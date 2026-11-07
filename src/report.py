#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""终端输出的小工具：中英混排表格对齐与数字格式化。

``aggregate_full.py`` 与 ``split_temporal.py`` 都用它，避免各自维护一份
"按显示宽度补空格" 的实现。全角字符按 2 列计算，这样在等宽字体终端里
中文与英文混排的表格才是齐的。
"""

from __future__ import annotations

import unicodedata


def disp_width(text: str) -> int:
    """字符串的终端显示宽度：中日韩全角字符占 2 列。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad_text(text: str, width: int, align: str = "l") -> str:
    """按显示宽度左右补空格。align: "l" 左对齐，"r" 右对齐。"""
    fill = " " * max(0, width - disp_width(text))
    return fill + text if align == "r" else text + fill


def print_table(headers: list[str], rows: list[list[str]], aligns: str | None = None) -> None:
    """打印定宽表格。aligns 是每列对齐方式串，例如 "lrrr"；默认首列左对齐、其余右对齐。"""
    if aligns is None:
        aligns = "l" + "r" * (len(headers) - 1)

    widths = []
    for i, head in enumerate(headers):
        width = disp_width(str(head))
        for row in rows:
            width = max(width, disp_width(str(row[i])))
        widths.append(width)

    print("  " + "  ".join(pad_text(str(h), widths[i], aligns[i]) for i, h in enumerate(headers)))
    print("  " + "-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print("  " + "  ".join(pad_text(str(c), widths[i], aligns[i]) for i, c in enumerate(row)))


def fmt_int(n: int | None) -> str:
    return "-" if n is None else f"{n:,}"


def fmt_pct(x: float | None, digits: int = 2) -> str:
    return "-" if x is None else f"{x * 100:.{digits}f}%"


def fmt_gb(n_bytes: int) -> str:
    return f"{n_bytes / 1024 ** 3:.2f} GB"
