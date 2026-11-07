# CFPB 消费者投诉数据分析

基于美国 CFPB（Consumer Financial Protection Bureau）消费者投诉数据库，分析不同类型的金融产品投诉的分布特征、处理结果与响应时效。

## 数据说明

- **数据来源**：美国消费者金融保护局（CFPB）消费者投诉数据库（Consumer Complaint Database）。
- **原始文件**：`data/complaints.csv`，约 8.4 GB，包含约 18 个字段，其中 `Consumer complaint narrative` 为较长的自由文本。
- **未纳入 Git**：原始数据文件体积过大，已在 `.gitignore` 中排除 `data/` 目录，仓库中不包含原始数据。
- **重新生成样本**：克隆仓库后，把原始 `complaints.csv` 放到 `data/` 目录下，运行 `src/sample_data.py` 即可重新生成分析所用的样本文件 `data/sample.csv`（约 20 万行，固定随机种子 42，结果可复现）。
- **样本文件字段**：

  | 字段 | 说明 |
  | --- | --- |
  | Date received | 投诉收到的日期 |
  | Product | 投诉涉及的产品大类 |
  | Issue | 投诉的问题类型 |
  | Company | 被投诉公司 |
  | State | 消费者所在州 |
  | Submitted via | 投诉提交渠道 |
  | Company response to consumer | 公司对消费者的回应结果 |
  | Timely response? | 公司是否按时回应 |
  | Complaint ID | 投诉唯一编号 |

  为控制内存占用和读取时间，样本未包含 `Consumer complaint narrative`、`Sub-product`、`Sub-issue` 等字段。

- **数据质量提示**：`Date received` 一列同时存在两种格式——纯日期 `YYYY-MM-DD`（约占 93.6%）和带时间戳的 `YYYY-MM-DDTHH:MM:SS.mmmZ`（约占 6.4%）。直接用 `pd.to_datetime()` 会因 pandas 按首个值推断格式而把另一种格式判为无效值，因此脚本中显式使用了 `format="mixed"`。

## 目录结构

```
cfpb-complaint-analysis/
├── data/                 # 数据目录（已加入 .gitignore，不入库）
│   ├── complaints.csv    # 原始数据（约 8.4 GB，需自行下载）
│   ├── sample.csv        # 由 src/sample_data.py 生成的样本（约 20 万行）
│   ├── full_stats.json   # 由 src/aggregate_full.py 生成的全量聚合结果
│   └── cache/            # 全量聚合的 parquet 缓存与临时目录（可随时删除重建）
├── notebooks/            # 分析用 Jupyter Notebook
├── src/                  # 源代码
│   ├── sample_data.py    # 分块读取 + 随机抽样，生成 data/sample.csv
│   ├── aggregate_full.py # 全量聚合（DuckDB 单遍扫描 → data/full_stats.json）
│   └── prototype.py      # 三分类基线原型（清洗 / 划分 / 多数类基线 / 查表基线 / 评估）
├── tests/                # 测试
├── requirements.txt
└── README.md
```

## 运行方法

环境要求：Python 3.13（3.9 及以上均可），依赖见 `requirements.txt`。

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 确认原始数据已就位
#    data/complaints.csv（约 8.4 GB，需从 CFPB 官网下载）

# 3. 在项目根目录执行，生成样本并查看统计报告
python src/sample_data.py

# 4. 全量聚合，输出 data/full_stats.json（首次会建 parquet 缓存，约 75 秒；之后几秒）
python src/aggregate_full.py
#    python src/aggregate_full.py --refresh    # 原始 CSV 变了才需要重建缓存

# 5. 运行三分类基线原型（需要已生成 data/sample.csv）
python src/prototype.py

# 6. 运行自动化测试
pytest -v
```

### 全量聚合（`src/aggregate_full.py`）

对 8.4 GB 原始文件做一次全量扫描，输出 `data/full_stats.json`，包含：总行数 / 有效行数、目标变量 8 类分布、按年份投诉量、`Product` 全类分布、前 30 家公司及三主类占比、16 列缺失率、公开叙述（`Consumer complaint narrative` 非空）行数、`State` 前 15，以及**样本 vs 全量的占比偏差表**（绝对百分点），用于验收 20 万行样本的代表性。

技术方案用 **DuckDB**（`duckdb==2.0.0`），理由：

- 9 组不同维度的分组聚合，在 SQL 里各是一条 `GROUP BY`，比在 pandas 分块里维护多套累加器更好核对；
- `memory_limit` 硬上限 + `temp_directory` 溢写，内存占用可控（实测峰值约 2.0 GB）；
- **本文件的 `Consumer complaint narrative` 含换行，16,545,376 条记录在磁盘上摊成 27,656,645 个物理行**，DuckDB 的并行 CSV 读取器不支持这类文件（会直接报 `NotImplemented`），脚本因此显式 `parallel=false`，用单线程 RFC4180 解析器读取——实测 8.3 GB / 75 秒，之后所有聚合都跑在 0.15 GB 的 parquet 缓存上，仍为 20 线程并行（2.4 秒出全部结果）；
- parquet 缓存可直接用于后续全量建模，不必再扫原始 CSV。

主要结果：总行数 16,545,376，有效行 16,270,799（1.66% 无响应结果或无 Product）；三主类占有效行 60.98% / 33.70% / 1.30%；20 万行样本与全量的三主类占比最大偏差仅 **0.095 pp**，`Product` 前 10 最大偏差 **0.083 pp**。

### 三分类任务

根据结构化特征 `Product, Issue, Company, State, Submitted via` 预测处理结果 `Company response to consumer`，只保留三个类别：

| 类别 | 样本量 | 占比 |
| --- | --- | --- |
| Closed with explanation | 122,049 | 63.61% |
| Closed with non-monetary relief | 67,202 | 35.02% |
| Closed with monetary relief | 2,622 | 1.37% |

`src/prototype.py` 会依次完成数据清洗（类别筛选、按 Complaint ID 去重、Product 旧命名合并、空值填充）、80/20 分层划分（随机种子 42），并实现两个基线——多数类基线与「Product × Issue 组合查表」基线——输出宏平均 F1、各类 precision/recall/F1 与混淆矩阵做对比。

### 测试

`tests/test_prototype.py` 使用 pytest，共 10 个测试，覆盖类别筛选、去重、Product 归一化、空值填充、划分比例与分层、随机种子可复现性、两个基线的行为（含未见组合回退）、无效输入的报错，以及在 10 行手工数据上与手工计算的逐条核对。

测试数据全部在测试文件内构造，**不依赖 `data/sample.csv`**，因此无需先下载原始数据即可运行。

运行结束后会在 `data/sample.csv` 生成约 20 万行的随机样本，并在终端打印原始总行数、抽样行数、`Company response to consumer` 分布、`Product` 前 10 类分布以及 `Date received` 的时间范围。

> 说明：脚本需要对原始文件读取两遍——第一遍统计总行数与有效行数，第二遍按 `目标行数 / 有效行数` 的比例分块抽样，因此结果与"一次性随机抽样"一致，且内存占用恒定（只与单个分块大小有关，与文件大小无关）。
