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
│   ├── train.parquet     # 时间划分：训练集（2011-12 ~ 2023-12-31）
│   ├── test.parquet      # 时间划分：测试集（2024-01-01 ~ 至今）
│   ├── split_stats.json  # 由 src/split_temporal.py 生成的划分统计
│   ├── baseline_temporal_stats.json  # 由 src/baseline_temporal.py 生成的时间划分基线结果
│   └── cache/            # 全量聚合的 parquet 缓存与临时目录（可随时删除重建）
├── notebooks/            # 分析用 Jupyter Notebook
├── src/                  # 源代码
│   ├── sample_data.py    # 分块读取 + 随机抽样，生成 data/sample.csv
│   ├── aggregate_full.py # 全量聚合（DuckDB 单遍扫描 → data/full_stats.json）
│   ├── cleaning.py       # 清洗规则的唯一实现（三主类筛选 / 去重 / Product 归一化 / 空值填充）
│   ├── split_temporal.py # 时间划分（读 parquet 缓存 → data/train.parquet、test.parquet）
│   ├── baseline_temporal.py # 时间划分基线重跑 + Bootstrap 置信区间（读 train/test.parquet）
│   ├── eda.py           # 探索性数据分析（6 张图回答 6 个问题，输出到 figures/）
│   ├── report.py         # 终端表格对齐与数字格式化（供上面两个脚本共用）
│   └── prototype.py      # 三分类基线原型（清洗 / 划分 / 多数类基线 / 查表基线 / 评估）
├── docs/                 # 文档（数据字典、数据集卡片、决策日志、阶段二验收）
├── figures/              # EDA 图表（PNG，dpi=150）
├── tests/                # 测试
├── requirements.txt
└── README.md
```

> 清洗逻辑只写在 `src/cleaning.py` 一处，`prototype.py`（样本基线）与 `split_temporal.py`（全量时间划分）都从它导入，避免两份实现各自漂移。

## 文档索引

| 文件 | 内容 |
| --- | --- |
| `docs/data_dictionary.md` | 数据字典：原始 CSV 全部 16 字段的类型、缺失率、角色与注意事项 |
| `docs/dataset_card.md` | 数据集卡片：来源、许可、规模、覆盖总体、已知偏差、隐私与用途边界 |
| `docs/decision_log.md` | 决策日志：阶段一 #1–#8、阶段二 #11–#20 的关键决策与理由 |
| `docs/groupkfold_notes.md` | 阶段三预习笔记 —— GroupKFold（按公司分组交叉验证），作者为周恒毅、由赵伟翔代为提交（其 GitHub 账号处于封禁申诉期） |
| `docs/verification_zhou.md` | 数据文档校对记录，独立环境复现记录。作者为周恒毅、由赵伟翔代为提交（其 GitHub 账号处于封禁申诉期） |

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

# 5. 时间划分，输出 data/train.parquet、data/test.parquet、data/split_stats.json（约 80 秒）
python src/split_temporal.py

# 6. 运行三分类基线原型（需要已生成 data/sample.csv）
python src/prototype.py

# 7. 时间划分基线重跑 + Bootstrap 置信区间（需要已生成 train/test.parquet，约 100 秒）
python src/baseline_temporal.py

# 8. 探索性数据分析（需要已生成 full_stats.json + complaints_slim.parquet，约 3 秒）
python src/eda.py

# 9. 运行自动化测试
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

### 时间划分（`src/split_temporal.py`）

阶段一的随机 80/20 有**时间泄漏**：同一家公司的历史投诉与未来投诉被分到两侧，模型等于提前知道这家公司往后会怎么处理。这里改成按 `Date received` 切：训练集 ≡ **2023-12-31 及以前**，测试集 ≡ **2024-01-01 及以后**，测试集在时间上完全晚于训练集。

数据直接从 `data/cache/complaints_slim.parquet` 读（不再解析 8.4 GB CSV），清洗调用 `src/cleaning.py`（与 `prototype.py` 同一份实现），输出 `data/train.parquet`、`data/test.parquet`、`data/split_stats.json`。

| 集合 | 时间范围 | 行数 | 占比 |
| --- | --- | --- | --- |
| 训练集 | 2011-12-01 ~ 2023-12-31 | 4,456,111 | 28.5% |
| 测试集 | 2024-01-01 ~ 2026-07-03 | 11,160,958 | 71.5% |

三点值得写进报告：

1. **分布确实漂移了**：三主类占比 训练 72.74% / 23.96% / 3.29% → 测试 59.86% / 39.56% / 0.58%，即 explanation **−12.89 pp**、non-monetary **+15.60 pp**、monetary **−2.71 pp**。时间划分下这个漂移是真实存在的，不该当成 bug。
2. **公司维度几乎没泛化压力**：测试集 3,962 家公司里 81.93% 在训练集出现过，且测试集 **99.85% 的行**来自训练集已见过的公司（投诉量前 10 的公司 10/10 都见过）。也就是说，时间划分消除了时间泄漏，但没消除"公司记忆"——想测真正的新公司泛化，还需要再做一版按公司分组的划分。
3. **金钱救济率长期下滑**：13.21%（2012）→ 6.27%（2014）→ 2.92%（2020）→ 1.58%（2023）→ 0.87%（2024）→ 0.48%（2025/2026，2026 为部分年份）。同时 2022 年起 non-monetary 占比从 ~15% 跳到 40%+，与征信类投诉激增同步——这是三分类里 monetary 类只有 1% 量级的直接原因。完整逐年表在 `data/split_stats.json` 的 `yearly` 字段（终端也会打印同名表格，可直接用于 EDA 绘图）。

内存方面两侧**依次**处理（训练集算完统计、写完盘即释放），并先把 Arrow 的字符串列转字典编码再落 pandas：否则 1600 万行的 object 列要 7 GB，转成 category 后只需约 1 GB，实测峰值约 3.1 GB。

### 三分类任务

根据结构化特征 `Product, Issue, Company, State, Submitted via` 预测处理结果 `Company response to consumer`，只保留三个类别：

| 类别 | 样本量 | 占比 |
| --- | --- | --- |
| Closed with explanation | 122,049 | 63.61% |
| Closed with non-monetary relief | 67,202 | 35.02% |
| Closed with monetary relief | 2,622 | 1.37% |

`src/prototype.py` 会依次完成数据清洗（类别筛选、按 Complaint ID 去重、Product 旧命名合并、空值填充）、80/20 分层划分（随机种子 42），并实现两个基线——多数类基线与「Product × Issue 组合查表」基线——输出宏平均 F1、各类 precision/recall/F1 与混淆矩阵做对比。

### 时间划分基线重跑（`src/baseline_temporal.py`）

在 `data/train.parquet` / `data/test.parquet`（时间划分）上重跑阶段一的三个基线（多数类、Product × Issue 查表、Product × Issue × Company 查表），清洗规则复用 `src/cleaning.py`（parquet 已由 `split_temporal.py` 用同一份 `clean_data` 清洗后落盘）。输出格式与 `prototype.py` 完全一致（宏平均 F1、各类 precision/recall/F1、混淆矩阵），并新增：

- **Bootstrap 置信区间**：测试集有放回抽样 1000 次（种子 42），取 2.5%/97.5% 分位。实现用多项分布抽样代替逐行重抽样——重抽样 N 个 (y_true, y_pred) 对后混淆矩阵各格子计数服从 Multinomial(N, p)，与逐行 Bootstrap 分布等价，但 1000 次迭代 < 0.1 秒（逐行需约 180 秒）。
- **与阶段一对比**：时间划分 F1 与随机划分结果（0.2592 / 0.2679 / 0.5226）并排打印，标注下降幅度。

| 基线 | 时间划分 F1 | Bootstrap 95% CI | 随机划分 F1 | 变化 |
| --- | --- | --- | --- | --- |
| 多数类 | 0.2496 | [0.2496, 0.2497] | 0.2592 | −3.69% |
| Product × Issue 查表 | 0.3361 | [0.3352, 0.3369] | 0.2679 | +25.44% |
| Product × Issue × Company 查表 | 0.4797 | [0.4786, 0.4807] | 0.5226 | −8.21% |

内存峰值 1,390 MB（预算 4 GB），总用时约 98 秒。

### 探索性数据分析（`src/eda.py`）

基于 `data/full_stats.json`、`data/split_stats.json` 和 `data/cache/complaints_slim.parquet`，生成 6 张 PNG 图表（dpi=150，中文标题与坐标），每张图回答一个明确的业务问题，生成时附 2-3 句解读打印到终端。图表 5、6 使用双面板布局（上面板=投诉量条形，下面板=金钱救济率条形）避免双轴标签重叠。

| 图 | 文件名 | 问题 | 关键发现 |
| --- | --- | --- | --- |
| 图 3 | `fig3_yearly_volume_relief.png` | 数据的时间漂移有多严重？ | 投诉量 2012→2025 增长 109 倍；金钱救济率从 13.2% 降至 0.48% |
| 图 4 | `fig4_train_test_class.png` | 时间划分两侧的分布差异 | explanation −12.9pp、non-monetary +15.6pp、monetary −2.7pp |
| 图 5 | `fig5_top_companies.png` | 头部公司的处理策略差异有多大？ | 征信三巨头占 77% 投诉但救济率≈0%；银行类 10%+ |
| 图 6 | `fig6_product_relief.png` | 产品间差异 | 征信类占 80% 但救济率 0.05%；预付卡 23% |
| 图 7 | `fig7_missing_rates.png` | 哪些字段不可用？ | narrative 76.94%、Tags 95.28%；核心建模特征 <0.4% |
| 图 8 | `fig8_narrative_consent.png` | 文本字段未来是否可用？ | 2015 年才开放；峰值 47.5%（2017）；2025 年降至 22.5% |

图 6 与图 8 用 DuckDB 直接查询 parquet 缓存（`read_parquet` + `GROUP BY`），分别做产品×目标交叉统计和按年叙述同意率。总用时约 2.5 秒。

### 测试

`pytest` 共 54 个测试，分五个文件：

- `tests/test_prototype.py`（10 个）：类别筛选、去重、Product 归一化、空值填充、划分比例与分层、随机种子可复现性、两个基线（含未见组合回退）、无效输入报错，以及在 10 行手工数据上与手工计算逐条核对；
- `tests/test_aggregate_full.py`（6 个）：用 7 行迷你 CSV 跑通全量聚合流水线，核对行数口径、两种日期格式解析、分布计数、公司占比和为 1、缺失率与叙述非空、样本偏差计算；
- `tests/test_split_temporal.py`（13 个）：划分边界不交叉（含 2023-12-31T23:59 归训练集、2024-01-01T00:00 归测试集）、日期全部落在正确区间、只留三主类、旧命名已归一化、同 ID 去重、空 State 填 Unknown、`split_temporal`/`prototype` 共用同一份 `cleaning.clean_data`（防实现漂移）、公司重叠率与年度金钱救济率的计算口径；
- `tests/test_baseline_temporal.py`（12 个）：`macro_f1_from_cm` 与 sklearn `f1_score` 完全一致（含 zero_division=0）、Bootstrap 区间覆盖率（合成数据上 200 次抽样验证 ~95% 覆盖）、随机种子可复现、多项分布 Bootstrap 与逐行重抽样等价、高效查表与 `prototype.baseline_lookup` 预测逐条一致（含平局打破、未见组合回退）、标签编码往返与未知类别报错；
- `tests/test_eda.py`（13 个）：`truncate` 截断逻辑（短串不截、长串加 …、边界值与 Unicode）、`normalize_product_name` 与 `cleaning.PRODUCT_NORMALIZATION` 完全一致（大小写无关、未知名称原样返回）、归一化后聚合计数与金钱救济率计算正确（模拟 DuckDB 查询结果）、三主类顺序与 `cleaning.TARGET_CLASSES` 一致、`load_stats` 能正确加载 JSON。

测试数据全部在测试文件内手工构造，**不依赖 `data/` 下的真实数据**，因此无需下载原始数据即可运行。

运行结束后会在 `data/sample.csv` 生成约 20 万行的随机样本，并在终端打印原始总行数、抽样行数、`Company response to consumer` 分布、`Product` 前 10 类分布以及 `Date received` 的时间范围。

> 说明：脚本需要对原始文件读取两遍——第一遍统计总行数与有效行数，第二遍按 `目标行数 / 有效行数` 的比例分块抽样，因此结果与"一次性随机抽样"一致，且内存占用恒定（只与单个分块大小有关，与文件大小无关）。
