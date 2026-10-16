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
│   └── sample.csv        # 由 src/sample_data.py 生成的样本（约 20 万行）
├── notebooks/            # 分析用 Jupyter Notebook
├── src/                  # 源代码
│   ├── sample_data.py    # 分块读取 + 随机抽样，生成 data/sample.csv
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

# 4. 运行三分类基线原型（需要已生成 data/sample.csv）
python src/prototype.py
```

### 三分类任务

根据结构化特征 `Product, Issue, Company, State, Submitted via` 预测处理结果 `Company response to consumer`，只保留三个类别：

| 类别 | 样本量 | 占比 |
| --- | --- | --- |
| Closed with explanation | 122,049 | 63.61% |
| Closed with non-monetary relief | 67,202 | 35.02% |
| Closed with monetary relief | 2,622 | 1.37% |

`src/prototype.py` 会依次完成数据清洗（类别筛选、按 Complaint ID 去重、Product 旧命名合并、空值填充）、80/20 分层划分（随机种子 42），并实现两个基线——多数类基线与「Product × Issue 组合查表」基线——输出宏平均 F1、各类 precision/recall/F1 与混淆矩阵做对比。

运行结束后会在 `data/sample.csv` 生成约 20 万行的随机样本，并在终端打印原始总行数、抽样行数、`Company response to consumer` 分布、`Product` 前 10 类分布以及 `Date received` 的时间范围。

> 说明：脚本需要对原始文件读取两遍——第一遍统计总行数与有效行数，第二遍按 `目标行数 / 有效行数` 的比例分块抽样，因此结果与"一次性随机抽样"一致，且内存占用恒定（只与单个分块大小有关，与文件大小无关）。
