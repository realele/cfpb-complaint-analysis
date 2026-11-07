## 任务 1：数据文档校对记录

### 发现的问题（4 处）

**问题 1（两处文档均有，内容性错误）——「三大征信局 non-monetary 占比 49%～59% 高度一致」不成立**

- 位置：`docs/data_dictionary.md` 第 8 节 Company 注意事项；`docs/dataset_card.md` 5.2 节
- 文档写的：「这三家的处理策略高度一致——non-monetary relief 占比 49%～59%」
- JSON 实际（`full_stats.json` → `top_companies` 前 3 名）：
  - TransUnion：non-monetary **58.81%**（explanation 36.87%）
  - Equifax：non-monetary **49.16%**（explanation 50.00%）
  - Experian：non-monetary **9.80%**（explanation **81.59%**）
- 结论：TransUnion、Equifax 落在 49%～59% 区间，但 **Experian 只有 9.80%**，其 81.6% 的投诉以 explanation 结案，三家策略并不一致。「monetary relief 几乎为 0」这句是对的（0.0013%～0.0289%）。建议改为「前两家高度一致，Experian 以 explanation 为主」或删去「三家一致」的表述。

**问题 2（口径错位）——「其余 5 类共 928,186 行被剔除」**

- 位置：`docs/data_dictionary.md` 第 14 节目标变量
- 文档写的：「其余 5 类（含 In progress 等）共 928,186 行被剔除」
- JSON 实际：5 个非主类合计 = 582,672 + 30,276 + 17,867 + 17,611 + 5,304 = **653,730 行**。928,186 是 `split_stats.json` → `totals.other_classes`，它的口径是「未进入三主类划分的全部行」，其中包含 274,456 行无目标值的无效行（274,577 无效行 − 121 行无 Product）。数字本身有出处，但把 928,186 安在「其余 5 类」头上是口径错位。建议改为「其余 5 类共 653,730 行，连同无目标值的行合计 928,186 行被剔除」。

**问题 3（事实错误）——缓存 parquet 的行数**

- 位置：`docs/dataset_card.md` 第 8 节「列式缓存：约 158 MB，全量 16,270,799 有效行」
- 实际：我用 DuckDB 实测 `data/cache/complaints_slim.parquet` 为 **16,545,376 行**（含无效行的全部原始行），与 `split_stats.json` → `totals.cache_rows` 一致。16,270,799 是有效行数，不是缓存行数；缓存是按原始行建的（158 MB 属实，无误）。

**问题 4（轻微，约数偏差）——Product 归一化影响行数「约 268 万」**

- 位置：`docs/data_dictionary.md` 第 2 节 Product 注意事项
- 文档写的：「影响行数合计约 268 万」
- JSON 实际：`split_stats.json` → `product_mapping` 7 项影响行数合计 = **2,629,913（约 263 万）**。偏差约 4 万行。按 `full_stats.json` 的 product_distribution 口径算也只有 263.8 万。建议改为「约 263 万」。

### 提示性发现（不算错误，供参考）

1. **舍入**：卡片 5.2 写 explanation 漂移 −12.88 pp，精确值是 72.7448 − 59.8588 = 12.886 → 四舍五入应为 12.89。文档是用两位截断值相减（72.74−59.86）得到的 12.88，属可接受的舍入路径差异。
2. **附带发现（任务范围外）**：`docs/decision_log.md` #15 写「P×I×C 查表 F1 = 0.4796」，而 JSON 是 0.479672，四舍五入应为 0.4797（两份正式文档都写对了 0.4797）。另 #12 写「实测 1000 次 < 0.1 秒」，JSON 里 `boot_seconds` 为 0.13。
3. **三个 JSON 中无出处、无法核对的表述**（非错误，仅注明依赖外部来源）：「1654 万记录摊成约 2766 万 CSV 物理行」；叙述公开率「峰值约 47.5%（2017）、2025 年约 22.5%」（出自 fig8，JSON 未存）；「CFPB 官方称约 98% 投诉获及时回应」。

### 核对无误的主要类别（104 项一致）

- 16 个字段的缺失数与缺失率（16×2 = 32 项）✓
- 目标变量 8 个原始类别的计数与占比（16 项）✓
- 总体规模：raw/valid/invalid 行数、有效行率 98.34%、公司 8,037、Product 23、目标 8 类、源文件 8.312 GB ✓
- 日期混合格式：92.04% / 7.96% 及行数、零不可解析 ✓
- 训练/测试集行数（4,456,111 / 11,160,958，28.5% / 71.5%）、日期边界、三主类计数与占比（12 项）✓
- 年份投诉量 10 个年份值、金钱救济率 6 个年度值（13.21%→0.48%）✓
- 征信类合并计数 13,138,944 = 80.75%；前 3 公司合计 77.14%；FL/TX/CA 占比 ✓
- 公司重叠：测试集 99.85% 行来自已见公司、前 10 公司 10/10 ✓
- 基线：288 个组合、命中率 99.9998%、三基线 F1 = 0.2496 / 0.3361 / 0.4797 及 CI ✓
- 阶段一随机划分 F1（0.2592 / 0.2679 / 0.5226，存于 baseline JSON 的 meta）✓
- 文档引用的决策编号（#3 #4 #6 #11 #15 #16 #17 #18 #19）在 decision_log.md 中均存在 ✓

---

## 任务 2：独立环境复现记录

**环境**：Windows 11（win32）；全新虚拟环境 `C:\Users\Lenovo\cfpb-venv`，Python 3.13.14；未复用任何既有包。

**逐步结果**：

1. `pip install -r requirements.txt` —— 5 个包全部按锁定版本装好（pandas 2.3.3 / numpy 2.3.5 / duckdb 2.0.0 / scikit-learn 1.7.2 / pytest 8.4.2）。
2. `python -m pytest tests/ -v` —— **首次运行失败（收集阶段 2 个错误）**：
   - `tests/test_split_temporal.py` → `ModuleNotFoundError: No module named 'pyarrow'`（`src/split_temporal.py` 导入 pyarrow；读写 parquet 也需要）
   - `tests/test_eda.py` → `ModuleNotFoundError: No module named 'matplotlib'`（`src/eda.py` 导入）
   - **原因：`requirements.txt` 漏了 pyarrow 和 matplotlib**。干净环境严格按 requirements 装是跑不起来的——这直接影响课程评分项「可复现性」，**建议把 `pyarrow`、`matplotlib`（锁定版本）补进 requirements.txt**。
3. 补装 pyarrow 26.0.1、matplotlib 3.11.2 后重跑 —— **54 个测试全部 PASSED，耗时 5.85 秒** ✓（与预期 54 个一致）
4. `python src/baseline_temporal.py` —— 成功，总用时 **62.2 秒**。我跑出的三个宏平均 F1：
   - 多数类：**0.2496**（JSON 0.249632）✓
   - Product × Issue 查表：**0.3361**（JSON 0.336056）✓
   - Product × Issue × Company 查表：**0.4797**（JSON 0.479672）✓
5. 种子固定核对：脚本重跑后覆盖了 `data/baseline_temporal_stats.json`，**新生成文件与原文件的 macro_f1、CI 上下界、boot 均值逐项完全相同**（仅时间戳变化），确定性复现成立 ✓

**结论**：代码与数据本身可复现、结果可逐位复现；唯一障碍是 requirements.txt 缺两个包，按现状在干净机器上必现 ImportError。

---