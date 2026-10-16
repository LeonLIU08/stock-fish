# 港股 Qlib 训练与测试

- 状态: 未开始
- 立案: 2026-10-16
- 运行环境: 现有镜像 `zhuhai123/qlib-rdagent:v1`，Python 3.10，`linux/amd64`。在 Mac 上通过 Docker 跑，不在本机安装 qlib
- 一期: Alpha158，固定时间切分，多套 qlib 自带表格模型
- 二期: 把现有趋势分析的输出接成因子。因子定义不在本文

一期回答一件事：在同一份港股日频数据、同一套切分上，qlib 自带模型用 Alpha158 的排序能力如何。二期再回答趋势因子能否在这条基线上增加信息。

具体参数只写在 `qlib-zh/hk/baseline.yaml`。训练程序读这一份文件，不在代码或本文里再写一套日期、标签、模型参数、费用和报告指标。改某一段不会牵动其他段：

| 段 | 管什么 |
|---|---|
| `data` | 下载区间、复权、落盘目录、数据检查 |
| `universe` | 时点入池规则 |
| `segments` | 预热、训练、验证、测试的日历边界 |
| `label` | 标签表达式，以及向前看几个交易日 |
| `features` | Alpha158 与二期附加列 |
| `models` | 每个算法的类、模块和参数 |
| `backtest` | 持仓规则、成交价、费用、基准 |
| `report` | 输出哪些指标、用哪一段选模型 |

`segments` 只写日历上的起止日。标签会看未来收盘价，跨段由 `label.horizon_trading_days` 处理：训练程序把写了结束日的 `train`、`valid`、`test` 都往前空出这么多个交易日。改标签的展望长度时只改 `label`，不改 `segments`。

`report.also_record` 里的年化收益只落盘，不参与 `report.selection`。测试段在第一次完整跑通之后按 `report.lock_test_after_first_run` 锁定。之后改特征、股票池或费用，都要重新生成报告，输出目录里带上当时的配置副本。

## 1. 时间切分

行情起点在 `data.download_start`。Alpha158 的滚动窗口最长约 60 个交易日，所以 `segments.warmup` 只算特征，不产生训练样本。

训练、验证、测试的边界在 `segments`。现行标签是 T+1 收盘买、T+2 收盘卖。若训练段一直放到 `train.end` 当天，最后两天的标签会用到下一段的收盘价，验证段同理。空出的天数等于 `label.horizon_trading_days`，不写进 `segments`。

## 2. 和现有 A 股链路分开

现有训练在 `qlib-zh/scripts/practice/run_stage2_walk_forward.py`，YAML 在 `qlib-zh/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml`。那套是沪深 300、按年滚动、大师策略回测。港股基线不改那些文件，也不把港股数据写进 `~/.qlib/qlib_data/cn_data`。

港股数据目录单独放：

```text
~/.qlib/qlib_data/hk_data_20261016/
  calendars/day.txt
  instruments/all.txt
  features/<instrument>/*.day.bin
```

拉取和预处理用 `bash scripts/fetch_hk_qlib_data.sh`。中间文件在 `~/.qlib/hk_staging/hk_data_20261016/`，中断后重跑会跳过已完成的股票。目录名来自 `data.freeze_dirname`。

容器内路径写在 `runtime.provider_uri`。`qlib.init` 不使用 `region: cn`。日频实验只依赖这份数据目录、港股日历，以及 `backtest` 段。港股没有涨跌停，`backtest.limit_threshold` 为空，不使用 A 股配置里的 `0.095`。

代码放在 `qlib-zh/hk/`，由 `train_runner.py` 那种 `docker run` 方式启动。一期不接 Flask，也不接飞书。

## 3. 数据

Alpha158 需要开、高、低、收、成交量、VWAP，字段列表在 `data.fields`。

落盘规则读 `data` 段：

- 价格调整方式用 `data.price_adjustment`。复权因子写入 `factor`。
- 成交额来自数据源。VWAP 用成交额除以成交量。数据源没有成交额时，用收盘价乘成交量代替，并在该股票的质量记录里标出来。
- 交易日历用 `data.calendar_instrument` 的交易日。个股多出来的日期丢掉，停牌日留在日历上，成交量为 0。
- 代码宽度用 `data.code_width`，与仓库里的港股写法一致。
- 下载完成后目录名用 `data.freeze_dirname`。测试期间不重新下载去改已经落盘的历史。

前复权会在新的公司行动之后重算。一期接受这一点，靠冻结目录来避免同一次实验里历史价格变化。退市股票若数据源已经不提供，一期会缺。这是存活偏差，报告里要写明，不把结果说成全市场无偏。

取数顺序在 `data.sources`，对应 `analysis/backtest/bars.py` 的港股来源。dump 脚本把这些日线写成 qlib bin，不直接调用趋势回测。

写入 bin 之前做检查，不通过则不训练：

- 日历覆盖 `data.download_start` 到数据末日，且与日历标的的交易日一致
- 每年可交易股票数量写入报告。某一年少于 `data.checks.min_names_per_year`，训练中止
- 收盘价缺失率、高低于收盘价之外的脏行、成交量为负的行
- 按 `label.horizon_trading_days` 空出边界后，标签不再跨段

## 4. 股票池

不用 `scripts/hkconnect_06Oct.txt`。那是 2026-10 的名单，放回 2015–2023 会把后来才入池、以及已经退市的股票弄错。

一期用行情自身能算的时点规则，数值在 `universe` 段。每个交易日 T 入池的条件：

- 到 T 为止的历史天数不少于 `universe.min_history_days`
- T 之前 `universe.median_amount_days` 个交易日的中位成交额不低于 `universe.min_median_amount`
- `universe.require_positive_volume` 为真时，T 当日成交量大于 0

`universe.exclude` 里的代码只作基准，不进入股票池，也不参与训练标签。

验证集上可以改入池门槛或 `backtest` 里的持仓只数，每次改动都要记下当年的入池数量。测试集上不再改。

以后如果有港股通或恒生指数的历史成分，可以加一个更严的池，作为同一套模型的对照实验，不替换上面这条默许池。

## 5. 模型

模型只通过 `models` 段里的 `class`、`module_path` 和 `kwargs` 切换。数据集、切分、标签、回测仍是同一份文件里的其他段，训练循环不因为多一个模型而改写。新增模型时在 `models` 下加一块。自定义模型实现 `qlib.model.base.Model` 的 `fit` 和 `predict`，输出与 `LGBModel` 相同的预测表（索引为日期和股票，一列分数）。

一期跑 `enabled: true` 且 `dataset_kind: tabular` 的模型。它们共用 `features` 里的 `DatasetH`。LightGBM 先跑通整条链路，其余的用同一份 `data`、`segments`、`label` 重跑。

镜像里还有 LSTM、GRU、Transformer、TCN、HIST 等。它们需要时间序列采样或概念矩阵。要加时把 `dataset_kind` 写成 `sequence` 或 `graph`，一期训练循环遇到非 `tabular` 就跳过，不和 Alpha158 基线混在一张表里。

`kwargs` 起始值来自镜像内官方 Alpha158 示例。验证集上只允许改树模型的学习率和树的数量，以及 `backtest.topk`、`backtest.n_drop`。线性模型不调参。

## 6. 回测

组合类、持仓只数、账户、成交价、调仓频率和是否按手数取整，都在 `backtest` 段。现行成交价与 `label` 一致：信号用 T 日及之前的数据，标签和回测都按之后的收盘价计算。趋势回测是下一根开盘成交、并受每手股数约束。两套数字不能直接比收益。`backtest.respect_lot_size` 为假时，组合回测不按手数取整。

费用分项在 `backtest.costs`，起始值和 `analysis/backtest/trend-strategy-v0.yaml` 的港股项相同。`stamp_both_sides` 为真时，买卖两边都把印花税加进单边成本。训练程序把佣金、印花税、交易征费、交收费和滑点相加，得到 qlib 的 `open_cost` 和 `close_cost`，不在配置里另写一个合计。最低佣金用 `min_commission`。

基准代码在 `backtest.benchmark`。报告按 `report.metrics` 给出超额收益和绝对收益。

## 7. 报告与选模

每个模型一个目录。写哪些段、哪些指标、选模顺序和测试集是否只公布选中模型，都在 `report` 段。`report.copy_config_into_output` 为真时，目录里放当时的整份配置和数据目录名。

选模只看 `report.selection.segment`。排序键是 `rank_by`，靠前的键优先；`prefer_lower` 里的指标越小越好。`publish_test_for: selected_only` 时，测试集正文只写选中的模型，其余模型的测试集数字进附录，不参与挑选。

一期没有最低收益门槛。流水线跑通、启用的表格模型都有报告、测试段没有被用来调参，即算完成。

## 8. 二期接入点

二期不改 `segments`、`universe`、`label` 和 `backtest`。只往 `features.extra` 加列。

趋势分析继续由 `analysis/backtest/trend/` 计算。它在日期 T 的输出只能使用 T 及之前的 K 线。结果落成一张表，索引与 Alpha158 相同：日期、五位股票代码。列名和含义到二期再定。

一期 `features.extra` 为空。二期把这张表按索引拼到 Alpha158 后面，仍用 `features` 里的 `DatasetH` 和 `models` 的同一接口。对照实验是同一模型的两行：`extra` 为空，以及 `extra` 带上趋势因子。

## 9. 实施顺序

1. 按 `data` 段下载并冻结数据目录，通过其中的检查。
2. 只启用 `models.lightgbm`，跑通训练、验证、测试和回测，确认标签没有跨段。
3. 打开其余 `dataset_kind: tabular` 的模型，用同一份 `data`、`segments`、`label`、`backtest` 重跑。
4. 按 `report` 段写出对照表，并标明选中的模型。

完成定义：

- [ ] 港股 bin 与 `cn_data` 分开，输出目录带上配置副本和 `data.freeze_dirname`
- [ ] 切分边界来自 `segments`，跨段空档来自 `label.horizon_trading_days`
- [ ] 股票池只读 `universe`，报告里有每年入池数量
- [ ] 启用的表格模型共用 `data`、`segments`、`label`、`features`、`backtest`，差异只在各自的 `models` 块
- [ ] 回测费用由 `backtest.costs` 相加，`limit_threshold` 为空
- [ ] 选模只看 `report.selection`，测试集按 `publish_test_for` 公布
- [ ] `features.extra` 可以加列，二期不用改训练循环
- [ ] 改标签、模型、切分或报告时，只改 `qlib-zh/hk/baseline.yaml` 里对应的段
