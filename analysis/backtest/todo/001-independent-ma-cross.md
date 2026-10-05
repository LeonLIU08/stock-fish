# 001 分股独立核算 + 均线金叉回测框架

- 状态: 已完成
- 立案: 2026-09-04
- 完成: 2026-09-04
- Commit: `455f625`

## 范围

每股独立资金、独立账户，策略与成交分离：策略只写标准信号列，引擎按收盘信号、次根开盘成交。

## 清单

- [x] YAML 配置（标的、区间、资金、成本、基准、参数方案）
- [x] `BacktestConfig` / `CostModel`（手数、佣金、双边印花税）
- [x] 策略插件：`Strategy` / `StrategyVariant` / registry
- [x] 均线金叉 `ma_cross`（多组 fast/slow）
- [x] 单票引擎 `simulate()`：只做多、`all_in`/`fraction`、每日成交上限
- [x] 港股 K 线与恒生 / 恒生科技基准加载
- [x] 指标、样本内/外、图表、Markdown/HTML 报告
- [x] `python scripts/run_backtest.py`
- [x] `tests/test_ma_backtest.py`

## 非目标（当时）

- 组合共享资金
- 做空、融资
