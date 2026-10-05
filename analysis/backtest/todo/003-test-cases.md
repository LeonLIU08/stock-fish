# 003 测试用例（防误差）

- 状态: 未完成
- 隶属: [003-portfolio-framework.md](003-portfolio-framework.md)
- 实现文件（建议）: `tests/test_portfolio_backtest.py`

本页是组合框架的验收清单。实现时按条打勾；**单票 all_in 与现有 `simulate()` 逐笔一致**是最高优先级。

成交分两步，测试不要混在一起：

1. `legacy_resolve_qty`：今日 all_in/fraction → 股数（A 组、C5）
2. `try_fill(qty)`：原样成交或 skip，**禁止缩量**（C7，004 的账本契约）

## 对齐约定（所有等价用例共用）

对照对象：同一套 bars、同一 `BacktestConfig` 资金/成本/手数/`position_mode`/`max_trades_per_day`/`timezone`。

- 独立路径：`independent=true`，`simulate(bars, cfg, symbol=S)`
- 组合路径：`independent=false`，`symbols=[S]`，`simulate_portfolio({S: bars}, cfg)`

必须**逐项相等**（股份、笔数用 `==`；金额用 `assert_allclose(..., rtol=0, atol=1e-8)`，禁止只比收益率小数位）：

| 对象 | 对齐字段 |
|---|---|
| 成交 | `timestamp, side, price, shares, notional, fee, cash_after, equity_after, equity_gross_after, reason, skipped` |
| 跳过 | 条数、时间、方向、原因字符串 |
| 权益序列 | 行数、`datetime` 完全一致；`equity, equity_gross, cash, shares, fee, turnover, leverage` |
| 期末 | `final_cash, final_shares, cumulative_fees` |

单票时组合权益的 `n_positions` 必须等于 `1 if shares>0 else 0`，`market_value` 必须等于 `shares * close`。`leverage` 分母是初始 `capital`，不是当期 equity。`turnover` 分母同此。

禁止：组合路径多出「合并日历」空行、少记某根 bar、拆成 open/close 两行、或用收盘价成交（现有引擎是次根开盘、收盘盯市）。

跳过原因字符串必须与今日引擎 **完全一致**（A 组 `==`）：`已持仓`、`空仓`、`资金不足`、`当日交易次数已达上限`、`无效开盘价`、`可卖数量为 0`。

---

## A. 单票等价（相对现有引擎，零误差）

组合里只有 1 只股票、每次 all_in 买/卖时，与当前策略引擎结果必须一致。

- [ ] **A1 均线金叉全路径（零费用）**  
  复用 `tests/test_ma_backtest.py` 的 `_bars` + `compute_ma_cross_signals(..., 5, 10)`（先平后涨后跌）。  
  `capital=100000`，`lot_size=1`，佣金/印花税=0，`all_in`。  
  买卖序列、股数、现金、权益曲线与 `simulate()` 一致；期末空仓时 `final_shares=0`。

- [ ] **A2 均线金叉 + 真实费率**  
  同上，但 `commission_rate=0.0003`，`stamp_duty_rate=0.001`。  
  每笔 `fee`、`cash_after`、`cumulative_fees` 与单票引擎一致（费用误差会在后续净值上放大）。

- [ ] **A3 手数向下取整**  
  复用现有 `test_lot_size_rounds_down_buy_quantity` 的资本 `10500`、`lot_size=100`。  
  组合买入股数 == `cost.max_shares(...)`，且 `% 100 == 0`，并与 `simulate(..., symbol=TEST)` 相同。

- [ ] **A4 分钟线 + 每日成交上限**  
  复用 `test_max_two_trades_per_day_on_minute_bars`（5m、一天内多次买卖、`max_trades_per_day=2`）。  
  成交恰好 buy+sell；skipped 含「当日交易次数已达上限」；权益列齐全。

- [ ] **A5 无信号 / 全程 hold**  
  全 `hold` 的日线。两边都无成交；权益恒等于初始资金；`shares` 全 0。

- [ ] **A6 买得起后卖光（手工信号）**  
  用 `attach_signals` 做一根买、若干 hold、一根卖，保证 fill 在 next open。  
  买卖价、股数、卖出后现金与单票引擎一致（验证组合事件循环没有把「次根」理解成「下一自然日」）。

- [ ] **A7 资金刚好不够一手**  
  价格与 lot 使得 `max_shares=0`。两边都 skipped「资金不足」，现金不动。

- [ ] **A8 短序列**  
  `len(bars) < 2`。两边都无成交、返回空/初始资金，不抛异常。

- [ ] **A9 独立开关回归**  
  同一单票配置 `independent=true` 跑现有 runner/引擎，与改造前单测行为一致（现有 `EngineTests` 全绿）。

- [ ] **A10 已持仓拒绝加仓**  
  空仓买进后，持仓期间再给 `buy` 信号。两边都 skip「已持仓」，股数不变（事件循环闸门，不是 `try_fill` 缩量）。

- [ ] **A11 最后一根不挂单**  
  倒数第一根 close 为 `buy` 且无后续 bar。两边都无这笔成交、无 pending 残留（现逻辑 `i < n-1`）。

建议实现一个 `assert_single_name_portfolio_matches_simulate(bars, cfg, symbol)`，A1–A7、A10、A11 都走它，避免各用例比对口径漂移。

---

## B. 会计恒等式（多票也会用）

不依赖「和旧引擎比」，用来抓组合盯市算错。

- [ ] **B1** 每一行：`equity == cash + Σ holdings[s] * last_price[s]`（`atol=1e-8`）
- [ ] **B2** 每一行：`equity_gross == cash_gross + Σ holdings[s] * last_price[s]`
- [ ] **B3** `cumulative_fees == sum(t.fee for t in trades)`（不含 skipped）
- [ ] **B4** `cash >= -1e-8`（不允许因取整/费用顺序变成明显负现金）
- [ ] **B5** 各票买入后 `shares % lot_size == 0`（卖出 fraction 后允许与今日一样出现非整手，A 组 all_in 仍整手）
- [ ] **B6** `n_positions == 持仓股数>0 的票数`
- [ ] **B7** `market_value == equity - cash`
- [ ] **B8** `leverage == market_value / capital`（`capital` 为初始资金且 >0，不是 / equity）
- [ ] **B9** skip 不增加该票当日成交计数；只有成功 fill 消耗 `max_trades_per_day`

---

## C. 共享资金与排队（框架语义，不是分散策略）

- [ ] **C1 第二票买不到**  
  两票同日历。t0 仅 A 买（all_in），现金耗尽；t1 仅 B 买。  
  B skipped「资金不足」；A 持仓不变。

- [ ] **C2 先卖后买（同时间戳）**  
  同一时刻 A 卖、B 买（YAML 顺序 A 在 B 前）。  
  B 的成交现金来自 A 卖出后余额；若先买后卖则会失败——本用例锁定「先卖后买」。

- [ ] **C3 同时买单按名单顺序**  
  同一开盘 A、B 都买，all_in，现金只够一只。  
  只成交 YAML 里靠前的；另一只资金不足。  
  **断言不得把现金均分**（两票都成交且股数按 50/50 即失败）。

- [ ] **C4 名单顺序反过来**  
  与 C3 相同 bars，仅 `symbols` 写成 `[B, A]`。成交方互换。证明平局规则是名单顺序不是代码字母序写死。

- [ ] **C5 fraction 不掏空**  
  `legacy_resolve_qty` + `position_mode=fraction`，A 的 `signal_size=0.4`。  
  A 成交后仍有现金，B 可以成交。数量来自适配器，不是账本均分。  
  卖出 fraction 的股数与今日 `_resolve_sell_qty` 相同（`int(shares * fraction)`，不按手数再取整）。

- [ ] **C6 卖出只动该票**  
  A、B 都持仓时只卖 A。B 股数不变；现金增加额 = A 成交额 - A 费用。

- [ ] **C7 显式 qty 不缩量（004 契约，003 的 try_fill 就要满足）**  
  直接 `try_fill(buy, qty=500)`，现金只够约 300 股 → skip，现金与持仓不变。  
  不得成交 300。all_in 的「能买几手买几手」只允许出现在 `legacy_resolve_qty`，不允许出现在 `try_fill`。

- [ ] **C8 已持仓时第二票仍可买**  
  A 已持仓，同时刻 A 再买、B 买。A skip「已持仓」；B 仍按自己的 qty 去 `try_fill`（闸门按票，不是整本账户锁死）。

---

## D. 时间轴（组合特有误差源）

- [ ] **D1 单票时间轴零膨胀**  
  A1 的权益 `datetime` 与 `simulate()` 的逐根 bar 一致（不能因「合并日历」多出午夜空点）。

- [ ] **D2 缺棒盯市**  
  A 每日有 bar，B 缺少中间一天。缺棒日组合仍有一行；B 用上一日收盘计价；不得把 B 持仓清零。

- [ ] **D3 只在该票的 open 成交**  
  B 的 fill 不得发生在「只有 A 的时间戳」上。成交 `timestamp` 必须是该票自己的下一根 bar 开盘时间。

- [ ] **D4 时区**  
  naive / `Asia/Hong_Kong` 混用时，成交日与 `max_trades_per_day` 的「日」与单票引擎相同（HK 日历日）。

- [ ] **D5 并集日历不延长该票挂单**  
  A 的最后一根 close 为 `buy`，B 在更晚还有 bar。A **不得**在 B 的后续 open 成交；A 最后一根不挂 pending（挂单条件是该票自己的下一根，不是组合下一根）。

---

## E. 配置与 runner

- [ ] **E1** `independent=false` 且未配 `capital.total`：**必须报错**（`ValueError` 或等价），禁止回退为 `per_symbol` / `per_symbol × N`。单测锁死「缺字段即失败」。
- [ ] **E2** `independent=true` 时 `capital.per_symbol` 语义不变；组合账本不被调用（可用 mock/spy 或结果结构断言）。即使 YAML 里同时写了 `capital.total` 也不走组合路径。
- [ ] **E3** 多票 + 多 variant：每个 variant **一次**组合结果，而不是 `N_symbols × N_variants` 份独立 30 万账户。

---

## 推荐落地顺序

1. `assert_single_name_portfolio_matches_simulate` + **A1、A2、A3、A6、A10、A11**（先保证零误差与闸门）
2. B 恒等式挂在所有组合用例末尾
3. C 共享资金（含 C8）
4. A4、A5、A7、A8、A9、D（含 D5）、E（E1 只测报错）

A 未全绿之前不要认为组合框架可合并；不要并行 004。
