# 003 组合回测框架

- 状态: 未完成
- 立案: 2026-09-04
- 完成: —
- Commit: —

只改**框架**：一份现金、多票成交、一条组合净值。账本按**已给定股数**原样成交或 skip。

买多少在 003 里仍由临时函数 `legacy_resolve_qty` 按现有 `position_mode` 计算（好让单票 A 组与今日 `simulate()` 一致）；该函数**不得**写进 `PortfolioBook`。004 把它换成像 `ma_cross.size_order` 这样的策略钩子。详见 [004-strategy-order-qty.md](004-strategy-order-qty.md)。

## 原则

| 层 | 负责 | 不负责 |
|---|---|---|
| 账本 `try_fill` | 共享现金、持仓、费用、排队、盯市；**给定 qty 则全成或 skip** | 计算买多少、组合权重 |
| 003 `legacy_resolve_qty` | 把今日 `all_in`/`fraction` 变成 qty（过渡） | 进入 PortfolioBook 类内部 |
| 策略（004 才改 ma_cross） | 方向 + 最终 qty | 账户记账 |

同一时刻两笔买单：按 YAML 名单排队，各自带自己的 qty 去 `try_fill`。不把现金均分。

`independent=false` **必须**配 `capital.total`，缺了就报错。禁止回退成 `per_symbol` 或 `per_symbol × N`。

合入后不要用 `ma_cross` + `all_in` 的组合净值做策略结论：名单第一只会吃光现金（C3）。等权买入持有是 P1 报告对照，不是 003 的成交路径。004 在 A 组全绿并合入之前不要并行。

## 事件循环（与今日 `simulate()` 对齐，A 组零误差）

这些是现有单票引擎已经在做、组合路径必须逐字复制的行为。写进账本规格，避免实现时另发明一套。

1. **每股一个 pending，容量为 1。** 只在该票自己的序列里「后面还有 K 线」时挂单（现逻辑 `i < n-1`）。不得按组合并集日历的下一根去挂。成交只发生在该票自己的下一根 open。
2. **只做多、不平仓不加仓（事件循环政策，不是 `try_fill` 缩量，003 也不下放到 `ma_cross`）。**
   - 入队：`holdings[s]==0` 才挂买单，`>0` 才挂卖单。
   - 成交时再检查一遍：买要求当时仍空仓，卖要求当时仍有仓。
   - 否则 skip，原因字符串与今日引擎相同：`已持仓` / `空仓`（以及 `当日交易次数已达上限` / `无效开盘价` / `资金不足` / `可卖数量为 0`）。
3. **盯市时点。**
   - 单票（`symbols` 长度为 1）：按该票 bars 循环，先 open 成交再 **close** 盯市，权益行数与 `datetime` 与 `simulate()` 相同。禁止因「合并日历」多出午夜空点或拆成 open 一行 + close 一行。
   - 多票：时间戳 = 各票 `datetime` 的并集；该时刻没有 bar 的票沿用上一次收盘；同一时刻先处理全部卖、再处理全部买，买卖内部按 YAML `symbols` 顺序。每条时间戳只拍 **一行** 权益（成交后、用 close 或沿用价盯市）。
4. **资金。** `independent=false` 且未配 `capital.total` → 明确错误，不回退。`independent=true` 仍只用 `capital.per_symbol`，不调用组合账本。
5. **计数与比率（与今日定义一致，只把分母换成组合初始资金）。**
   - `max_trades_per_day` 仍按 **每股**；只对成功 `try_fill` 计数，skip 不加。
   - `turnover = 该事件 notional / capital`（`capital` = 组合 `capital.total` 或单票初始资金）。
   - `leverage = market_value / capital`（初始资金，**不是** / equity）。

`legacy_resolve_qty` 必须与今日 `_resolve_buy_qty` / `_resolve_sell_qty` **同义**：卖出 fraction 只做 `int(shares * fraction)`，**不要**在 003 里向下取整到手数。手数对齐若要做，单独立项，不要混进 A 组。

## 解耦检查（相对「策略买入资金分配」）

**结论：003 的账本必须已经符合 004 契约；all_in 只允许活在账本外的过渡适配器里。**

账本允许：

- `try_fill(symbol, side, qty)`：买得起 → 按 qty 成交；否则 skip，**不缩量**
- 买单：`qty > 0`、整手、`notional + fee <= cash`。卖单：`qty > 0` 且 `qty <= holdings`；**不**因非整手拒卖（与今日引擎一致）
- 先卖后买、按名单顺序处理同一时刻的订单
- 日成交上限（成功 fill 才计数）
- **不**负责「已持仓拒绝加仓」（那是事件循环入队政策）

账本禁止：

- `position_mode` / `signal_size` / `max_weight` / 1/N / slot
- `qty = max_shares(剩余现金)` 写在 `PortfolioBook` 内（那是「引擎决定买多少」）

003 为 A 组零误差保留：

```
qty = legacy_resolve_qty(position_mode, signal_size, cash, price, lot, holdings)
book.try_fill(..., qty)
```

`legacy_resolve_qty` 与今日 `_resolve_buy_qty` / `_resolve_sell_qty` 同义，供 `simulate()` 与组合路径共用。004 用 `strategy.size_order` 替换这一行。

P1「组合等权买入持有」只允许作为对照基准曲线，禁止写进成交路径。


## 非目标（本计划不做）

- [ ] `allocation: slot` / `max_weight: 1/N` / 等权再平衡（策略或后续 allocator）
- [ ] 做空、融资
- [ ] 用对齐后的公共 K 线代替「该票下一根开盘」
- [ ] 把独立模式的多份净值加总冒充组合
- [ ] 改 `ma_cross` 的发信逻辑（放 [004](004-strategy-order-qty.md)）
- [ ] 在 `PortfolioBook` 内根据剩余现金计算买单股数

## P0 账本与成交

- [ ] 配置：`universe.independent: false` 合法；**`capital.total` 必填**（缺则报错，不回退）；`independent: true` 行为与现在完全一致
- [ ] `Trade` 增加 `symbol`；独立路径同样带上，报告可统一
- [ ] 新建组合账本（建议 `analysis/backtest/portfolio.py`）
  - [ ] 共享 `cash` / `cash_gross`
  - [ ] `holdings: {symbol: shares}`
  - [ ] **`try_fill(symbol, side, qty)`：原样成交或 skip，禁止缩量**
  - [ ] 费用与「是否买得起」用现有 `CostModel`；只做校验，不用 `max_shares` 去改 qty
  - [ ] 事件循环按上文「事件循环」五条实现（每股 pending、已持仓闸门、单票/多票盯市、资金报错、fill 计数）
  - [ ] 信号与成交：该票收盘出信号 → 该票下一根开盘成交；最后一根不挂 pending
  - [ ] 同一时刻先卖后买（卖出所得可立刻用于后续买单）
  - [ ] 同时多笔买单：按 YAML `symbols` 顺序排队（撮合平局，不均分金额）
  - [ ] 每日成交上限：仍按**每股** `max_trades_per_day`；只计成功 fill
- [ ] 抽出 `legacy_resolve_qty`（与今日 `_resolve_buy_qty` / `_resolve_sell_qty` **同义**，卖出 fraction 不按手数再取整），**账本外**调用后再 `try_fill`；`simulate()` 改为同一对调用，避免两套规则
- [ ] 禁止新增 `allocation` / `max_weight` / 均分等组合配置；`position_mode` 只进 `legacy_resolve_qty`
- [ ] 组合盯市：每个时间戳成交之后拍一行 `equity = cash + Σ shares_i * last_price_i`（该时刻无 bar 的票沿用最新收盘；单票不得多行）
- [ ] 权益表字段：`datetime, equity, equity_gross, cash, market_value, n_positions, turnover, leverage, fee`；持仓明细可另表或宽列
- [ ] `turnover = notional / capital`，`leverage = market_value / capital`（均为初始资金）
- [ ] runner：`independent=false` 时每个 `variant` 只跑一次账本，不再每股复制一份资金

## P0 报告与指标

- [ ] 组合级 `compute_metrics`：总收益、年化、回撤、Sharpe、成交次数（可按票拆）
- [ ] 净值图：组合净值 vs 现有指数基准（恒生 / 恒生科技）
- [ ] 成交明细带股票代码；skipped 同样带代码与原因
- [ ] HTML/Markdown：每个 variant 一份组合报告（不再是「假组合」的三份独立账户首页加总）
- [ ] 独立模式报告保持不变

## P0 测试

细则与勾选清单见 [003-test-cases.md](003-test-cases.md)。核心要求：

- [ ] **单票 + all_in：组合路径与现有 `simulate()` 逐笔、逐根权益一致**（成交、费用、现金、股数、时间戳；金额 `atol=1e-8`）
- [ ] 会计恒等式：`equity == cash + Σ shares_i * price_i`
- [ ] 两票共享资金：第一票 all_in 后第二票资金不足
- [ ] 先卖后买：同时间戳卖 A 的钱能买 B
- [ ] 同时买单：YAML 顺序排队，不均分现金
- [ ] **C7 显式 qty 不缩量**（004 契约，003 的 `try_fill` 就要绿）
- [ ] `fraction` 经 `legacy_resolve_qty`，不是账本均分；卖出 fraction 与今日一样只 `int(shares * fraction)`，不按手数再取整
- [ ] 单票组合权益行数/时间戳不膨胀；最后一根不挂单；已持仓 skip「已持仓」
- [ ] `independent: true`：现有 `EngineTests` 全绿
- [ ] `independent: false` 缺 `capital.total` 报错，不回退

## P1 框架配套（003 账本合入之后；004 不要并行）

没有等权买入持有对照之前，不要用 `independent: false` 的均线组合净值做决策。图表可以后做。

- [ ] **优先**：组合等权买入持有基准（**仅报告对照，不参与成交、不写进 `PortfolioBook`**）：起点把 `capital.total` 按名单尽量分成 N 份各买各的，之后持有
- [ ] 持仓图：持股只数 / 各票市值，不再画单票 0/1
- [ ] 分股归因页：该票在组合里的买卖点叠 K 线（归因，不是独立 30 万账户）
- [ ] 可选：组合级 `max_trades_per_day`（账户级经纪约束，默认关闭）

## P2 以后

- [ ] 组合均匀定投基准（共享资金版本的 002）
- [ ] 样本内/外拆分沿用组合净值
- [ ] CLI：`--independent` / `--capital` 在组合模式下覆盖 `capital.total`

## 验收

1. YAML `independent: false` + `capital.total` 能跑通当前名单、现有均线方案（只证明账本没崩，**不是**策略结论）。
2. 未配 `capital.total` 时明确报错；同一配置改回 `independent: true`，结果与 003 之前一致。
3. 003 不改 `ma_cross`；账本无 `position_mode` 分支；已持仓闸门在事件循环不在 `try_fill`；数量只经 `legacy_resolve_qty` → `try_fill`（为 004 预留）。
4. [003-test-cases.md](003-test-cases.md) A 组全绿：单票 all_in 与现有引擎零误差（含跳过原因字符串）。
5. 文档（本文件勾选 + changelog 一行 + commit）。

## 建议实现顺序

1. 配置（`capital.total` 必填）、`Trade.symbol`、`try_fill` + `legacy_resolve_qty`（`simulate()` 先切到这一对，保证行为不变）
2. 组合事件循环按上文五条 + [003-test-cases.md](003-test-cases.md) A 组
3. C/B/D 组；含「显式 qty 买不起则 skip、不缩量」和「已持仓不加仓」（为 004 打底）
4. runner 分支 + E 组（E1 只测报错）
5. 指标与最小报告
6. P1 **先**等权买入持有基准，**再**图表；A 组未绿、003 未合入之前不要开工 004
