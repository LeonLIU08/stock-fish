# 004 策略决定下单数量，账本只成交或跳过

- 状态: 未完成
- 立案: 2026-09-04
- 完成: —
- Commit: —
- 依赖: [003-portfolio-framework.md](003-portfolio-framework.md)（账本 `try_fill` 已按本设计预留）

策略决定买/卖多少股。账本只做：数量合法且买得起 → **按请求数量原样成交**；否则 **整单 skip**，不把 500 股改成 300 股。

`ma_cross` 的 all_in / fraction 从引擎挪到策略（或策略的 `size_order`）。003 不得把「按剩余现金算股数」写进 `PortfolioBook`。

## 目标契约

```
策略:  (action, qty: int)     # 买单 qty 为手数整数倍；卖单与今日引擎相同，fraction 可不整手
账本:  if affordable(qty): fill exactly qty
      else: skip「资金不足」或「可卖数量不足」
```

- 买：`notional + fee <= cash` 且 `qty % lot_size == 0` 且 `qty > 0` 才成交。
- 卖：`qty <= holdings[symbol]` 且 `qty > 0` 才成交。**不**因非整手拒卖（今日 `_resolve_sell_qty` 不做手数对齐；要对齐就单独立项）。
- **禁止**账本对请求数量做 `min(qty, max_shares(cash))` 式缩量。

现有引擎的 all_in（`qty = max_shares(剩余现金)`）是**策略/适配器的产出**，不是账本规则。004 之后由 `MACross.size_order` 算出这个 qty，再交给账本。

## 与 003 的分工

| | 003 | 004 |
|---|---|---|
| `PortfolioBook.try_fill(order)` | 做：原样成交或 skip | 不改语义 |
| `legacy_resolve_qty(...)` | 暂放 `engine`/`orders.py`，供 `simulate` 与组合账本**在入账前**调用 | 删除调用；改由策略 `size_order` |
| `ma_cross` | 不改，仍只写 `buy`/`sell` | 实现 `size_order`（all_in / fraction） |
| 单票 A 组等价 | `legacy_resolve_qty` + `try_fill` ≡ 今日 `simulate()` | `size_order` + `try_fill` ≡ 003 结果（再 ≡ 今日 all_in） |

003 实现检查：`PortfolioBook` 内不得出现 `position_mode` / `signal_size` / `max_shares` 决定买单股数。这些只出现在 `legacy_resolve_qty`。

## 策略接口（004 实现）

`prepare()` 仍只根据 K 线写方向（及可选 `signal_size` 分数）。数量在**成交时**才知道现金，因此增加 fill-time 钩子：

```python
@dataclass
class SizeContext:
    symbol: str
    side: str            # buy / sell
    price: float         # 待成交开盘价
    cash: float
    holdings: Mapping[str, int]
    lot_size: int
    capital: float
    signal_size: Optional[float]   # 策略在 prepare 里写的 0~1，可空
    position_mode: str             # 过渡期基类默认可读 config；账本不读。004 不把 slot/1/N 塞进这里

class Strategy:
    def size_order(self, ctx: SizeContext) -> int:
        """返回拟成交股数；0 表示放弃本单。"""
```

`ma_cross` 默认（**与今日引擎同义**，003 的 `legacy_resolve_qty` 也必须如此）：

- 买 + all_in：`cost.max_shares(ctx.cash, ctx.price, lot)`
- 买 + fraction：`max_shares(ctx.cash * fraction, ...)`
- 卖 + all_in：`ctx.holdings[symbol]`
- 卖 + fraction：`int(holdings * fraction)`，**不要**再向下取整到手数（今日 `_resolve_sell_qty` 不做这一步；手数对齐单独立项，不要混进 F5）

可选列 `signal_qty`：若 prepare 已写出明确股数，`size_order` 直接返回它（供单测打出「请求 500、只买得起 300 → skip」）。

基类 `Strategy.size_order` 默认：有 `signal_qty` 用 qty，否则走与今日相同的 all_in/fraction（可读 `ctx.position_mode`）。账本不读 `position_mode`。`ma_cross` 覆盖同行为。slot / max_weight / 等权不要做进本计划。

## 非目标

- [ ] 组合等权、slot、max_weight（仍不是账本的事；要做就新开策略或 005，写在 `size_order`）
- [ ] 卖出 fraction 按手数再取整（今日引擎不做；单独立项，禁止夹进 F5）
- [ ] 部分成交、冰山单
- [ ] 做空
- [ ] 与 003 并行开工（A 组未绿、003 未合入之前不做 004）

## 清单

- [ ] 信号契约：文档化 `signal_qty`（可选）；`attach_signals` 可写 qty
- [ ] `SizeContext` + `Strategy.size_order`（基类默认：有 `signal_qty` 用 qty，否则走与今日相同的 all_in/fraction，避免未改的策略立刻坏掉）
- [ ] `ma_cross` 覆盖 `size_order`，行为与今日引擎一致
- [ ] `simulate()` 与组合路径都改为：`qty = strategy.size_order(ctx)` → `try_fill`
- [ ] 删除成交路径上的 `legacy_resolve_qty`（或仅测遗留适配器）
- [ ] `position_mode` 从「引擎必读」降为「ma_cross 读取的策略参数」；账本不再分支 all_in/fraction
- [ ] 单测见下；003 A 组在 004 后仍绿

## 测试

- [ ] **F1** 显式 `signal_qty=500`，现金只够 300 → skip，现金与持仓不变（禁止缩成 300）
- [ ] **F2** 显式 `qty=300`，现金够 500 → 成交 300，剩余现金留下
- [ ] **F3** 卖出 `qty > holdings` → skip，持仓不变
- [ ] **F4** `ma_cross` + all_in 单票：与 003 / 今日 `simulate()` 仍逐笔一致
- [ ] **F5** `ma_cross` + fraction：与今日 `_resolve_buy_qty` / `_resolve_sell_qty` 逐笔一致（卖出为 `int(shares * fraction)`，不按手数再取整）
- [ ] **F6** 组合两票：A 的 `size_order` 只要 40% 现金对应股数，B 仍能买（数量来自策略，不是账本均分）

## 验收

1. 账本模块无 `position_mode` 分支。
2. 缩量成交不存在：请求多少要么全成要么 skip。
3. 默认 `ma_cross` 独立核算结果不因 004 改变（含 fraction 卖出取整规则）。
4. changelog + 本文件勾选。

依赖：003 A 组已绿并合入。不要在本计划夹带 slot / max_weight。
