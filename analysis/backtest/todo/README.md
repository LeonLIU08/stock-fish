# 回测开发计划（留底）

路径：`analysis/backtest/todo/`

本目录记录规则回测框架的全部开发计划。每条计划单独成文，完成后不删除，只改状态并写入 [changelog.md](changelog.md)。

## 状态约定

| 标记 | 含义 |
|---|---|
| `未完成` | 尚未开工 |
| `进行中` | 正在实现 |
| `已完成` | 已合入，见 changelog 与 commit |
| `取消` | 明确不做，保留原文说明原因 |

计划正文用 GitHub 勾选框：

- `[ ]` 未完成
- `[x]` 已完成

改状态时同时更新本页总览表、计划文首 `状态` 字段、勾选框，并在 changelog 追加一行。

## 总览

| ID | 计划 | 状态 | 文件 |
|---|---|---|---|
| 001 | 分股独立核算 + 均线金叉回测框架 | 已完成 | [001-independent-ma-cross.md](001-independent-ma-cross.md) |
| 002 | 单票均匀定投基准曲线 | 已完成 | [002-even-dca-benchmark.md](002-even-dca-benchmark.md) |
| 003 | 组合回测框架（共享资金 / 跨股下单 / 组合净值） | 未完成 | [003-portfolio-framework.md](003-portfolio-framework.md) |
| 003-test | 组合框架测试用例（单票 all_in 与现有引擎零误差） | 未完成 | [003-test-cases.md](003-test-cases.md) |
| 004 | 策略决定下单数量，账本只成交或 skip | 未完成 | [004-strategy-order-qty.md](004-strategy-order-qty.md) |

新增计划：复制编号（004、005…），加入上表，正文里写清范围、非目标、验收。

## 分层（避免和策略设计混淆）

- **账本 `try_fill`**：给定股数，买得起就原样成交，否则 skip。不算买多少、不做 1/N、不负责「已持仓拒绝加仓」。
- **事件循环（003）**：每股 pending（最后一根不挂单）、只做多不加仓、单票 close 盯市 / 多票并集日历、先卖后买、YAML 顺序。`independent=false` 必须有 `capital.total`，缺了报错不回退。
- **003 过渡**：`legacy_resolve_qty`（账本外）把今日 `all_in`/`fraction` 变成 qty，与 `_resolve_*_qty` 同义（卖出 fraction 不按手数再取整）。
- **004**：`ma_cross.size_order` 取代 `legacy_resolve_qty`；账本接口不变。A 组未绿前不要开工。
- **策略方向**：`signal_action`。仓位/等权/slot 写在策略的 `size_order`，不写进账本。
- **组合净值**：`ma_cross` + `all_in` + 名单排队不是分散策略；等权买入持有是 003 的 P1 报告对照，做完之前不要用组合均线净值做决策。
