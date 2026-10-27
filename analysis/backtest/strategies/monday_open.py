"""每个交易日开盘买入，小时检查 +10% 止盈 / -5% 止损。

每只股票一个独立账户。空仓在每个交易日第一根小时 K 的开盘买一次；
当天没有 K 线就跳过这一天。已有持仓不加仓。
持仓收益 = 检查价 / 买入成交价 - 1，不含费用、不含分红。
「超过」是严格不等式：刚好 +10% 或刚好 -5% 不卖。

离场有两个因果检查点，都不用小时线的最高价和最低价：
- 这根 K 线的开盘已经穿过阈值（跳空）：按该开盘价卖出。
- 收盘才穿过：下一根 K 线开盘卖出。卖出决定不因下一根开盘回到区间内而取消。
跳空时成交价可以差于 -5% 或好于 +10%，不把阈值当成限价。

同一时点先卖后买会被规则对冲掉，因此开盘若用来止盈或止损，当天不再开仓。
当天稍后打回空仓，也要等到下一个交易日开盘。卖出后用剩余现金满仓，
按整手取整。样本结束时仍持仓的，按最后一根收盘盯市，不强制平仓。
"""
from __future__ import annotations

from statistics import median
from typing import List, Optional, Sequence

import pandas as pd

from analysis.backtest.config import BacktestConfig
from analysis.backtest.engine import EngineResult, Trade
from analysis.backtest.signals import PreparedBars, SignalAction, attach_signals
from analysis.backtest.strategy.base import Strategy, StrategyVariant
from analysis.backtest.strategy.registry import register_strategy

_EPS = 1e-6
_REASON_ENTRY = "daily_open"
_REASON_TAKE_PROFIT = "take_profit"
_REASON_STOP = "stop_loss"


def breach_reason(price: float, entry: float, take_profit: float, stop_loss: float) -> Optional[str]:
    """价格相对买入价是否严格超过止盈或止损。"""
    if entry <= 0 or price <= 0:
        return None
    ret = price / entry - 1.0
    if ret > take_profit + _EPS:
        return _REASON_TAKE_PROFIT
    if ret < -stop_loss - _EPS:
        return _REASON_STOP
    return None


def session_open_flags(datetimes: Sequence, tz: str) -> List[bool]:
    """每个交易日的第一根 K 线为 True。"""
    ts = pd.to_datetime(pd.Series(list(datetimes)))
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(tz)
    else:
        ts = ts.dt.tz_convert(tz)
    flags: List[bool] = []
    prev = None
    for day in ts.dt.date:
        flags.append(bool(day != prev))
        prev = day
    return flags


def simulate_monday_open(
    df: pd.DataFrame,
    config: BacktestConfig,
    symbol: str,
    *,
    take_profit: float = 0.10,
    stop_loss: float = 0.05,
) -> EngineResult:
    """按每日开盘买入、小时检查止盈止损重放。返回与通用引擎相同的结果。"""
    if config.interval != "60m":
        raise ValueError(f"monday_open 只接受 60m K 线，收到 {config.interval}")
    empty_eq = pd.DataFrame(columns=[
        "datetime", "equity", "equity_gross", "cash", "shares", "price",
        "position", "turnover", "leverage", "fee",
    ])
    if df is None or df.empty:
        return EngineResult(equity=empty_eq, final_cash=config.capital)

    cost = config.cost
    lot_size = config.lot_size_for(symbol)
    cash = float(config.capital)
    cash_gross = float(config.capital)
    capital = float(config.capital)
    shares = 0
    entry = 0.0
    pending: Optional[str] = None
    trades: List[Trade] = []
    skipped: List[Trade] = []
    equity_rows = []
    cumulative_fees = 0.0
    opens = df["open"].astype(float).tolist()
    closes = df["close"].astype(float).tolist()
    times = df["datetime"].tolist()
    is_session_open = session_open_flags(times, config.timezone or "Asia/Hong_Kong")
    n = len(df)

    def snapshot(ts, side, price, qty, fee, reason, skipped_trade=False) -> Trade:
        return Trade(
            timestamp=pd.Timestamp(ts),
            side=side,
            price=float(price),
            shares=int(qty),
            notional=float(price * qty),
            fee=float(fee),
            cash_after=float(cash),
            equity_after=float(cash + shares * price),
            equity_gross_after=float(cash_gross + shares * price),
            reason=reason,
            skipped=skipped_trade,
        )

    def sell_all(ts, price, reason) -> None:
        nonlocal cash, cash_gross, shares, entry, pending, cumulative_fees
        qty = shares
        notional = qty * price
        fee = cost.fee(notional)
        cash += notional - fee
        cash_gross += notional
        shares = 0
        entry = 0.0
        pending = None
        cumulative_fees += fee
        trades.append(snapshot(ts, "sell", price, qty, fee, reason))

    for i in range(n):
        ts = times[i]
        o = opens[i]
        c = closes[i]
        fee_bar = 0.0
        turnover = 0.0
        sold = False

        if shares > 0 and pending and o > 0:
            fee_before = cumulative_fees
            sell_all(ts, o, pending)
            fee_bar = cumulative_fees - fee_before
            turnover = trades[-1].notional / capital if capital else 0.0
            sold = True
        elif shares > 0 and o > 0:
            gap = breach_reason(o, entry, take_profit, stop_loss)
            if gap:
                fee_before = cumulative_fees
                sell_all(ts, o, f"{gap}_gap")
                fee_bar = cumulative_fees - fee_before
                turnover = trades[-1].notional / capital if capital else 0.0
                sold = True

        if not sold and shares == 0 and is_session_open[i] and o > 0:
            qty = cost.max_shares(cash, o, lot_size=lot_size)
            if qty > 0:
                notional = qty * o
                fee = cost.fee(notional)
                cash -= notional + fee
                cash_gross -= notional
                shares = qty
                entry = o
                fee_bar = fee
                turnover = notional / capital if capital else 0.0
                cumulative_fees += fee
                trades.append(snapshot(ts, "buy", o, qty, fee, _REASON_ENTRY))
            else:
                skipped.append(snapshot(ts, "buy", o, 0, 0.0, "一手金额超过可用现金", True))

        if shares > 0 and i < n - 1 and c > 0:
            hit = breach_reason(c, entry, take_profit, stop_loss)
            if hit:
                pending = hit

        mark = c if c > 0 else (o if o > 0 else 0.0)
        mv = shares * mark
        equity_rows.append({
            "datetime": ts,
            "equity": cash + mv,
            "equity_gross": cash_gross + mv,
            "cash": cash,
            "shares": shares,
            "price": mark,
            "position": 1 if shares > 0 else 0,
            "turnover": turnover,
            "leverage": (mv / capital) if capital else 0.0,
            "fee": fee_bar,
        })

    return EngineResult(
        trades=trades,
        skipped=skipped,
        equity=pd.DataFrame(equity_rows),
        final_cash=cash,
        final_shares=shares,
        final_price=float(closes[-1]) if closes else 0.0,
        cumulative_fees=cumulative_fees,
    )


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def summarize_monday_open(results: Sequence[dict]) -> str:
    """跨股票汇总。均值等于这些独立账户等权放在一起的收益，不是单一 100 万组合。"""
    rows = [row for row in results if (row.get("metrics") or {}).get("total_return") is not None]
    if not rows:
        return "没有可汇总的成交结果。"
    rets = [float(row["metrics"]["total_return"]) for row in rows]
    drawdowns = [float(row["metrics"]["max_drawdown"]) for row in rows]
    excesses = []
    for row in rows:
        bh = (row.get("metrics") or {}).get("buy_hold_return")
        if bh is not None:
            excesses.append(float(row["metrics"]["total_return"]) - float(bh))
    open_positions = sum(1 for row in rows if (row.get("metrics") or {}).get("open_position"))
    reasons = {"take_profit": 0, "take_profit_gap": 0, "stop_loss": 0, "stop_loss_gap": 0, "daily_open": 0}
    skipped_lots = 0
    round_trips = 0
    for row in rows:
        round_trips += int((row.get("metrics") or {}).get("n_round_trips") or 0)
        for trade in row.get("trades") or []:
            reason = str(trade.get("reason") or "")
            if reason in reasons:
                reasons[reason] += 1
        skipped_lots += sum(1 for trade in (row.get("skipped") or []) if not trade.get("side") or trade.get("side") == "buy")

    rets_sorted = sorted(rets)
    beat = sum(1 for item in excesses if item > 0)
    lines = [
        "## 每日开盘策略汇总",
        "",
        "每只股票单独一个 100 万港币账户，账户之间不调仓、不共享现金。",
        "均值是这些账户等权放在一起的收益；中位数是单只股票的典型结果。",
        "止盈止损打在前复权价上，除息缺口不会单独触发止损。",
        "跳空按开盘价成交，实际盈亏可以超出 +10% / -5%。样本结束仍持仓的按最后收盘盯市。",
        "",
        f"- 有结果的股票: {len(rows)}",
        f"- 收益中位数: {_pct(median(rets))}，均值: {_pct(sum(rets) / len(rets))}",
        f"- 收益 25% / 75% 分位: {_pct(rets_sorted[len(rets_sorted) // 4])} / {_pct(rets_sorted[(len(rets_sorted) * 3) // 4])}",
        f"- 正收益占比: {_pct(sum(1 for item in rets if item > 0) / len(rets))}",
        f"- 最大回撤中位数: {_pct(median(drawdowns))}",
        f"- 跑赢自身买入持有: {beat} / {len(excesses)}，超额中位数 {_pct(median(excesses) if excesses else None)}",
        f"- 完整回合: {round_trips}，期末仍持仓: {open_positions}",
        f"- 买入次数: {reasons['daily_open']}",
        f"- 止盈: 收盘触发 {reasons['take_profit']}，跳空 {reasons['take_profit_gap']}",
        f"- 止损: 收盘触发 {reasons['stop_loss']}，跳空 {reasons['stop_loss_gap']}",
        f"- 买不起一手而跳过的开盘: {skipped_lots}",
    ]
    return "\n".join(lines)


@register_strategy
class MondayOpenStrategy(Strategy):
    name = "monday_open"
    display_name = "每日开盘止盈止损"

    def variants(self, config: BacktestConfig) -> List[StrategyVariant]:
        take_profit = float(config.strategy_params.get("take_profit", 0.10))
        stop_loss = float(config.strategy_params.get("stop_loss", 0.05))
        if not (0 < stop_loss < take_profit < 10):
            raise ValueError(f"止盈止损非法: take_profit={take_profit}, stop_loss={stop_loss}")
        variant = StrategyVariant(
            key="band",
            label=f"每日开盘 / 止盈{take_profit:.0%} / 止损{stop_loss:.0%}",
            params={"take_profit": take_profit, "stop_loss": stop_loss},
        )
        if config.variants and variant.key not in config.variants:
            raise ValueError(f"未知方案 {config.variants}，可选: [{variant.key}]")
        return [variant]

    def warmup_bars(self, config: BacktestConfig, variant: StrategyVariant) -> int:
        return 0

    def prepare(self, bars: pd.DataFrame, config: BacktestConfig, variant: StrategyVariant) -> PreparedBars:
        actions = [SignalAction.HOLD.value] * len(bars)
        return PreparedBars(
            bars=attach_signals(bars, actions),
            context={
                "take_profit": variant.params["take_profit"],
                "stop_loss": variant.params["stop_loss"],
            },
        )

    def simulate(self, bars: pd.DataFrame, config: BacktestConfig, variant: StrategyVariant, symbol: str) -> EngineResult:
        return simulate_monday_open(
            bars,
            config,
            symbol,
            take_profit=float(variant.params["take_profit"]),
            stop_loss=float(variant.params["stop_loss"]),
        )

    def summarize_results(self, results: Sequence[dict]) -> str:
        return summarize_monday_open(results)

    def result_metadata(self, variant: StrategyVariant) -> dict:
        return {
            "variant_params": dict(variant.params),
            "rules": {
                "entry": "daily_first_hour_open",
                "take_profit": variant.params["take_profit"],
                "stop_loss": variant.params["stop_loss"],
                "check": "hourly_open_gap_or_close_then_next_open",
                "reentry": "next_session_open_only",
            },
        }
