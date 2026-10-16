"""趋势回测里的买入持有、均匀定投和指数基准。

买卖用复权价。复权价里的股息因子要拿掉，否则除息日没跌的价格再加一笔股息，
同一笔钱会算两次。拆合股留在价格里，股数按当时的每股价格换算。
股息按「当时股数 × 每股股息 × (1 − 预扣税率)」计入收入和现金。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Sequence

import pandas as pd

from analysis.backtest.analytics import _dca_buy_lots_by_day
from analysis.backtest.trend.fills import whole_shares
from analysis.backtest.trend.markets import trade_fees
from analysis.backtest.trend.prices import dividend_credit


@dataclass
class BaselineResult:
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_return: float = 0.0
    dividend_income: float = 0.0
    final_equity: float = 0.0
    final_cash: float = 0.0
    final_shares: int = 0
    start_price: float = 0.0
    lot_size: int = 1
    n_lots: int = 0
    n_trading_days: int = 0
    interval: Optional[float] = None
    n_lots_bought: int = 0
    n_buy_days: int = 0


def split_adjusted_price(adjusted: float, factor: float, split_ratio: float) -> float:
    """复权价去掉累计股息因子，得到按拆合股调整后的成交价。"""
    if factor <= 0 or split_ratio <= 0:
        raise ValueError("复权因子和拆股比例必须为正")
    if adjusted <= 0:
        raise ValueError("复权价必须为正")
    return float(adjusted) * float(split_ratio) / float(factor)


def simulate_buy_hold(
    datetimes: Sequence,
    adjusted_opens: Sequence[float],
    adjusted_closes: Sequence[float],
    factors: Sequence[float],
    split_ratios: Sequence[float],
    dividends: Sequence[float],
    *,
    lot_size: int,
    capital: float,
    market,
    withholding: float,
) -> BaselineResult:
    """评估窗口第一根开盘按复权价整手买入，之后只收股息。"""
    count = _count(datetimes, adjusted_opens, adjusted_closes, factors, split_ratios, dividends)
    if count == 0:
        return BaselineResult(final_equity=capital, final_cash=capital, lot_size=lot_size)
    first_close = split_adjusted_price(adjusted_closes[0], factors[0], split_ratios[0])
    shares = _max_broker_shares(capital, _raw_price(first_close, split_ratios[0]), lot_size, market)
    lots = [0] * count
    if shares >= lot_size:
        lots[0] = shares // lot_size
    return _simulate(
        datetimes, adjusted_opens, adjusted_closes, factors, split_ratios, dividends,
        lot_size=lot_size, capital=capital, market=market, withholding=withholding, buy_lots=lots,
        start_price=first_close,
    )


def simulate_even_dca(
    datetimes: Sequence,
    adjusted_opens: Sequence[float],
    adjusted_closes: Sequence[float],
    factors: Sequence[float],
    split_ratios: Sequence[float],
    dividends: Sequence[float],
    *,
    lot_size: int,
    capital: float,
    market,
    withholding: float,
) -> BaselineResult:
    """间隔仍按第一根收盘能买的手数均分。每一手按当天复权开盘价成交，股息入账。"""
    count = _count(datetimes, adjusted_opens, adjusted_closes, factors, split_ratios, dividends)
    if count == 0:
        return BaselineResult(final_equity=capital, final_cash=capital, lot_size=lot_size)
    first_close = split_adjusted_price(adjusted_closes[0], factors[0], split_ratios[0])
    shares = _max_broker_shares(capital, _raw_price(first_close, split_ratios[0]), lot_size, market)
    n_lots = shares // lot_size if lot_size else 0
    return _simulate(
        datetimes, adjusted_opens, adjusted_closes, factors, split_ratios, dividends,
        lot_size=lot_size, capital=capital, market=market, withholding=withholding,
        buy_lots=_dca_buy_lots_by_day(n_lots, count),
        start_price=first_close,
    )


def _simulate(
    datetimes,
    adjusted_opens,
    adjusted_closes,
    factors,
    split_ratios,
    dividends,
    *,
    lot_size,
    capital,
    market,
    withholding,
    buy_lots,
    start_price,
) -> BaselineResult:
    cash = float(capital)
    original = Decimal(0)
    income = 0.0
    bought_lots = 0
    buy_days = 0
    rows = []
    for index, stamp in enumerate(datetimes):
        split_ratio = float(split_ratios[index])
        price_open = split_adjusted_price(adjusted_opens[index], factors[index], split_ratio)
        price_close = split_adjusted_price(adjusted_closes[index], factors[index], split_ratio)
        raw_open = _raw_price(price_open, split_ratio)
        broker = _broker_shares(original, split_ratio)
        dividend = float(dividends[index] or 0.0)
        if broker > 0 and dividend > 0:
            credit = dividend_credit(broker, dividend, withholding)
            cash += credit.cash
            income += credit.cash
        lots = buy_lots[index] if index < len(buy_lots) else 0
        if lots > 0 and raw_open > 0:
            wanted = lots * lot_size
            affordable = _max_broker_shares(cash, raw_open, lot_size, market)
            qty = min(wanted, affordable)
            qty = (qty // lot_size) * lot_size
            if qty > 0:
                notional = raw_open * qty
                fee = trade_fees(market, side="buy", notional=notional, shares=qty).total
                cash -= notional + fee
                original += Decimal(qty) / Decimal(str(split_ratio))
                bought_lots += qty // lot_size
                buy_days += 1
                broker = _broker_shares(original, split_ratio)
        value = float(original) * price_close
        rows.append({
            "datetime": stamp,
            "equity": cash + value,
            "equity_gross": cash + value,
            "cash": cash,
            "shares": broker,
            "price": price_close,
            "position": 1 if broker else 0,
            "turnover": 0.0,
            "leverage": 0.0 if capital <= 0 else value / float(capital),
            "fee": 0.0,
        })
    equity = pd.DataFrame(rows)
    final = float(equity["equity"].iloc[-1]) if not equity.empty else float(capital)
    n_lots = sum(buy_lots)
    count = len(datetimes)
    return BaselineResult(
        equity=equity,
        total_return=0.0 if capital <= 0 else final / float(capital) - 1.0,
        dividend_income=income,
        final_equity=final,
        final_cash=cash,
        final_shares=_broker_shares(original, float(split_ratios[-1])) if count else 0,
        start_price=float(start_price),
        lot_size=lot_size,
        n_lots=n_lots,
        n_trading_days=count,
        interval=(count / n_lots) if n_lots else None,
        n_lots_bought=bought_lots,
        n_buy_days=buy_days,
    )


def _max_broker_shares(cash: float, raw_price: float, lot_size: int, market) -> int:
    if cash <= 0 or raw_price <= 0 or lot_size < 1:
        return 0
    hi = int(float(cash) // (raw_price * lot_size)) + 1
    best = 0
    lo = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        shares = mid * lot_size
        if shares == 0:
            lo = 1
            continue
        notional = raw_price * shares
        fee = trade_fees(market, side="buy", notional=notional, shares=shares).total
        if notional + fee <= cash + 1e-6:
            best = shares
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _raw_price(split_adjusted: float, split_ratio: float) -> float:
    return float(split_adjusted) / float(split_ratio)


def _broker_shares(original: Decimal, split_ratio: float) -> int:
    if original == 0:
        return 0
    return whole_shares(original * Decimal(str(split_ratio)))


def _count(*columns) -> int:
    count = len(columns[0])
    if any(len(column) != count for column in columns):
        raise ValueError("买入持有的价格、因子和股息长度必须一致")
    return count
