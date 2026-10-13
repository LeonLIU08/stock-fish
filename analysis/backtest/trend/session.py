"""一个账户：一只股票、一个 run、一个入场插件。

空会话只把评估窗口标成现金。`run_account` 按开盘、盘中、收盘的顺序处理
止损、退出、减仓和入场。跟踪止损在收盘后才写下，下一根才生效。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import pandas as pd

from analysis.backtest.engine import EngineResult
from analysis.backtest.trend.config import StrategySpec
from analysis.backtest.trend.entries.base import EntrySignal
from analysis.backtest.trend.exits import exhaustion_due, long_trend_failed, next_long_stop, reduce_shares
from analysis.backtest.trend.fills import (
    apply_buy,
    apply_sell,
    apply_split,
    credit_dividend,
    entry_cancel_reason,
    raw_stop,
    slipped_buy,
    slipped_sell,
)
from analysis.backtest.trend.risk import format_percent, size_position

_COLUMNS = [
    "datetime",
    "equity",
    "equity_gross",
    "cash",
    "shares",
    "price",
    "position",
    "turnover",
    "leverage",
    "fee",
]


def run_empty_session(bars: pd.DataFrame, capital: float) -> EngineResult:
    """评估窗口内保持空仓。暖机不在这里出现，也不留下待执行单。"""
    if capital <= 0:
        raise ValueError("capital 必须 > 0")
    if bars is None or bars.empty:
        return EngineResult(
            equity=pd.DataFrame(columns=_COLUMNS),
            final_cash=float(capital),
        )

    rows = []
    last_price = 0.0
    for _, bar in bars.iterrows():
        price = float(bar["close"])
        last_price = price
        rows.append({
            "datetime": bar["datetime"],
            "equity": float(capital),
            "equity_gross": float(capital),
            "cash": float(capital),
            "shares": 0,
            "price": price,
            "position": 0,
            "turnover": 0.0,
            "leverage": 0.0,
            "fee": 0.0,
        })
    return EngineResult(
        equity=pd.DataFrame(rows),
        final_cash=float(capital),
        final_shares=0,
        final_price=last_price,
        cumulative_fees=0.0,
    )


@dataclass(frozen=True)
class ReplayBar:
    """手写或由磁带拼出来的一根 K 线。价格是不复权价，因子把它们换回复权价。

    `ema` 在对数轴上是几何价格，和阶段方法的快均线同一套坐标。
    `phase_start_index`、`direction_start_index` 是这段 bars 里的序号。
    结构间断和方向段用来清掉回调记忆。
    """

    time: str
    open: float
    high: float
    low: float
    close: float
    structure: Optional[str]
    category: Optional[str]
    a_t: Optional[float]
    factor: float = 1.0
    ema: Optional[float] = None
    dividend: float = 0.0
    split: float = 1.0
    gap_before: bool = False
    phase_start_index: Optional[int] = None
    direction_start_index: Optional[int] = None


@dataclass(frozen=True)
class Decision:
    index: int
    time: str
    action: str
    reason: str
    shares_before: int
    shares_after: int
    price: Optional[float]
    adjusted_stop: Optional[float]
    raw_stop: Optional[float]
    planned_risk: Optional[float]
    actual_pnl: Optional[float]
    actual_loss_fraction: Optional[float]
    one_lot: bool
    text: str


@dataclass(frozen=True)
class EquityRow:
    index: int
    time: str
    equity: float
    cash: float
    shares: int
    close: float
    effective_stop: Optional[float]
    next_stop: Optional[float]
    raw_stop: Optional[float]


@dataclass(frozen=True)
class AccountResult:
    decisions: Tuple[Decision, ...]
    equity: Tuple[EquityRow, ...]
    final_cash: float
    final_shares: int


@dataclass
class _PendingEntry:
    adjusted_stop: float
    adjusted_close: float
    a_t: float
    text: str
    signal_index: int


@dataclass
class _Position:
    shares: int
    adjusted_stop: float
    planned_risk: float
    entry_equity: float
    cost_basis: float
    reduced: bool
    hh: float
    one_lot: bool


_STRUCTURE_LABEL = {
    "downtrend": "下降",
    "range": "震荡",
    "undecided": "未定",
}


def run_account(
    bars: Sequence[ReplayBar],
    tactic,
    strategy: StrategySpec,
    market,
    *,
    lot_size: int,
    capital: float,
    price_axis: str = "log",
    withholding: float = 0.0,
) -> AccountResult:
    """按一根 K 线的开盘、盘中、收盘顺序回放。插件只在空仓且没有待执行入场时被询问。"""
    if capital <= 0:
        raise ValueError("capital 必须 > 0")
    if isinstance(lot_size, bool) or not isinstance(lot_size, int) or lot_size < 1:
        raise ValueError("每手股数必须是正整数")
    if price_axis not in ("log", "uniform"):
        raise ValueError(f"价格轴必须是 log 或 uniform，收到 {price_axis}")
    if not 0 <= withholding <= 1:
        raise ValueError("预扣税率必须在 [0, 1]")
    replay = _Replay(
        bars=tuple(bars),
        tactic=tactic,
        strategy=strategy,
        market=market,
        lot_size=lot_size,
        capital=float(capital),
        price_axis=price_axis,
        withholding=float(withholding),
    )
    return replay.run()


class _Replay:
    def __init__(self, bars, tactic, strategy, market, lot_size, capital, price_axis, withholding):
        self.bars = bars
        self.tactic = tactic
        self.strategy = strategy
        self.market = market
        self.lot_size = lot_size
        self.price_axis = price_axis
        self.withholding = withholding
        self.slippage = float(strategy.order.slippage_rate)
        self.max_trades = int(strategy.order.max_trades_per_day)
        self.cash = float(capital)
        self.position: Optional[_Position] = None
        self.pending_entry: Optional[_PendingEntry] = None
        self.pending_exit: Optional[str] = None
        self.pending_exit_text = ""
        self.fills: dict = {}
        self.decisions: list = []
        self.equity: list = []

    def run(self) -> AccountResult:
        for index, bar in enumerate(self.bars):
            if not isinstance(bar, ReplayBar):
                raise TypeError("回放需要 ReplayBar")
            self._on_bar(index, bar)
        self._expire_entry()
        shares = 0 if self.position is None else self.position.shares
        return AccountResult(tuple(self.decisions), tuple(self.equity), self.cash, shares)

    def _on_bar(self, index: int, bar: ReplayBar) -> None:
        self._check_bar(bar)
        governed = None if self.position is None else self.position.adjusted_stop
        self._corporate(index, bar)
        self._at_open(index, bar)
        if self.position is not None and governed is None:
            governed = self.position.adjusted_stop
        self._intraday(index, bar, governed)
        next_stop = None
        if self.position is not None:
            self._schedule(bar)
            next_stop = self.position.adjusted_stop
        elif self.pending_entry is None:
            self._ask(index, bar)
        self._mark(index, bar, governed, next_stop)

    def _at_open(self, index: int, bar: ReplayBar) -> None:
        if self.position is not None:
            stop = raw_stop(self.position.adjusted_stop, bar.factor)
            if bar.open <= stop:
                self._stop_out(index, bar, bar.open, gap=True)
                return
            if self.pending_exit == "trend":
                self._trend_exit(index, bar)
                return
            if self.pending_exit == "reduce" and self._can_trade(bar.time):
                self._reduce(index, bar)
                self._mark_trade(bar.time)
                self.pending_exit = None
            return
        if self.pending_entry is not None:
            self._try_entry(index, bar)

    def _intraday(self, index: int, bar: ReplayBar, governed: Optional[float]) -> None:
        if self.position is None or governed is None:
            return
        stop = raw_stop(governed, bar.factor)
        if bar.open > stop and bar.low <= stop:
            self._stop_out(index, bar, stop, gap=False)

    def _try_entry(self, index: int, bar: ReplayBar) -> None:
        pending = self.pending_entry
        reason = entry_cancel_reason(
            raw_open=bar.open,
            factor=bar.factor,
            adjusted_stop=pending.adjusted_stop,
            adjusted_signal_close=pending.adjusted_close,
            a_t=pending.a_t,
            gap_cancel=self.strategy.order.gap_cancel,
            max_stop_distance=self.strategy.order.max_stop_distance,
            price_axis=self.price_axis,
        )
        if reason:
            self._cancel(index, bar, reason, pending.adjusted_stop, pending.text)
            self.pending_entry = None
            return
        if not self._can_trade(bar.time):
            self._cancel(index, bar, "当日成交次数", pending.adjusted_stop, pending.text)
            self.pending_entry = None
            return
        fill_price = slipped_buy(bar.open, self.slippage)
        stop = raw_stop(pending.adjusted_stop, bar.factor)
        equity = self.cash
        sized = size_position(
            equity=equity,
            cash=self.cash,
            open_price=fill_price,
            raw_stop=stop,
            lot_size=self.lot_size,
            risk_per_trade=self.strategy.risk.risk_per_trade,
            one_lot_max_risk=self.strategy.risk.one_lot_max_risk,
            max_gross_exposure=self.strategy.risk.max_gross_exposure,
            slippage_rate=self.slippage,
            market=self.market,
        )
        if sized.shares <= 0:
            self._cancel(index, bar, sized.reason, pending.adjusted_stop, sized.text)
            self.pending_entry = None
            return
        fill = apply_buy(self.cash, fill_price, sized.shares, self.market)
        self.cash = fill.cash_after
        self._mark_trade(bar.time)
        self.position = _Position(
            shares=sized.shares,
            adjusted_stop=pending.adjusted_stop,
            planned_risk=sized.planned_risk,
            entry_equity=equity,
            cost_basis=fill.notional + fill.fee,
            reduced=False,
            hh=bar.high * bar.factor,
            one_lot=sized.one_lot,
        )
        self.pending_entry = None
        entry_reason = "一手规则" if sized.one_lot else ""
        text = (
            f"入场：买入 {sized.shares} 股，成交价 {_num(fill.price)} {self._currency()}，"
            f"复权止损 {_num(pending.adjusted_stop)}，不复权止损 {_num(stop)}。"
            f"{sized.text}{pending.text}"
        )
        self._record(
            index, bar, action="入场", reason=entry_reason, shares_before=0, shares_after=sized.shares,
            price=fill.price, adjusted_stop=pending.adjusted_stop, stop=stop,
            planned_risk=sized.planned_risk, one_lot=sized.one_lot, text=text,
        )

    def _stop_out(self, index: int, bar: ReplayBar, raw_price: float, *, gap: bool) -> None:
        position = self.position
        stop = raw_stop(position.adjusted_stop, bar.factor)
        fill_price = slipped_sell(raw_price, self.slippage)
        if gap:
            reason = "跳空穿过止损"
            lead = (
                f"止损：开盘价 {_num(bar.open)} 不高于不复权止损 {_num(stop)}，"
                f"按开盘价 {_num(fill_price)} 成交。"
            )
        else:
            reason = "盘中触及止损"
            lead = f"止损：最低价触及不复权止损 {_num(stop)}，按止损价 {_num(fill_price)} 成交。"
        self._sell(index, bar, position.shares, fill_price, action="止损", reason=reason, lead=lead, already_slipped=True)

    def _trend_exit(self, index: int, bar: ReplayBar) -> None:
        price = slipped_sell(bar.open, self.slippage)
        self._sell(
            index, bar, self.position.shares, price,
            action="趋势退出", reason="趋势失效", lead=self.pending_exit_text, already_slipped=True,
        )

    def _reduce(self, index: int, bar: ReplayBar) -> None:
        qty, flatten = reduce_shares(
            self.position.shares,
            self.strategy.exit.reduce_fraction,
            self.lot_size,
        )
        price = slipped_sell(bar.open, self.slippage)
        if flatten:
            lead = "衰竭减仓：取整后不足一手，改为全部退出。"
            reason = "取整全平"
        else:
            lead = "衰竭减仓：卖出部分仓位，剩余沿用同一条止损。"
            reason = "衰竭减仓"
        self._sell(index, bar, qty, price, action="减仓", reason=reason, lead=lead, already_slipped=True)

    def _sell(self, index, bar, qty, price, *, action, reason, lead, already_slipped) -> None:
        position = self.position
        if qty < 1 or qty > position.shares:
            raise ValueError("卖出股数必须在持仓之内")
        fill_price = price if already_slipped else slipped_sell(price, self.slippage)
        fill = apply_sell(self.cash, fill_price, qty, self.market)
        basis = position.cost_basis * qty / position.shares
        pnl = (fill.notional - fill.fee) - basis
        loss_fraction = 0.0
        if pnl < 0 and position.entry_equity > 0:
            loss_fraction = -pnl / position.entry_equity
        before = position.shares
        self.cash = fill.cash_after
        left = before - qty
        text = (
            f"{lead}卖出 {qty} 股，成交价 {_num(fill.price)} {self._currency()}，"
            f"股数由 {before} 变为 {left}。"
            f"计划风险 {format_percent(position.planned_risk)}，"
            f"实际盈亏 {_num(pnl)} {self._currency()}，"
            f"实际亏损 {format_percent(loss_fraction)}。"
        )
        self._record(
            index, bar, action=action, reason=reason, shares_before=before, shares_after=left,
            price=fill.price, adjusted_stop=position.adjusted_stop,
            stop=raw_stop(position.adjusted_stop, bar.factor),
            planned_risk=position.planned_risk, actual_pnl=pnl, actual_loss_fraction=loss_fraction,
            one_lot=position.one_lot, text=text,
        )
        if left == 0:
            self.position = None
            self.pending_exit = None
            return
        position.shares = left
        position.cost_basis -= basis

    def _schedule(self, bar: ReplayBar) -> None:
        position = self.position
        position.hh = max(position.hh, bar.high * bar.factor)
        failed = long_trend_failed(bar.structure)
        reduce = exhaustion_due(bar.category, position.reduced, self.strategy.exit.exhaustion_action)
        if failed:
            if reduce:
                position.reduced = True
            label = _STRUCTURE_LABEL.get(bar.structure, "不可用")
            if reduce or self.pending_exit == "reduce":
                self.pending_exit_text = f"趋势变成{label}，同时出现衰竭，全部退出，不开反向仓。"
            else:
                self.pending_exit_text = f"趋势变成{label}，下一根全部退出，不开反向仓。"
            self.pending_exit = "trend"
        elif reduce:
            position.reduced = True
            self.pending_exit = "reduce"
            self.pending_exit_text = "本次持仓第一次出现衰竭，下一根减仓。"
        position.adjusted_stop = next_long_stop(
            position.adjusted_stop,
            position.hh,
            bar.a_t,
            buffer=self.strategy.exit.trail_buffer,
            price_axis=self.price_axis,
            category=bar.category,
            trail_states=self.strategy.exit.trail_states,
        )

    def _ask(self, index: int, bar: ReplayBar) -> None:
        signal = self.tactic.propose(tuple(self.bars[: index + 1]))
        if signal is None:
            return
        if not isinstance(signal, EntrySignal):
            raise TypeError("插件必须返回 EntrySignal 或 None")
        stop = signal.adjusted_stop
        if isinstance(stop, bool) or not isinstance(stop, (int, float)) or not math.isfinite(float(stop)) or float(stop) <= 0:
            self._cancel(index, bar, "止损无效", None, signal.text or "")
            return
        if bar.a_t is None:
            self._cancel(index, bar, "波动单位尚未就绪", None, signal.text or "")
            return
        self.pending_entry = _PendingEntry(
            adjusted_stop=float(stop),
            adjusted_close=bar.close * bar.factor,
            a_t=float(bar.a_t),
            text=signal.text or "入场信号",
            signal_index=index,
        )

    def _corporate(self, index: int, bar: ReplayBar) -> None:
        position = self.position
        if position is None:
            return
        if bar.dividend > 0:
            before = self.cash
            self.cash, credit = credit_dividend(self.cash, position.shares, bar.dividend, self.withholding)
            text = (
                f"派息入账：每股股息 {_num(bar.dividend)} {self._currency()}，"
                f"预扣税 {_num(credit.tax)} {self._currency()}，"
                f"入账现金 {_num(credit.cash)} {self._currency()}，"
                f"现金由 {_num(before)} 变为 {_num(self.cash)} {self._currency()}，"
                f"股数 {position.shares}，复权止损 {_num(position.adjusted_stop)}。"
            )
            self._record(
                index, bar, action="派息入账", reason="", shares_before=position.shares,
                shares_after=position.shares, adjusted_stop=position.adjusted_stop,
                stop=raw_stop(position.adjusted_stop, bar.factor), text=text,
            )
        if bar.split != 1:
            before = position.shares
            position.shares = apply_split(position.shares, bar.split)
            text = (
                f"拆股：1 股变为 {_num(bar.split)} 股，股数由 {before} 变为 {position.shares}，"
                f"现金不变仍为 {_num(self.cash)} {self._currency()}，"
                f"复权止损 {_num(position.adjusted_stop)}。"
            )
            self._record(
                index, bar, action="拆股", reason="", shares_before=before, shares_after=position.shares,
                adjusted_stop=position.adjusted_stop, stop=raw_stop(position.adjusted_stop, bar.factor),
                text=text,
            )

    def _expire_entry(self) -> None:
        pending = self.pending_entry
        if pending is None or not self.bars:
            return
        bar = self.bars[pending.signal_index]
        self._cancel(pending.signal_index, bar, "没有下一根", pending.adjusted_stop, pending.text)
        self.pending_entry = None

    def _cancel(self, index, bar, reason, adjusted_stop, detail) -> None:
        stop = None if adjusted_stop is None else raw_stop(adjusted_stop, bar.factor)
        text = f"取消：{reason}。{detail}"
        self._record(
            index, bar, action="取消", reason=reason, shares_before=0, shares_after=0,
            price=bar.open, adjusted_stop=adjusted_stop, stop=stop, text=text,
        )

    def _record(
        self, index, bar, *, action, reason, shares_before, shares_after, text,
        price=None, adjusted_stop=None, stop=None, planned_risk=None, actual_pnl=None,
        actual_loss_fraction=None, one_lot=False,
    ) -> None:
        self.decisions.append(Decision(
            index=index,
            time=bar.time,
            action=action,
            reason=reason,
            shares_before=shares_before,
            shares_after=shares_after,
            price=price,
            adjusted_stop=adjusted_stop,
            raw_stop=stop,
            planned_risk=planned_risk,
            actual_pnl=actual_pnl,
            actual_loss_fraction=actual_loss_fraction,
            one_lot=one_lot,
            text=text,
        ))

    def _mark(self, index, bar, governed, next_stop) -> None:
        shares = 0 if self.position is None else self.position.shares
        self.equity.append(EquityRow(
            index=index,
            time=bar.time,
            equity=self.cash + shares * bar.close,
            cash=self.cash,
            shares=shares,
            close=bar.close,
            effective_stop=governed,
            next_stop=next_stop,
            raw_stop=None if governed is None else governed / bar.factor,
        ))

    def _can_trade(self, when: str) -> bool:
        return self.fills.get(when, 0) < self.max_trades

    def _mark_trade(self, when: str) -> None:
        self.fills[when] = self.fills.get(when, 0) + 1

    def _currency(self) -> str:
        return getattr(self.market, "currency", "") or ""

    def _check_bar(self, bar: ReplayBar) -> None:
        for name, value in ("open", bar.open), ("high", bar.high), ("low", bar.low), ("close", bar.close):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{bar.time} 的{name}必须为正")
        if bar.high < bar.low or bar.high < max(bar.open, bar.close) or bar.low > min(bar.open, bar.close):
            raise ValueError(f"{bar.time} 的最高价和最低价包不住开盘价或收盘价")
        if bar.factor <= 0:
            raise ValueError(f"{bar.time} 的复权因子必须为正")
        if bar.dividend < 0 or bar.split <= 0:
            raise ValueError(f"{bar.time} 的除权数据无效")
        if bar.a_t is not None and bar.a_t < 0:
            raise ValueError(f"{bar.time} 的 a_t 不能为负")


def _num(value: float) -> str:
    text = f"{float(value):.8f}".rstrip("0").rstrip(".")
    return text or "0"
