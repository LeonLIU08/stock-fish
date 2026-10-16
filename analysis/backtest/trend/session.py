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
from analysis.backtest.trend.entries.base import EntrySignal, offset_price, scale_label
from analysis.backtest.trend.exits import (
    drawdown_stop,
    exhaustion_due,
    long_trend_failed,
    next_long_stop,
    reduce_shares,
)
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
from analysis.backtest.trend.prices import upward_gap_cancel
from analysis.backtest.trend.risk import add_risk_budget, format_percent, size_exposure, size_position
from analysis.structure.methods.phase import (
    DOWNTREND,
    INSUFFICIENT,
    UP_CONTINUE,
    UP_PULLBACK,
    _NAMES as PHASE_NAMES,
)

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
    exit_scale: Optional[str] = None
    exit_structure: Optional[str] = None
    exit_direction_start_index: Optional[int] = None
    up_breaks: Tuple[str, ...] = ()
    down_breaks: Tuple[str, ...] = ()
    support_valid: bool = False
    phase_trigger: Optional[str] = None
    range_floor: Optional[float] = None


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
    structure: Optional[str] = None
    category: Optional[str] = None
    a_t: Optional[float] = None
    ema: Optional[float] = None
    factor: float = 1.0
    fee: Optional[float] = None
    cash_after: Optional[float] = None
    equity_after: Optional[float] = None
    position_fraction: Optional[float] = None
    exit_rule: str = ""
    reference_equity: Optional[float] = None


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
    structure: Optional[str] = None
    category: Optional[str] = None
    position_fraction: float = 0.0
    planned_risk: Optional[float] = None
    factor: float = 1.0
    dividend_cash: float = 0.0
    equity_gross: Optional[float] = None
    fee: float = 0.0


@dataclass(frozen=True)
class AccountResult:
    decisions: Tuple[Decision, ...]
    equity: Tuple[EquityRow, ...]
    final_cash: float
    final_shares: int
    blocked: Tuple[Tuple[str, int], ...] = ()


@dataclass
class _PendingEntry:
    adjusted_stop: Optional[float]
    adjusted_close: float
    a_t: float
    text: str
    signal_index: int
    kind: str = "starter"
    risk: Optional[float] = None


@dataclass
class _Tranche:
    role: str
    shares: int
    adjusted_fill: float
    cash_basis: float
    planned_risk: float


@dataclass
class _Position:
    shares: int
    adjusted_stop: Optional[float]
    planned_risk: float
    entry_equity: float
    cost_basis: float
    reduced: bool
    hh: float
    one_lot: bool
    initial_stop: Optional[float]


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
    trade_from: int = 0,
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
        trade_from=trade_from,
    )
    return replay.run()


class _Replay:
    def __init__(self, bars, tactic, strategy, market, lot_size, capital, price_axis, withholding, trade_from=0):
        self.bars = bars
        self.tactic = tactic
        self.strategy = strategy
        self.market = market
        self.lot_size = lot_size
        self.price_axis = price_axis
        self.withholding = withholding
        if isinstance(trade_from, bool) or not isinstance(trade_from, int) or trade_from < 0:
            raise ValueError("trade_from 必须是非负整数")
        self.trade_from = trade_from
        self.slippage = float(strategy.order.slippage_rate)
        self.max_trades = int(strategy.order.max_trades_per_day)
        self.cash = float(capital)
        self.position: Optional[_Position] = None
        self.pending_entry: Optional[_PendingEntry] = None
        self.pending_add: Optional[_PendingEntry] = None
        self.pending_exit: Optional[str] = None
        self.pending_exit_text = ""
        self.pending_exit_rule = ""
        self.tranches: list = []
        self.add_count = 0
        self.exhaustion_latched = False
        self.exhaustion_peeled = False
        self.consumed_breaks: set = set()
        self.prev_category: Optional[str] = None
        self.fills: dict = {}
        self.decisions: list = []
        self.equity: list = []
        self.blocked: dict = {}
        self.cumulative_fees = 0.0
        self.bar_fee = 0.0
        self.bar_dividend = 0.0

    def run(self) -> AccountResult:
        for index, bar in enumerate(self.bars):
            if not isinstance(bar, ReplayBar):
                raise TypeError("回放需要 ReplayBar")
            self._on_bar(index, bar)
        self._expire_entry()
        shares = 0 if self.position is None else self.position.shares
        return AccountResult(
            tuple(self.decisions),
            tuple(self.equity),
            self.cash,
            shares,
            blocked=tuple(sorted(self.blocked.items())),
        )

    def _on_bar(self, index: int, bar: ReplayBar) -> None:
        self._check_bar(bar)
        self.bar_fee = 0.0
        self.bar_dividend = 0.0
        if index < self.trade_from:
            self._remember_phase(bar)
            return
        governed = None if self.position is None else self.position.adjusted_stop
        self._corporate(index, bar)
        self._at_open(index, bar)
        if self.position is not None and governed is None:
            governed = self.position.adjusted_stop
        self._intraday(index, bar, governed)
        next_stop = None
        if self.position is not None:
            self._schedule(index, bar)
            next_stop = self.position.adjusted_stop
        elif self.pending_entry is None:
            self._ask(index, bar)
        self._mark(index, bar, governed, next_stop)
        self._remember_phase(bar)

    def _remember_phase(self, bar: ReplayBar) -> None:
        if self._scale_book():
            self.prev_category = bar.category

    def _scale_book(self) -> bool:
        return getattr(self.tactic, "exit_kind", "") in ("scale_book", "break_book")

    def _break_book(self) -> bool:
        return getattr(self.tactic, "exit_kind", "") == "break_book"

    def _at_open(self, index: int, bar: ReplayBar) -> None:
        if self._scale_book():
            self._scale_open(index, bar)
            return
        if self.position is not None:
            if self.position.adjusted_stop is not None:
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
        if pending.adjusted_stop is None:
            self._try_trend_entry(index, bar)
            return
        reason = entry_cancel_reason(
            raw_open=bar.open,
            factor=bar.factor,
            adjusted_stop=pending.adjusted_stop,
            adjusted_signal_close=pending.adjusted_close,
            a_t=pending.a_t,
            gap_cancel=self.strategy.order.gap_cancel,
            max_stop_distance=self.strategy.order.max_stop_distance,
            min_stop_distance=self.strategy.order.min_stop_distance,
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
            risk_per_trade=self.strategy.risk.risk_per_trade if pending.risk is None else pending.risk,
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
            initial_stop=pending.adjusted_stop,
        )
        self.cumulative_fees += fill.fee
        self.bar_fee += fill.fee
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
            fee=fill.fee,             reference_equity=equity,
        )
        if self._scale_book():
            self._reset_book()
            self.tranches = [_Tranche(
                "starter",
                sized.shares,
                fill.price * bar.factor,
                fill.notional + fill.fee,
                sized.planned_risk,
            )]

    def _try_trend_entry(self, index: int, bar: ReplayBar) -> None:
        pending = self.pending_entry
        if upward_gap_cancel(
            bar.open,
            bar.factor,
            pending.adjusted_close,
            pending.a_t,
            self.strategy.order.gap_cancel,
            price_axis=self.price_axis,
        ):
            self._cancel(index, bar, "跳空", None, pending.text)
            self.pending_entry = None
            return
        if not self._can_trade(bar.time):
            self._cancel(index, bar, "当日成交次数", None, pending.text)
            self.pending_entry = None
            return
        fill_price = slipped_buy(bar.open, self.slippage)
        equity = self.cash
        sized = size_exposure(
            equity=equity,
            cash=self.cash,
            open_price=fill_price,
            lot_size=self.lot_size,
            max_gross_exposure=self.strategy.risk.max_gross_exposure,
            market=self.market,
        )
        if sized.shares <= 0:
            self._cancel(index, bar, sized.reason, None, sized.text)
            self.pending_entry = None
            return
        fill = apply_buy(self.cash, fill_price, sized.shares, self.market)
        self.cash = fill.cash_after
        self._mark_trade(bar.time)
        drawdown = self.strategy.exit.drawdown
        adjusted_fill = fill_price * bar.factor
        if drawdown is None:
            stop = None
            hh = bar.high * bar.factor
            entry_text = (
                f"入场：买入 {sized.shares} 股，成交价 {_num(fill.price)} {self._currency()}。"
                f"没有价格止损。{sized.text}{pending.text}"
            )
        else:
            stop = drawdown_stop(adjusted_fill, drawdown)
            hh = adjusted_fill
            entry_text = (
                f"入场：买入 {sized.shares} 股，成交价 {_num(fill.price)} {self._currency()}。"
                f"前高先记成交价，回撤 {format_percent(drawdown)} 止损。"
                f"复权止损 {_num(stop)}，不复权止损 {_num(stop / bar.factor)}。"
                f"{sized.text}{pending.text}"
            )
        self.position = _Position(
            shares=sized.shares,
            adjusted_stop=stop,
            planned_risk=sized.planned_risk,
            entry_equity=equity,
            cost_basis=fill.notional + fill.fee,
            reduced=False,
            hh=hh,
            one_lot=False,
            initial_stop=stop,
        )
        self.cumulative_fees += fill.fee
        self.bar_fee += fill.fee
        self.pending_entry = None
        self._record(
            index, bar, action="入场", reason="", shares_before=0, shares_after=sized.shares,
            price=fill.price, adjusted_stop=stop,
            stop=None if stop is None else stop / bar.factor,
            planned_risk=sized.planned_risk, one_lot=False, text=entry_text,
            fee=fill.fee, reference_equity=equity,
        )

    def _stop_out(self, index: int, bar: ReplayBar, raw_price: float, *, gap: bool) -> None:
        position = self.position
        stop = raw_stop(position.adjusted_stop, bar.factor)
        fill_price = slipped_sell(raw_price, self.slippage)
        if gap:
            reason = "跳空穿过止损"
            rule = self._stop_rule()
            lead = (
                f"止损：开盘价 {_num(bar.open)} 不高于不复权止损 {_num(stop)}，"
                f"按开盘价 {_num(fill_price)} 成交。"
            )
        else:
            reason = "盘中触及止损"
            rule = self._stop_rule()
            lead = f"止损：最低价触及不复权止损 {_num(stop)}，按止损价 {_num(fill_price)} 成交。"
        self._sell(
            index, bar, position.shares, fill_price, action="止损", reason=reason,
            lead=lead, already_slipped=True, exit_rule=rule,
        )

    def _trend_exit(self, index: int, bar: ReplayBar) -> None:
        price = slipped_sell(bar.open, self.slippage)
        rule = self.pending_exit_rule or "趋势失效"
        self._sell(
            index, bar, self.position.shares, price,
            action="趋势退出", reason=rule, lead=self.pending_exit_text, already_slipped=True,
            exit_rule=rule,
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
            rule = "衰竭取整后全平"
        else:
            lead = "衰竭减仓：卖出部分仓位，剩余沿用同一条止损。"
            reason = "衰竭减仓"
            rule = "衰竭减仓"
        self._sell(
            index, bar, qty, price, action="减仓", reason=reason, lead=lead,
            already_slipped=True, exit_rule=rule,
        )

    def _sell(self, index, bar, qty, price, *, action, reason, lead, already_slipped, exit_rule="", basis_cash=None) -> None:
        position = self.position
        if qty < 1 or qty > position.shares:
            raise ValueError("卖出股数必须在持仓之内")
        fill_price = price if already_slipped else slipped_sell(price, self.slippage)
        fill = apply_sell(self.cash, fill_price, qty, self.market)
        self.cumulative_fees += fill.fee
        self.bar_fee += fill.fee
        if basis_cash is None:
            basis = position.cost_basis * qty / position.shares
        else:
            basis = float(basis_cash)
        pnl = (fill.notional - fill.fee) - basis
        loss_fraction = 0.0
        if pnl < 0 and position.entry_equity > 0:
            loss_fraction = -pnl / position.entry_equity
        before = position.shares
        self.cash = fill.cash_after
        left = before - qty
        if position.adjusted_stop is None:
            risk_line = f"入场名义仓位 {format_percent(position.planned_risk)}，"
        else:
            risk_line = f"计划风险 {format_percent(position.planned_risk)}，"
        text = (
            f"{lead}卖出 {qty} 股，成交价 {_num(fill.price)} {self._currency()}，"
            f"股数由 {before} 变为 {left}。"
            f"{risk_line}"
            f"实际盈亏 {_num(pnl)} {self._currency()}，"
            f"实际亏损 {format_percent(loss_fraction)}。"
        )
        self._record(
            index, bar, action=action, reason=reason, shares_before=before, shares_after=left,
            price=fill.price, adjusted_stop=position.adjusted_stop,
            stop=None if position.adjusted_stop is None else raw_stop(position.adjusted_stop, bar.factor),
            planned_risk=position.planned_risk, actual_pnl=pnl, actual_loss_fraction=loss_fraction,
            one_lot=position.one_lot, text=text, fee=fill.fee, exit_rule=exit_rule,
            reference_equity=position.entry_equity,
        )
        if left == 0:
            self.position = None
            self.pending_exit = None
            self.pending_exit_rule = ""
            self.pending_add = None
            self._reset_book()
            return
        position.shares = left
        position.cost_basis -= basis

    def _schedule(self, index: int, bar: ReplayBar) -> None:
        if self._scale_book():
            self._schedule_scale(index, bar)
            return
        if getattr(self.tactic, "exit_kind", "") == "scale_downtrend":
            self._raise_drawdown_stop(bar)
            self._schedule_scale_exit(index, bar)
            return
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
            self.pending_exit_rule = "趋势失效"
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

    def _schedule_scale_exit(self, index: int, bar: ReplayBar) -> None:
        """下降趋势在当根确认，下一根开盘退出。不抬止损，不减仓。

        策略 C 看卖出尺度。策略 D0 的买卖是同一个尺度，看这根本身的方向。
        """
        if getattr(self.tactic, "sell_on_entry_scale", False):
            structure = bar.structure
            start = bar.direction_start_index
            name = scale_label(getattr(self.tactic, "scale", None))
        else:
            structure = bar.exit_structure
            start = bar.exit_direction_start_index
            name = scale_label(bar.exit_scale)
        if structure != DOWNTREND or start != index:
            return
        self.pending_exit = "trend"
        self.pending_exit_rule = f"{name}下降"
        self.pending_exit_text = f"{name}下降趋势在当根确认，下一根开盘全部退出，不开反向仓。"

    def _raise_drawdown_stop(self, bar: ReplayBar) -> None:
        """收盘后用当根最高价抬前高。新止损只升不降，下一根才生效。"""
        fraction = self.strategy.exit.drawdown
        position = self.position
        if fraction is None or position is None:
            return
        high = bar.high * bar.factor
        if high > position.hh:
            position.hh = high
        candidate = drawdown_stop(position.hh, fraction)
        if position.adjusted_stop is None or candidate > position.adjusted_stop:
            position.adjusted_stop = candidate

    def _scale_open(self, index: int, bar: ReplayBar) -> None:
        if self.position is not None:
            if self.position.adjusted_stop is not None:
                stop = raw_stop(self.position.adjusted_stop, bar.factor)
                if bar.open <= stop:
                    self.pending_add = None
                    self._stop_out(index, bar, bar.open, gap=True)
                    return
            if self.pending_exit == "flatten":
                self.pending_add = None
                self._trend_exit(index, bar)
                return
            if self.pending_exit == "peel" and self._can_trade(bar.time):
                if self.pending_exit_rule == "上涨衰竭":
                    self.exhaustion_peeled = True
                self._peel_oldest(index, bar)
                self._mark_trade(bar.time)
                self.pending_exit = None
                self.pending_exit_rule = ""
                self.pending_exit_text = ""
            if self.position is not None and self.pending_add is not None:
                self._try_scale_add(index, bar)
            return
        if self.pending_entry is not None:
            self._try_entry(index, bar)

    def _schedule_scale(self, index: int, bar: ReplayBar) -> None:
        position = self.position
        if position is None:
            return
        high = bar.high * bar.factor
        if high > position.hh:
            position.hh = high
        if bar.category in self.strategy.exit.trail_states and position.adjusted_stop is not None:
            position.adjusted_stop = next_long_stop(
                position.adjusted_stop,
                position.hh,
                bar.a_t,
                buffer=self.strategy.exit.trail_buffer,
                price_axis=self.price_axis,
                category=bar.category,
                trail_states=self.strategy.exit.trail_states,
            )
        if self._should_flatten(bar):
            label = PHASE_NAMES.get(bar.category, "不可用" if bar.category is None else str(bar.category))
            self.pending_exit = "flatten"
            self.pending_exit_rule = label
            self.pending_exit_text = f"主状态是{label}，下一根开盘全部退出，不开反向仓。"
            self.pending_add = None
            return
        peel = False
        structural = self._sell_weight(index) if self._break_book() else 0
        if bar.category == self.strategy.exit.peel_on:
            if not self.exhaustion_latched:
                self.exhaustion_latched = True
                if structural < 2:
                    peel = True
                    self.pending_exit = "peel"
                    self.pending_exit_rule = "上涨衰竭"
                    self.pending_exit_text = "上涨衰竭预警第一次出现，下一根减掉最早的一笔。"
        else:
            self.exhaustion_latched = False
        if structural >= 3 or (structural == 2 and self.exhaustion_peeled):
            self.pending_exit = "flatten"
            self.pending_exit_rule = "结构向下突破"
            self.pending_exit_text = "向下突破达到清仓权重，下一根开盘全部退出，不开反向仓。"
            self.pending_add = None
            self._consume(index, "down")
            return
        if structural == 2:
            self.pending_exit = "peel"
            self.pending_exit_rule = "结构向下突破"
            self.pending_exit_text = "短尺度向下突破，下一根减掉最早的一笔。"
            self.pending_add = None
            self._consume(index, "down")
            return
        if not peel and self._starter_banked(bar):
            self.pending_exit = "peel"
            self.pending_exit_rule = "试仓止盈"
            self.pending_exit_text = "试仓上涨达到门槛，下一根只卖掉试仓。"
        self._schedule_scale_add(index, bar)

    def _active(self, index: int, side: str) -> Tuple[str, ...]:
        spec = self.strategy.breakout
        if spec is None:
            return ()
        start = max(0, index - int(spec.fresh_bars) + 1)
        found = []
        for cursor in range(start, index + 1):
            tokens = self.bars[cursor].up_breaks if side == "up" else self.bars[cursor].down_breaks
            for token in tokens:
                if token in self.consumed_breaks or token in found:
                    continue
                if side == "up" and self._down_since(cursor, index):
                    continue
                found.append(token)
        return tuple(found)

    def _down_since(self, born: int, index: int) -> bool:
        for cursor in range(born, index + 1):
            if self.bars[cursor].down_breaks:
                return True
        return False

    def _sell_weight(self, index: int) -> int:
        spec = self.strategy.breakout
        entry = self.strategy.entry
        if spec is None:
            return 0
        weight = 0
        context = getattr(entry, "context_scale", None)
        for token in self._active(index, "down"):
            scale = token.split("|", 1)[0]
            weight = max(weight, spec.mid_down if scale == context else spec.short_down)
        return weight

    def _position_risk(self, index: int) -> float:
        spec = self.strategy.breakout
        base = self.strategy.risk.risk_per_trade
        if spec is None:
            return base
        extra = (1 if self._active(index, "up") else 0) + (1 if self.bars[index].support_valid else 0)
        return min(base + spec.step * extra, self.strategy.risk.max_open_risk)

    def _consume(self, index: int, side: str, kind: Optional[str] = None) -> None:
        for token in self._active(index, side):
            if kind is not None and token.split("|")[1] != kind:
                continue
            self.consumed_breaks.add(token)

    def _schedule_range_entry(self, index: int, bar: ReplayBar) -> None:
        spec = self.strategy.breakout
        if spec is None or bar.phase_trigger != "range_break_up" or bar.phase_start_index != index:
            return
        if bar.a_t is None:
            return
        close = bar.close * bar.factor
        stop = offset_price(close, bar.a_t, -self.strategy.exit.trail_buffer, self.price_axis)
        floor = bar.range_floor
        if floor is not None and 0 < floor < close and floor > stop:
            stop = floor
        if not math.isfinite(stop) or stop <= 0:
            return
        text = (
            "策略 F：整理区间向上突破，方向尚未确认，下一根开盘买入小仓。"
            f"复权止损 {_num(stop)}。"
        )
        self.pending_entry = _PendingEntry(
            adjusted_stop=float(stop),
            adjusted_close=close,
            a_t=float(bar.a_t),
            text=text,
            signal_index=index,
            risk=spec.range_break_risk,
        )
        self._consume(index, "up", kind="phase")

    def _should_flatten(self, bar: ReplayBar) -> bool:
        if bar.category in self.strategy.exit.flatten_categories:
            return True
        return bar.category == INSUFFICIENT and bar.structure is None

    def _starter_banked(self, bar: ReplayBar) -> bool:
        starter = self._tranche("starter")
        if starter is None or bar.a_t is None or self.strategy.exit.bank_multiple <= 0:
            return False
        close = bar.close * bar.factor
        target = offset_price(starter.adjusted_fill, bar.a_t, self.strategy.exit.bank_multiple, self.price_axis)
        return close >= target

    def _schedule_scale_add(self, index: int, bar: ReplayBar) -> None:
        if self._break_book() and self._active(index, "down"):
            self.pending_add = None
            return
        if self.pending_exit == "flatten" or bar.a_t is None or self.position is None:
            return
        if self.add_count >= self.strategy.exit.max_adds:
            self.pending_add = None
            return
        stop = self.position.adjusted_stop
        if stop is None:
            return
        close = bar.close * bar.factor
        if self.add_count == 0:
            starter = self._tranche("starter")
            if starter is None or stop < starter.adjusted_fill:
                return
            if bar.category != UP_CONTINUE or self.prev_category != UP_PULLBACK:
                return
            kind = "add1"
            text = "加仓一：上涨回调结束回到上涨延续，止损已不低于试仓成交价。"
        else:
            first = self._tranche("add1")
            wanted = self.strategy.exit.second_category or UP_CONTINUE
            if first is None or bar.category != wanted:
                return
            target = offset_price(first.adjusted_fill, bar.a_t, self.strategy.exit.extend_multiple, self.price_axis)
            if close < target or stop < self._average_fill():
                return
            kind = "add2"
            text = "加仓二：上涨延续里高出加仓一，止损已不低于持仓成本。"
        risk = self._position_risk(index) if self._break_book() else None
        self.pending_add = _PendingEntry(stop, close, float(bar.a_t), text, index, kind, risk)
        if self._break_book():
            self._consume(index, "up")

    def _try_scale_add(self, index: int, bar: ReplayBar) -> None:
        pending = self.pending_add
        self.pending_add = None
        if pending is None or self.position is None or not self.tranches:
            return
        if pending.kind == "add1" and self._tranche("starter") is None:
            self._cancel(index, bar, "试仓已卖出", self.position.adjusted_stop, pending.text)
            return
        if pending.kind == "add2" and self._tranche("add1") is None:
            self._cancel(index, bar, "加仓一已卖出", self.position.adjusted_stop, pending.text)
            return
        stop = self.position.adjusted_stop
        reason = entry_cancel_reason(
            raw_open=bar.open,
            factor=bar.factor,
            adjusted_stop=stop,
            adjusted_signal_close=pending.adjusted_close,
            a_t=pending.a_t,
            gap_cancel=self.strategy.order.gap_cancel,
            max_stop_distance=self.strategy.order.max_stop_distance,
            min_stop_distance=self.strategy.order.min_stop_distance,
            price_axis=self.price_axis,
        )
        if reason:
            self._cancel(index, bar, reason, stop, pending.text)
            return
        if pending.kind == "add2" and bar.open * bar.factor <= self._average_fill():
            self._cancel(index, bar, "开盘价不高于成本", stop, pending.text)
            return
        if not self._can_trade(bar.time):
            self._cancel(index, bar, "当日成交次数", stop, pending.text)
            return
        fill_price = slipped_buy(bar.open, self.slippage)
        equity = self.cash + self.position.shares * bar.open
        budget = add_risk_budget(
            self._cash_open_risk(bar, equity),
            self.strategy.risk.max_open_risk,
            self.strategy.risk.risk_per_trade if pending.risk is None else pending.risk,
            self.strategy.exit.add_floor,
        )
        if budget is None:
            self._cancel(index, bar, "当前风险", stop, pending.text)
            return
        sized = size_position(
            equity=equity,
            cash=self.cash,
            open_price=fill_price,
            raw_stop=raw_stop(stop, bar.factor),
            lot_size=self.lot_size,
            risk_per_trade=budget,
            one_lot_max_risk=0.0,
            max_gross_exposure=self.strategy.risk.max_gross_exposure,
            slippage_rate=self.slippage,
            market=self.market,
            occupied_notional=self.position.shares * bar.open,
        )
        if sized.shares <= 0:
            self._cancel(index, bar, sized.reason or "风险预算", stop, sized.text + pending.text)
            return
        fill = apply_buy(self.cash, fill_price, sized.shares, self.market)
        before = self.position.shares
        self.cash = fill.cash_after
        self.position.shares = before + sized.shares
        self.position.cost_basis += fill.notional + fill.fee
        self.position.planned_risk += sized.planned_risk
        self.cumulative_fees += fill.fee
        self.bar_fee += fill.fee
        self._mark_trade(bar.time)
        role = "add1" if pending.kind == "add1" else "add2"
        self.tranches.append(_Tranche(
            role, sized.shares, fill.price * bar.factor, fill.notional + fill.fee, sized.planned_risk,
        ))
        self.add_count += 1
        text = (
            f"加仓：买入 {sized.shares} 股，成交价 {_num(fill.price)} {self._currency()}，"
            f"复权止损 {_num(stop)}。{sized.text}{pending.text}"
        )
        self._record(
            index, bar, action="加仓", reason="", shares_before=before,
            shares_after=self.position.shares, price=fill.price, adjusted_stop=stop,
            stop=raw_stop(stop, bar.factor), planned_risk=sized.planned_risk,
            text=text, fee=fill.fee, reference_equity=equity,
        )

    def _peel_oldest(self, index: int, bar: ReplayBar) -> None:
        if not self.tranches or self.position is None:
            return
        tranche = self.tranches[0]
        qty = (tranche.shares // self.lot_size) * self.lot_size
        left = self.position.shares - qty
        lead = self.pending_exit_text or "减掉最早的一笔。"
        rule = self.pending_exit_rule or "减仓"
        if qty <= 0 or (left > 0 and left < self.lot_size):
            self.pending_exit_text = lead + "取整后不足一手，改为全部退出。"
            self.pending_exit_rule = "取整全平"
            self._trend_exit(index, bar)
            return
        sold_basis = tranche.cash_basis * qty / tranche.shares
        sold_risk = tranche.planned_risk * qty / tranche.shares
        self._sell(
            index, bar, qty, slipped_sell(bar.open, self.slippage),
            action="减仓", reason=rule, lead=lead, already_slipped=True, exit_rule=rule,
            basis_cash=sold_basis,
        )
        if self.position is None:
            return
        tranche.shares -= qty
        tranche.cash_basis -= sold_basis
        tranche.planned_risk -= sold_risk
        self.position.planned_risk = max(0.0, self.position.planned_risk - sold_risk)
        if tranche.shares <= 0:
            self.tranches.pop(0)

    def _tranche(self, role: str) -> Optional[_Tranche]:
        for tranche in self.tranches:
            if tranche.role == role and tranche.shares > 0:
                return tranche
        return None

    def _cash_open_risk(self, bar: ReplayBar, equity: float) -> float:
        if equity <= 0 or self.position is None or self.position.adjusted_stop is None:
            return 0.0
        raw = raw_stop(self.position.adjusted_stop, bar.factor)
        total = 0.0
        for tranche in self.tranches:
            if tranche.shares <= 0:
                continue
            per_share = tranche.cash_basis / tranche.shares
            total += tranche.shares * max(per_share - raw, 0.0)
        return total / equity

    def _average_fill(self) -> float:
        shares = sum(tranche.shares for tranche in self.tranches)
        if shares <= 0:
            return 0.0
        return sum(tranche.shares * tranche.adjusted_fill for tranche in self.tranches) / shares

    def _reset_book(self) -> None:
        self.tranches = []
        self.add_count = 0
        self.exhaustion_latched = False
        self.exhaustion_peeled = False

    def _stop_rule(self) -> str:
        if self.strategy.exit.drawdown is not None:
            return "前高回撤"
        position = self.position
        if (
            position is not None
            and position.adjusted_stop is not None
            and position.initial_stop is not None
            and position.adjusted_stop > position.initial_stop + 1e-8
        ):
            return "跟踪止损"
        return "止损"

    def _ask(self, index: int, bar: ReplayBar) -> None:
        if self._break_book() and self._active(index, "down"):
            return
        prefix = tuple(self.bars[: index + 1])
        if hasattr(self.tactic, "skip_reason"):
            blocked = self.tactic.skip_reason(prefix)
        elif hasattr(self.tactic, "environment_block"):
            blocked = self.tactic.environment_block(bar)
        else:
            blocked = None
        signal = self.tactic.propose(prefix)
        if signal is None and blocked:
            self.blocked[blocked] = self.blocked.get(blocked, 0) + 1
        if signal is None:
            if self._break_book():
                self._schedule_range_entry(index, bar)
            return
        if not isinstance(signal, EntrySignal):
            raise TypeError("插件必须返回 EntrySignal 或 None")
        stop = signal.adjusted_stop
        if stop is None:
            if getattr(self.tactic, "exit_kind", "") != "scale_downtrend":
                self._cancel(index, bar, "止损无效", None, signal.text or "")
                return
        elif isinstance(stop, bool) or not isinstance(stop, (int, float)) or not math.isfinite(float(stop)) or float(stop) <= 0:
            self._cancel(index, bar, "止损无效", None, signal.text or "")
            return
        else:
            stop = float(stop)
        if bar.a_t is None:
            self._cancel(index, bar, "波动单位尚未就绪", None, signal.text or "")
            return
        risk = None
        if self._break_book():
            risk = self._position_risk(index)
            self._consume(index, "up")
        self.pending_entry = _PendingEntry(
            adjusted_stop=stop,
            adjusted_close=bar.close * bar.factor,
            a_t=float(bar.a_t),
            text=signal.text or "入场信号",
            signal_index=index,
            risk=risk,
        )

    def _corporate(self, index: int, bar: ReplayBar) -> None:
        position = self.position
        if position is None:
            return
        if bar.dividend > 0:
            before = self.cash
            self.cash, credit = credit_dividend(self.cash, position.shares, bar.dividend, self.withholding)
            self.bar_dividend += credit.cash
            stop_note = _stop_note(position.adjusted_stop)
            text = (
                f"派息入账：每股股息 {_num(bar.dividend)} {self._currency()}，"
                f"预扣税 {_num(credit.tax)} {self._currency()}，"
                f"入账现金 {_num(credit.cash)} {self._currency()}，"
                f"现金由 {_num(before)} 变为 {_num(self.cash)} {self._currency()}，"
                f"股数 {position.shares}。{stop_note}"
            )
            self._record(
                index, bar, action="派息入账", reason="", shares_before=position.shares,
                shares_after=position.shares, adjusted_stop=position.adjusted_stop,
                stop=None if position.adjusted_stop is None else raw_stop(position.adjusted_stop, bar.factor),
                text=text,
            )
        if bar.split != 1:
            before = position.shares
            position.shares = apply_split(position.shares, bar.split)
            if self.tranches:
                for tranche in self.tranches:
                    tranche.shares = apply_split(tranche.shares, bar.split)
                drift = position.shares - sum(tranche.shares for tranche in self.tranches)
                self.tranches[-1].shares += drift
            text = (
                f"拆股：1 股变为 {_num(bar.split)} 股，股数由 {before} 变为 {position.shares}，"
                f"现金不变仍为 {_num(self.cash)} {self._currency()}。"
                f"{_stop_note(position.adjusted_stop)}"
            )
            self._record(
                index, bar, action="拆股", reason="", shares_before=before, shares_after=position.shares,
                adjusted_stop=position.adjusted_stop,
                stop=None if position.adjusted_stop is None else raw_stop(position.adjusted_stop, bar.factor),
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
        actual_loss_fraction=None, one_lot=False, fee=None, exit_rule="", reference_equity=None,
    ) -> None:
        mark = bar.close if price is None else price
        equity = self.cash + shares_after * mark
        fraction = 0.0 if equity <= 0 else shares_after * mark / equity
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
            structure=bar.structure,
            category=bar.category,
            a_t=bar.a_t,
            ema=bar.ema,
            factor=bar.factor,
            fee=fee,
            cash_after=self.cash,
            equity_after=equity,
            position_fraction=fraction,
            exit_rule=exit_rule,
            reference_equity=reference_equity,
        ))

    def _mark(self, index, bar, governed, next_stop) -> None:
        shares = 0 if self.position is None else self.position.shares
        equity = self.cash + shares * bar.close
        fraction = 0.0 if equity <= 0 else shares * bar.close / equity
        self.equity.append(EquityRow(
            index=index,
            time=bar.time,
            equity=equity,
            cash=self.cash,
            shares=shares,
            close=bar.close,
            effective_stop=governed,
            next_stop=next_stop,
            raw_stop=None if governed is None else governed / bar.factor,
            structure=bar.structure,
            category=bar.category,
            position_fraction=fraction,
            planned_risk=None if self.position is None else self.position.planned_risk,
            factor=bar.factor,
            dividend_cash=self.bar_dividend,
            equity_gross=equity + self.cumulative_fees,
            fee=self.bar_fee,
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


def _stop_note(adjusted_stop: Optional[float]) -> str:
    if adjusted_stop is None:
        return ""
    return f"复权止损 {_num(adjusted_stop)}。"


def _num(value: float) -> str:
    text = f"{float(value):.8f}".rstrip("0").rstrip(".")
    return text or "0"
