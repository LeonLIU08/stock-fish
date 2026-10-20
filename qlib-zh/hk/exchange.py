"""回测下单按每只股票的每手股数取整。训练不经过这里。"""

from __future__ import annotations

from qlib.backtest.decision import Order
from qlib.backtest.exchange import Exchange
from qlib.contrib.strategy import TopkDropoutStrategy

from lots import floor_to_lot


class HKLotExchange(Exchange):
    """用调用时填入的 lot_sizes，把成交股数向下取整到该股票的整手。"""

    lot_sizes: dict[str, int] = {}
    default_lot: int = 100
    _round_stock_id: str | None = None

    def _lot_for(self, stock_id: str | None) -> int | None:
        if not stock_id:
            return None
        raw = str(stock_id).strip().upper()
        digits = "".join(ch for ch in raw if ch.isdigit())
        code = digits.zfill(5) if digits else raw
        if code in self.lot_sizes:
            return int(self.lot_sizes[code])
        if raw in self.lot_sizes:
            return int(self.lot_sizes[raw])
        return None

    def round_amount_by_trade_unit(
        self,
        deal_amount,
        factor=None,
        stock_id=None,
        start_time=None,
        end_time=None,
    ):
        stock_id = stock_id or self._round_stock_id
        lot = self._lot_for(stock_id)
        if lot is None and stock_id:
            lot = self.default_lot
        if lot is None:
            # 策略内部先按 1 股取整，此时还没有股票代码，但带了复权因子。
            return super().round_amount_by_trade_unit(
                deal_amount, factor, stock_id, start_time, end_time
            )
        return floor_to_lot(float(deal_amount), lot)

    def deal_order(self, order, trade_account=None, position=None, dealt_order_amount=None):
        self._round_stock_id = getattr(order, "stock_id", None)
        try:
            if dealt_order_amount is None:
                return super().deal_order(order, trade_account, position)
            return super().deal_order(order, trade_account, position, dealt_order_amount)
        finally:
            self._round_stock_id = None


class LotTopkDropoutStrategy(TopkDropoutStrategy):
    """买入股数再按该股票的每手股数向下取整。卖出仍一次卖完已有持仓。"""

    def generate_trade_decision(self, execute_result=None):
        decision = super().generate_trade_decision(execute_result)
        exchange = self.trade_exchange
        kept = []
        for order in decision.order_list:
            if order.direction == Order.BUY:
                order.amount = exchange.round_amount_by_trade_unit(
                    order.amount,
                    getattr(order, "factor", None),
                    stock_id=order.stock_id,
                    start_time=order.start_time,
                    end_time=order.end_time,
                )
                if order.amount <= 0:
                    continue
            kept.append(order)
        decision.order_list = kept
        return decision
