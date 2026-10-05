"""三尺度因果拐点。

每个尺度自己找高点或找低点。进入一段时，用当时已经算好的波动尺度固定
门槛；段内候选极值可以移到同价平台的最后一根，门槛不再改。确认之后，
极值位置、价格和确认时间只追加、不回写。

序列开头还没有方向，先同时跟踪高点和低点。哪一侧先走出门槛，就确认
那一侧，然后只寻找相反的极值。三个尺度不要求互相包含。

历史回放里，系统可用时间等于确认时间。末端没走出门槛的候选放在
`temporary`，不进入 `confirmed`。间断把未确认的候选丢掉，新的一段从
间断后的第一根重新开始。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from analysis.structure.config import SCALE_NAMES
from analysis.structure.series import StandardSeries
from analysis.structure.volatility import compute_volatility, reversal_threshold

MODE_UNDECIDED = "undecided"
MODE_SEEK_HIGH = "seek_high"
MODE_SEEK_LOW = "seek_low"
ROLE_HIGH = "high"
ROLE_LOW = "low"
ROLE_PENDING = "pending"


@dataclass(frozen=True)
class Pivot:
    """已确认拐点。字段在确认当时写定。"""

    scale: str
    role: str
    extreme_index: int
    confirm_index: int
    extreme_time: datetime
    confirm_time: datetime
    available_time: datetime
    price: float
    geometric_price: float
    threshold: float
    volatility: float
    threshold_index: int
    segment_start_index: int

    def __post_init__(self) -> None:
        if self.scale not in SCALE_NAMES:
            raise ValueError(f"未知尺度 {self.scale!r}")
        if self.role not in (ROLE_HIGH, ROLE_LOW):
            raise ValueError(f"已确认拐点的角色必须是 high 或 low，收到 {self.role!r}")
        if self.confirm_index <= self.extreme_index:
            raise ValueError(
                "确认序号必须晚于极值序号: "
                f"extreme={self.extreme_index} confirm={self.confirm_index}"
            )
        if not (self.segment_start_index <= self.extreme_index <= self.confirm_index):
            raise ValueError(
                "极值必须落在本段内: "
                f"start={self.segment_start_index} extreme={self.extreme_index} "
                f"confirm={self.confirm_index}"
            )
        if not (self.segment_start_index <= self.threshold_index <= self.confirm_index):
            raise ValueError(
                "门槛取样序号必须落在本段开始和确认之间: "
                f"start={self.segment_start_index} threshold={self.threshold_index} "
                f"confirm={self.confirm_index}"
            )
        if self.threshold <= 0 or self.volatility <= 0:
            raise ValueError("门槛和波动尺度必须为正")

    def to_dict(self) -> dict:
        return {
            "scale": self.scale,
            "role": self.role,
            "extreme_index": self.extreme_index,
            "confirm_index": self.confirm_index,
            "extreme_time": self.extreme_time.isoformat(),
            "confirm_time": self.confirm_time.isoformat(),
            "available_time": self.available_time.isoformat(),
            "price": self.price,
            "geometric_price": self.geometric_price,
            "threshold": self.threshold,
            "volatility": self.volatility,
            "threshold_index": self.threshold_index,
            "segment_start_index": self.segment_start_index,
        }


@dataclass(frozen=True)
class TemporaryExtreme:
    """末端尚未确认的候选。`pending` 表示这段高低还落在同一根上。"""

    scale: str
    role: str
    extreme_index: int
    extreme_time: datetime
    price: float
    geometric_price: float
    threshold: Optional[float]
    volatility: Optional[float]
    threshold_index: Optional[int]
    segment_start_index: int

    def __post_init__(self) -> None:
        if self.scale not in SCALE_NAMES:
            raise ValueError(f"未知尺度 {self.scale!r}")
        if self.role not in (ROLE_HIGH, ROLE_LOW, ROLE_PENDING):
            raise ValueError(f"临时拐点角色无效: {self.role!r}")
        if self.extreme_index < self.segment_start_index:
            raise ValueError("临时极值不能早于本段起点")
        if (self.threshold is None) != (self.volatility is None):
            raise ValueError("临时拐点的门槛和波动尺度必须同时存在或同时缺失")

    def to_dict(self) -> dict:
        return {
            "scale": self.scale,
            "role": self.role,
            "extreme_index": self.extreme_index,
            "extreme_time": self.extreme_time.isoformat(),
            "price": self.price,
            "geometric_price": self.geometric_price,
            "threshold": self.threshold,
            "volatility": self.volatility,
            "threshold_index": self.threshold_index,
            "segment_start_index": self.segment_start_index,
        }


@dataclass(frozen=True)
class PivotResult:
    confirmed: Tuple[Pivot, ...]
    temporary: Tuple[TemporaryExtreme, ...]

    def confirmed_for(self, scale: str) -> Tuple[Pivot, ...]:
        return tuple(pivot for pivot in self.confirmed if pivot.scale == scale)

    def temporary_for(self, scale: str) -> Tuple[TemporaryExtreme, ...]:
        return tuple(item for item in self.temporary if item.scale == scale)


@dataclass
class _ScaleState:
    name: str
    mode: str = MODE_UNDECIDED
    segment_start_index: int = 0
    high_index: Optional[int] = None
    low_index: Optional[int] = None
    threshold: Optional[float] = None
    volatility: Optional[float] = None
    threshold_index: Optional[int] = None


class PivotDetector:
    """逐根推进。`advance_to` 可以中途停下再继续，已确认拐点不回写。"""

    def __init__(self, series: StandardSeries) -> None:
        self.series = series
        self.vol = compute_volatility(series)
        self._states = {name: _ScaleState(name=name) for name in SCALE_NAMES}
        self._confirmed: List[Pivot] = []
        self._cursor = 0

    def advance_to(self, bar_count: int) -> PivotResult:
        count = len(self.series)
        if bar_count < 0 or bar_count > count:
            raise ValueError(f"bar_count 必须在 0 和 {count} 之间，收到 {bar_count}")
        if bar_count < self._cursor:
            raise ValueError("拐点检测不能回退到更短的前缀")
        while self._cursor < bar_count:
            self._consume(self._cursor)
            self._cursor += 1
        return self.snapshot()

    def snapshot(self) -> PivotResult:
        temporary: List[TemporaryExtreme] = []
        for name in SCALE_NAMES:
            temporary.extend(self._temporary_for(self._states[name]))
        return PivotResult(confirmed=tuple(self._confirmed), temporary=tuple(temporary))

    def _consume(self, index: int) -> None:
        if self.series.bars[index].gap_before:
            for name in SCALE_NAMES:
                self._reset_region(self._states[name], index)
            return
        for name in SCALE_NAMES:
            self._consume_scale(self._states[name], index)

    def _reset_region(self, state: _ScaleState, index: int) -> None:
        state.mode = MODE_UNDECIDED
        state.segment_start_index = index
        state.high_index = index
        state.low_index = index
        state.threshold = None
        state.volatility = None
        state.threshold_index = None
        vol = self.vol[index]
        if vol is not None:
            self._freeze(state, index, vol)

    def _consume_scale(self, state: _ScaleState, index: int) -> None:
        vol = self.vol[index]
        if state.threshold is None and vol is not None:
            self._freeze(state, index, vol)
        close = self.series.bars[index].close
        if state.mode == MODE_UNDECIDED:
            self._update_undecided(state, index, close)
            if state.threshold is not None:
                self._confirm_undecided_if_ready(state, index)
            return
        if state.mode == MODE_SEEK_HIGH:
            self._seek_high(state, index, close)
            return
        if state.mode == MODE_SEEK_LOW:
            self._seek_low(state, index, close)
            return
        raise RuntimeError(f"未知拐点状态 {state.mode}")

    def _freeze(self, state: _ScaleState, index: int, vol: float) -> None:
        anchor = self.series.bars[state.segment_start_index].close
        scale = self.series.params.scale(state.name)
        state.volatility = vol
        state.threshold_index = index
        state.threshold = reversal_threshold(
            scale,
            vol,
            self.series.price_axis,
            anchor,
        )

    def _update_undecided(self, state: _ScaleState, index: int, close: float) -> None:
        if state.high_index is None or close >= self._close(state.high_index):
            state.high_index = index
        if state.low_index is None or close <= self._close(state.low_index):
            state.low_index = index

    def _confirm_undecided_if_ready(self, state: _ScaleState, index: int) -> None:
        options = []
        if state.high_index is not None and state.high_index != index:
            reversal = self._geom(state.high_index) - self._geom(index)
            if reversal >= state.threshold:
                options.append((ROLE_HIGH, state.high_index, reversal))
        if state.low_index is not None and state.low_index != index:
            reversal = self._geom(index) - self._geom(state.low_index)
            if reversal >= state.threshold:
                options.append((ROLE_LOW, state.low_index, reversal))
        if not options:
            return
        # 同一根上两侧都够门槛时，取反向幅度更大的一侧；幅度相同则取更早的极值。
        options.sort(key=lambda item: (-item[2], item[1]))
        role, extreme_index, _reversal = options[0]
        self._confirm(state, role, extreme_index, index)

    def _seek_high(self, state: _ScaleState, index: int, close: float) -> None:
        if state.high_index is None or close >= self._close(state.high_index):
            state.high_index = index
            return
        if state.threshold is None:
            return
        reversal = self._geom(state.high_index) - self._geom(index)
        if reversal >= state.threshold:
            self._confirm(state, ROLE_HIGH, state.high_index, index)

    def _seek_low(self, state: _ScaleState, index: int, close: float) -> None:
        if state.low_index is None or close <= self._close(state.low_index):
            state.low_index = index
            return
        if state.threshold is None:
            return
        reversal = self._geom(index) - self._geom(state.low_index)
        if reversal >= state.threshold:
            self._confirm(state, ROLE_LOW, state.low_index, index)

    def _confirm(
        self,
        state: _ScaleState,
        role: str,
        extreme_index: int,
        confirm_index: int,
    ) -> None:
        if state.threshold is None or state.volatility is None or state.threshold_index is None:
            return
        extreme = self.series.bars[extreme_index]
        confirm = self.series.bars[confirm_index]
        self._confirmed.append(
            Pivot(
                scale=state.name,
                role=role,
                extreme_index=extreme_index,
                confirm_index=confirm_index,
                extreme_time=extreme.timestamp,
                confirm_time=confirm.timestamp,
                available_time=confirm.timestamp,
                price=extreme.close,
                geometric_price=extreme.geometric,
                threshold=state.threshold,
                volatility=state.volatility,
                threshold_index=state.threshold_index,
                segment_start_index=state.segment_start_index,
            )
        )
        if role == ROLE_HIGH:
            state.mode = MODE_SEEK_LOW
            state.low_index = confirm_index
            state.high_index = None
        else:
            state.mode = MODE_SEEK_HIGH
            state.high_index = confirm_index
            state.low_index = None
        state.segment_start_index = confirm_index
        state.threshold = None
        state.volatility = None
        state.threshold_index = None
        vol = self.vol[confirm_index]
        if vol is not None:
            self._freeze(state, confirm_index, vol)

    def _temporary_for(self, state: _ScaleState) -> List[TemporaryExtreme]:
        if state.mode == MODE_SEEK_HIGH and state.high_index is not None:
            return [self._temporary(state, ROLE_HIGH, state.high_index)]
        if state.mode == MODE_SEEK_LOW and state.low_index is not None:
            return [self._temporary(state, ROLE_LOW, state.low_index)]
        if state.high_index is None:
            return []
        if state.low_index == state.high_index:
            return [self._temporary(state, ROLE_PENDING, state.high_index)]
        items = []
        if state.high_index is not None:
            items.append(self._temporary(state, ROLE_HIGH, state.high_index))
        if state.low_index is not None:
            items.append(self._temporary(state, ROLE_LOW, state.low_index))
        return items

    def _temporary(self, state: _ScaleState, role: str, index: int) -> TemporaryExtreme:
        bar = self.series.bars[index]
        return TemporaryExtreme(
            scale=state.name,
            role=role,
            extreme_index=index,
            extreme_time=bar.timestamp,
            price=bar.close,
            geometric_price=bar.geometric,
            threshold=state.threshold,
            volatility=state.volatility,
            threshold_index=state.threshold_index,
            segment_start_index=state.segment_start_index,
        )

    def _close(self, index: int) -> float:
        return self.series.bars[index].close

    def _geom(self, index: int) -> float:
        return self.series.bars[index].geometric


def detect_pivots(series: StandardSeries, bar_count: Optional[int] = None) -> PivotResult:
    """计算前 `bar_count` 根上的拐点。缺省使用整段。

    传入 `bar_count` 会按这段前缀重新构建序列，和把更短的输入单独喂进来一致。
    """
    if bar_count is not None:
        series = series.prefix(bar_count)
    return PivotDetector(series).advance_to(len(series))


def format_trace(series: StandardSeries, result: PivotResult) -> str:
    """给人核对极值序号和确认序号的文本。不参与拟合。"""
    lines = [
        (
            f"bars={len(series)} axis={series.price_axis} interval={series.interval} "
            f"hash={series.data_hash} version={series.param_version}"
        )
    ]
    for pivot in result.confirmed:
        lines.append(
            "CONFIRMED "
            f"{pivot.scale:5} {pivot.role:4} "
            f"extreme={pivot.extreme_index} confirm={pivot.confirm_index} "
            f"price={pivot.price:.6f} threshold={pivot.threshold:.6f} "
            f"extreme_time={pivot.extreme_time.isoformat()} "
            f"confirm_time={pivot.confirm_time.isoformat()}"
        )
    for item in result.temporary:
        threshold = "None" if item.threshold is None else f"{item.threshold:.6f}"
        lines.append(
            "TEMPORARY "
            f"{item.scale:5} {item.role:7} "
            f"extreme={item.extreme_index} price={item.price:.6f} threshold={threshold}"
        )
    return "\n".join(lines)
