"""趋势中的停顿：上涨旗杆之后的旗形和三角旗。

旗杆、局部高低点和两条边界都在这里计算，不调用拐点识别，也不写回基础层。
拟合只用到整理终点，信号根只用来判断收盘是否离开通道。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from analysis.structure.config import pause_params_from_dict
from analysis.structure.methods.base import (
    KIND_INSTANCE,
    CategorySpec,
    ColumnSpec,
    MethodResult,
    MethodSample,
    MethodSkipped,
    TableSpec,
    make_mark,
)

FLAG = "flag"
PENNANT = "pennant"
OTHER = "other"
EXIT_UP = "up"
EXIT_DOWN = "down"
_NAMES = {FLAG: "旗形", PENNANT: "三角旗"}
_TEXT_COLOR = "#F3EFE4"
_NOTE = "只判断上涨旗杆之后的旗形和三角旗。边界拟合到信号根的前一根，信号根只用来判断离开。"
PARAM_LABELS = {
    "pole_min": "旗杆最短",
    "pole_max": "旗杆最长",
    "consolidation_min": "整理最短",
    "consolidation_max": "整理最长",
    "atr_lookback": "A 的回看",
    "pole_rise": "旗杆涨幅",
    "efficiency": "方向效率",
    "pole_high": "旗杆收在高位",
    "drawdown_max": "回撤上限",
    "amplitude_max": "整理振幅",
    "pivot_half_width": "拐点半宽",
    "min_highs": "最少高点",
    "min_lows": "最少低点",
    "pivot_span": "拐点跨度",
    "fit_error": "拟合误差",
    "coverage": "覆盖率",
    "boundary_tolerance": "边界容差",
    "min_width": "最小宽度",
    "flag_slope": "旗形斜率差",
    "flag_width_min": "旗形宽度比下限",
    "flag_width_max": "旗形宽度比上限",
    "midline_min": "旗形中线下限",
    "midline_max": "旗形中线上限",
    "pennant_slope": "三角旗斜率",
    "pennant_width_min": "三角旗宽度比下限",
    "pennant_width_max": "三角旗宽度比上限",
    "exit_buffer": "离开缓冲",
    "overlap": "重叠阈值",
    "volume_filter": "整理量能过滤",
    "volume_ratio": "整理量能比",
    "breakout_volume_filter": "突破量能过滤",
    "breakout_volume": "突破量能倍数",
    "pullback_bars": "快速回落根数",
}


@dataclass(frozen=True)
class _Candidate:
    a: int
    e: int
    s: int
    c: int
    t: int
    p: int
    anchor: float
    pole_rise: float
    efficiency: float
    category: str
    exit_direction: Optional[str]
    mean_error: float
    fit: dict


class PauseMethod:
    key = "pause"
    name = "停顿"
    section = "2.3"
    version = "pause-v1"
    kind = KIND_INSTANCE
    requires = ("closes", "highs", "lows", "volumes")
    categories = (
        CategorySpec(FLAG, "旗形", "#E24A4A"),
        CategorySpec(PENNANT, "三角旗", "#E0A045"),
    )
    replaces = ()

    def default_params(self) -> dict:
        return pause_params_from_dict(None).to_dict()

    def check_params(self, params: dict) -> None:
        pause_params_from_dict(params)

    def judge(self, context, params: dict, upstream: dict) -> MethodResult:
        del upstream
        problem = _quote_problem(context.series)
        if problem:
            raise MethodSkipped(problem)
        pack = pause_params_from_dict(params)
        bars = context.series.bars
        highs = [bar.high_geometric for bar in bars]
        lows = [bar.low_geometric for bar in bars]
        closes = [bar.geometric for bar in bars]
        volumes = [bar.volume for bar in bars]
        gaps = [bar.gap_before for bar in bars]
        objects: Dict[str, dict] = {}
        open_ids: List[str] = []
        events: List[dict] = []
        extra_fits: List[dict] = []
        others: List[dict] = []
        seen_other: set = set()
        birth_rank: Dict[int, int] = {}
        watchers: List[dict] = []

        for t in range(len(bars)):
            if gaps[t]:
                _expire_open(objects, open_ids, events, bars, t, lambda _obj: "间断")
            else:
                _expire_open(
                    objects,
                    open_ids,
                    events,
                    bars,
                    t,
                    lambda obj: "整理超过最长根数"
                    if t - obj["consolidation_start"] > pack.consolidation_max
                    else None,
                )
            found = _candidates_at(highs, lows, closes, volumes, gaps, t, pack)
            shaped = []
            loose = []
            for cand in found:
                if cand.category == OTHER:
                    loose.append(cand)
                    continue
                existing = objects.get(_object_id(cand))
                if existing is not None and existing["status"] != "validated":
                    continue
                shaped.append(cand)
            selected = _choose(shaped, pack.overlap)
            applied: List[_Candidate] = []
            for cand in selected:
                obj = _apply_candidate(
                    cand,
                    objects,
                    open_ids,
                    events,
                    extra_fits,
                    bars,
                    birth_rank,
                    watchers,
                )
                if obj is not None:
                    applied.append(cand)
            _demote(objects, open_ids, applied, t, pack.overlap)
            for cand in _choose(
                [item for item in loose if not _overlaps(item, applied, pack.overlap)],
                pack.overlap,
            ):
                key = (cand.a, cand.e)
                if key in seen_other:
                    continue
                seen_other.add(key)
                others.append(
                    {
                        "pole_start": cand.a,
                        "pole_end": cand.e,
                        "consolidation_start": cand.s,
                        "consolidation_end": cand.c,
                        "available_index": cand.t,
                        "mean_error": cand.mean_error,
                        "width_ratio": cand.fit["width_ratio"],
                        "upper_slope": cand.fit["upper_slope"],
                        "lower_slope": cand.fit["lower_slope"],
                    }
                )
            _watch_pullbacks(watchers, objects, bars, closes, t, pack.pullback_bars)

        rows = list(objects.values())
        rows.sort(key=lambda item: (item["start_index"], item["id"]))
        extra_fits.sort(key=lambda item: (item["available_index"], item["object_id"]))
        others.sort(key=lambda item: (item["available_index"], item["pole_start"], item["pole_end"]))
        return MethodResult(
            objects=rows,
            events=events,
            extra={"fits": extra_fits, "others": others},
            notes=[_NOTE],
        )

    def present(self, record: dict, view: dict) -> List[dict]:
        category = record.get("category") or ""
        names = {item.key: item.name for item in self.categories}
        colors = {item.key: item.color for item in self.categories}
        if category not in names or not _draw(record):
            return []
        fits = record.get("fits") or []
        if not fits:
            return []
        fit = fits[-1]
        start = record.get("consolidation_start")
        end = fit.get("consolidation_end")
        if start is None or end is None or int(end) <= int(start):
            return []
        start = int(start)
        end = int(end)
        axis = view.get("price_axis") or "log"
        upper_start = _price(fit["upper_intercept"], axis)
        upper_end = _price(fit["upper_intercept"] + fit["upper_slope"], axis)
        lower_end = _price(fit["lower_intercept"] + fit["lower_slope"], axis)
        lower_start = _price(fit["lower_intercept"], axis)
        corners = (upper_start, upper_end, lower_end, lower_start)
        if any(point is None for point in corners):
            return []
        color = colors[category]
        hover = _hover_lines(record, fit, names[category])
        shared = dict(
            object_id=record["id"],
            method=self.key,
            scale=record.get("scale") or "multi",
            category=category,
            legend="停顿",
            start_index=start,
            end_index=end,
            available_index=int(record.get("available_index") or start),
            hover=hover,
        )
        marks = [
            make_mark(
                kind="band",
                points=[[start, lower_start], [end, lower_end], [end, upper_end], [start, upper_start]],
                color=_rgba(color, 0.22),
                **shared,
            ),
            make_mark(
                kind="text",
                text=names[category],
                color=_TEXT_COLOR,
                points=[[end, upper_end]],
                anchor="price",
                index=end,
                price=upper_end,
                **shared,
            ),
        ]
        closes = {
            int(bar["index"]): float(bar["close"])
            for bar in view.get("closes") or []
            if bar.get("index") is not None and bar.get("close") is not None
        }
        pole_start = record.get("start_index")
        pole_end = record.get("pole_end")
        if pole_start in closes and pole_end in closes:
            marks.append(
                make_mark(
                    kind="segment",
                    points=[[int(pole_start), closes[pole_start]], [int(pole_end), closes[pole_end]]],
                    color=color,
                    start_index=int(pole_start),
                    end_index=int(pole_end),
                    **{key: value for key, value in shared.items() if key not in ("start_index", "end_index")},
                )
            )
        exits = record.get("exit") or []
        if record.get("status") == "broken" and exits:
            signal = int(exits[-1].get("index"))
            span = end - start
            x_value = (signal - start) / span
            upper_signal = _price(fit["upper_intercept"] + fit["upper_slope"] * x_value, axis)
            lower_signal = _price(fit["lower_intercept"] + fit["lower_slope"] * x_value, axis)
            if upper_signal is not None and lower_signal is not None:
                marks.append(
                    make_mark(
                        kind="segment",
                        points=[[end, upper_end], [signal, upper_signal]],
                        color=color,
                        line_type="dashed",
                        start_index=end,
                        end_index=signal,
                        **{key: value for key, value in shared.items() if key not in ("start_index", "end_index")},
                    )
                )
                marks.append(
                    make_mark(
                        kind="segment",
                        points=[[end, lower_end], [signal, lower_signal]],
                        color=color,
                        line_type="dashed",
                        start_index=end,
                        end_index=signal,
                        **{key: value for key, value in shared.items() if key not in ("start_index", "end_index")},
                    )
                )
            if signal in closes:
                marks.append(
                    make_mark(
                        kind="marker",
                        color=color,
                        anchor="price",
                        index=signal,
                        price=closes[signal],
                        start_index=signal,
                        end_index=signal,
                        **{key: value for key, value in shared.items() if key not in ("start_index", "end_index")},
                    )
                )
        return marks

    def tables(self) -> Tuple[TableSpec, ...]:
        columns = (
            ColumnSpec("标识", "id", "text"),
            ColumnSpec("类别", "category", "token", hover=True),
            ColumnSpec("状态", "status", "status", hover=True),
            ColumnSpec("旗杆起点", "start_index", "index"),
            ColumnSpec("旗杆终点", "pole_end", "index"),
            ColumnSpec("整理终点", "fits.-1.consolidation_end", "index", hover=True),
            ColumnSpec("离开", "exit.-1.direction", "token", hover=True),
            ColumnSpec("涨幅 / A", "rise_multiple", "multiple", hover=True),
            ColumnSpec("效率", "efficiency", "multiple", hover=True),
            ColumnSpec("回撤", "fits.-1.drawdown", "multiple", hover=True),
            ColumnSpec("宽度比", "fits.-1.width_ratio", "multiple", hover=True),
            ColumnSpec("拟合误差 / A", "fits.-1.mean_error", "multiple", hover=True),
            ColumnSpec("快速回落", "pullbacks.-1.index", "index", hover=True),
        )
        fit_columns = (
            ColumnSpec("对象", "object_id", "text"),
            ColumnSpec("可用序号", "available_index", "index"),
            ColumnSpec("整理终点", "consolidation_end", "index"),
            ColumnSpec("上截距", "upper_intercept", "multiple"),
            ColumnSpec("上斜率", "upper_slope", "multiple"),
            ColumnSpec("下截距", "lower_intercept", "multiple"),
            ColumnSpec("下斜率", "lower_slope", "multiple"),
            ColumnSpec("起点宽度", "width_start", "multiple"),
            ColumnSpec("终点宽度", "width_end", "multiple"),
            ColumnSpec("宽度比", "width_ratio", "multiple"),
            ColumnSpec("中线", "midline", "multiple"),
            ColumnSpec("高点误差", "upper_error", "multiple"),
            ColumnSpec("低点误差", "lower_error", "multiple"),
            ColumnSpec("覆盖率", "coverage", "multiple"),
            ColumnSpec("回撤", "drawdown", "multiple"),
            ColumnSpec("振幅", "amplitude", "multiple"),
            ColumnSpec("量能比", "volume_ratio", "multiple"),
            ColumnSpec("拟合误差 / A", "mean_error", "multiple"),
        )
        other_columns = (
            ColumnSpec("旗杆起点", "pole_start", "index"),
            ColumnSpec("旗杆终点", "pole_end", "index"),
            ColumnSpec("整理起点", "consolidation_start", "index"),
            ColumnSpec("整理终点", "consolidation_end", "index"),
            ColumnSpec("可用序号", "available_index", "index"),
            ColumnSpec("拟合误差 / A", "mean_error", "multiple"),
            ColumnSpec("宽度比", "width_ratio", "multiple"),
            ColumnSpec("上斜率", "upper_slope", "multiple"),
            ColumnSpec("下斜率", "lower_slope", "multiple"),
        )
        return (
            TableSpec("pauses", "停顿", columns, source="objects"),
            TableSpec("fits", "拟合", fit_columns, appendix=True, source="extra.fits"),
            TableSpec("others", "其他整理", other_columns, appendix=True, source="extra.others"),
        )

    def samples(self) -> Tuple[MethodSample, ...]:
        flag = _sample_prices("flag")
        pennant = _sample_prices("pennant")
        return (
            MethodSample(
                symbol="SAMPLE-FLAG",
                title="停顿：上涨旗形",
                blurb="快速上涨之后，整理的上下边界接近平行并缓降，收盘向上离开。",
                levels=(),
                price_axis="log",
                closes=flag["closes"],
                highs=flag["highs"],
                lows=flag["lows"],
                volumes=flag["volumes"],
            ),
            MethodSample(
                symbol="SAMPLE-PENNANT",
                title="停顿：上涨三角旗",
                blurb="同一段上涨之后，上边界下降、下边界上升，末端宽度收到起点的一半左右。",
                levels=(),
                price_axis="log",
                closes=pennant["closes"],
                highs=pennant["highs"],
                lows=pennant["lows"],
                volumes=pennant["volumes"],
            ),
        )


def example_geometry(kind: str = FLAG, exit_mode: str = EXIT_UP) -> dict:
    """默认参数下的一段几何高低收。`log` 用指数换成实际股价，`uniform` 直接用这些正数。"""
    count = 36
    closes = [1.02] * count
    highs = [1.025] * count
    lows = [1.015] * count
    for step, index in enumerate(range(15, 23)):
        closes[index] = 1.0 + 0.05 * step / 7.0
        highs[index] = closes[index] + 0.002
        lows[index] = closes[index] - 0.002
    upper, lower = _rails(kind)
    margin = 0.003
    start, end = 23, 34
    span = end - start
    for index in range(start, end + 1):
        x_value = (index - start) / span
        top = upper(x_value)
        bottom = lower(x_value)
        if index in (25, 31):
            highs[index] = top
            lows[index] = bottom + margin
        elif index in (26, 32):
            highs[index] = top - margin
            lows[index] = bottom
        else:
            highs[index] = top - margin
            lows[index] = bottom + margin
        closes[index] = (top + bottom) / 2.0
    x_signal = (35 - start) / span
    top_signal = upper(x_signal)
    bottom_signal = lower(x_signal)
    if exit_mode == EXIT_DOWN:
        closes[35] = bottom_signal - 0.008
    elif exit_mode == "forming":
        closes[35] = (top_signal + bottom_signal) / 2.0
    else:
        closes[35] = top_signal + 0.008
    highs[35] = closes[35] + 0.002
    lows[35] = closes[35] - 0.002
    return {
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": [1000.0] * count,
        "a": 15,
        "e": 22,
        "s": 23,
        "c": 34,
        "t": 35,
    }


def _rails(kind: str):
    if kind == PENNANT:
        return (lambda x: 1.06 - 0.008 * x, lambda x: 1.04 + 0.002 * x)
    return (lambda x: 1.056 - 0.004 * x, lambda x: 1.044 - 0.004 * x)


def _sample_prices(kind: str) -> dict:
    geometry = example_geometry(kind, EXIT_UP)
    return {
        "highs": tuple(math.exp(value) for value in geometry["highs"]),
        "lows": tuple(math.exp(value) for value in geometry["lows"]),
        "closes": tuple(math.exp(value) for value in geometry["closes"]),
        "volumes": tuple(geometry["volumes"]),
    }


def _quote_problem(series) -> Optional[str]:
    bars = series.bars
    if any(bar.high is None or bar.low is None or bar.volume is None for bar in bars):
        return "缺少最高价、最低价或成交量，停顿跳过"
    for bar in bars:
        if not (
            math.isfinite(bar.high)
            and math.isfinite(bar.low)
            and bar.high > 0
            and bar.low > 0
            and bar.low <= bar.close <= bar.high
        ):
            return "最高价或最低价不是有限正数，或没有落在最低价和最高价之间，停顿跳过"
        if bar.high_geometric is None or bar.low_geometric is None:
            return "最高价或最低价无法换到当前价格轴，停顿跳过"
        if not (math.isfinite(bar.volume) and bar.volume >= 0):
            return "成交量不是有限非负数，停顿跳过"
    return None


def _candidates_at(highs, lows, closes, volumes, gaps, t: int, pack) -> List[_Candidate]:
    found = []
    for length in range(pack.consolidation_min, pack.consolidation_max + 1):
        start = t - length
        end = start - 1
        consolidation_end = t - 1
        if start < 0 or end < 0 or consolidation_end <= start:
            continue
        for pole in range(pack.pole_min, pack.pole_max + 1):
            anchor_index = end - pole + 1
            if anchor_index < 0:
                continue
            cand = _evaluate(
                highs,
                lows,
                closes,
                volumes,
                gaps,
                anchor_index,
                end,
                start,
                consolidation_end,
                t,
                pole,
                pack,
            )
            if cand is not None:
                found.append(cand)
    return found


def _evaluate(highs, lows, closes, volumes, gaps, a, e, s, c, t, pole, pack) -> Optional[_Candidate]:
    lookback = pack.atr_lookback
    if a - lookback - 1 < 0:
        return None
    if any(gaps[index] for index in range(a - lookback, t + 1)):
        return None
    anchor = _anchor(highs, lows, closes, a, lookback)
    if anchor <= 0:
        return None
    rise = closes[e] - closes[a]
    if rise < pack.pole_rise * anchor:
        return None
    path = sum(abs(closes[index] - closes[index - 1]) for index in range(a + 1, e + 1))
    if path == 0:
        return None
    efficiency = rise / path
    if efficiency < pack.efficiency:
        return None
    if abs(max(highs[a : e + 1]) - closes[e]) > pack.pole_high * anchor:
        return None
    lowest = min(lows[s : c + 1])
    highest = max(highs[s : c + 1])
    drawdown = max(0.0, closes[e] - lowest) / rise
    amplitude = highest - lowest
    if drawdown > pack.drawdown_max or amplitude > pack.amplitude_max * rise:
        return None
    volume_ratio = _volume_ratio(volumes, a, e, s, c)
    if pack.volume_filter and (volume_ratio is None or volume_ratio > pack.volume_ratio):
        return None
    half = pack.pivot_half_width
    high_indexes = _locals(highs, s, c, half, True, gaps)
    low_indexes = _locals(lows, s, c, half, False, gaps)
    if len(high_indexes) < pack.min_highs or len(low_indexes) < pack.min_lows:
        return None
    span = c - s
    if max(high_indexes) - min(high_indexes) < pack.pivot_span * span:
        return None
    if max(low_indexes) - min(low_indexes) < pack.pivot_span * span:
        return None
    high_x = [(index - s) / span for index in high_indexes]
    low_x = [(index - s) / span for index in low_indexes]
    upper_intercept, upper_slope = _ols(high_x, [highs[index] for index in high_indexes])
    lower_intercept, lower_slope = _ols(low_x, [lows[index] for index in low_indexes])
    upper_error = _mae(high_x, [highs[index] for index in high_indexes], upper_intercept, upper_slope)
    lower_error = _mae(low_x, [lows[index] for index in low_indexes], lower_intercept, lower_slope)
    if upper_error > pack.fit_error * anchor or lower_error > pack.fit_error * anchor:
        return None
    tolerance = pack.boundary_tolerance * anchor
    covered = 0
    for index in range(s, c + 1):
        x_value = (index - s) / span
        upper = upper_intercept + upper_slope * x_value
        lower = lower_intercept + lower_slope * x_value
        if highs[index] <= upper + tolerance and lows[index] >= lower - tolerance:
            covered += 1
    coverage = covered / (c - s + 1)
    if coverage < pack.coverage:
        return None
    width_start = upper_intercept - lower_intercept
    width_end = (upper_intercept + upper_slope) - (lower_intercept + lower_slope)
    if width_start < pack.min_width * anchor or width_end < pack.min_width * anchor:
        return None
    width_ratio = width_end / width_start
    midline = (upper_slope + lower_slope) / 2.0
    width_mean = (width_start + width_end) / 2.0
    category = _shape(
        upper_slope,
        lower_slope,
        midline,
        width_ratio,
        width_mean,
        rise,
        anchor,
        pack,
    )
    exit_direction = None
    if category in (FLAG, PENNANT):
        exit_direction = _exit_direction(
            closes,
            volumes,
            s,
            c,
            t,
            upper_intercept,
            upper_slope,
            lower_intercept,
            lower_slope,
            anchor,
            pack,
        )
        if exit_direction == "void":
            return None
        if exit_direction == "forming":
            exit_direction = None
    mean_error = ((upper_error + lower_error) / 2.0) / anchor
    fit = {
        "available_index": t,
        "available_time": None,
        "consolidation_end": c,
        "upper_intercept": upper_intercept,
        "upper_slope": upper_slope,
        "lower_intercept": lower_intercept,
        "lower_slope": lower_slope,
        "width_start": width_start,
        "width_end": width_end,
        "width_ratio": width_ratio,
        "midline": midline,
        "upper_error": upper_error,
        "lower_error": lower_error,
        "coverage": coverage,
        "drawdown": drawdown,
        "amplitude": amplitude,
        "volume_ratio": volume_ratio,
        "mean_error": mean_error,
        "highs": [{"index": index, "price": highs[index]} for index in high_indexes],
        "lows": [{"index": index, "price": lows[index]} for index in low_indexes],
    }
    return _Candidate(
        a=a,
        e=e,
        s=s,
        c=c,
        t=t,
        p=pole,
        anchor=anchor,
        pole_rise=rise,
        efficiency=efficiency,
        category=category,
        exit_direction=exit_direction,
        mean_error=mean_error,
        fit=fit,
    )


def _anchor(highs, lows, closes, a: int, lookback: int) -> float:
    total = 0.0
    for index in range(a - lookback, a):
        previous = closes[index - 1]
        total += max(
            highs[index] - lows[index],
            abs(highs[index] - previous),
            abs(lows[index] - previous),
        )
    return total / lookback


def _volume_ratio(volumes, a: int, e: int, s: int, c: int) -> Optional[float]:
    pole = _mean(volumes[a : e + 1])
    if pole <= 0:
        return None
    return _mean(volumes[s : c + 1]) / pole


def _locals(values, s: int, c: int, half: int, want_max: bool, gaps) -> List[int]:
    found = []
    for index in range(s, c - half + 1):
        left = index - half
        right = index + half
        if left < 0 or right >= len(values):
            continue
        if any(gaps[cursor] for cursor in range(left + 1, right + 1)):
            continue
        window = values[left : right + 1]
        extreme = max(window) if want_max else min(window)
        if values[index] != extreme:
            continue
        if any(values[cursor] == extreme for cursor in range(index + 1, right + 1)):
            continue
        found.append(index)
    return found


def _ols(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    count = len(xs)
    x_mean = sum(xs) / count
    y_mean = sum(ys) / count
    variance = sum((x_value - x_mean) ** 2 for x_value in xs)
    if variance == 0:
        return y_mean, 0.0
    slope = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(xs, ys)) / variance
    return y_mean - slope * x_mean, slope


def _mae(xs, ys, intercept: float, slope: float) -> float:
    return sum(abs(y_value - (intercept + slope * x_value)) for x_value, y_value in zip(xs, ys)) / len(xs)


def _shape(upper_slope, lower_slope, midline, width_ratio, width_mean, rise, anchor, pack) -> str:
    if width_mean > 0 and abs(upper_slope - lower_slope) / width_mean <= pack.flag_slope:
        if pack.flag_width_min <= width_ratio <= pack.flag_width_max:
            if pack.midline_min * rise <= midline <= pack.midline_max * rise:
                return FLAG
    if upper_slope <= -pack.pennant_slope * anchor and lower_slope >= pack.pennant_slope * anchor:
        if pack.pennant_width_min <= width_ratio <= pack.pennant_width_max:
            return PENNANT
    return OTHER


def _exit_direction(closes, volumes, s, c, t, upper_intercept, upper_slope, lower_intercept, lower_slope, anchor, pack):
    span = c - s
    x_signal = (t - s) / span
    upper_signal = upper_intercept + upper_slope * x_signal
    lower_signal = lower_intercept + lower_slope * x_signal
    if upper_signal <= lower_signal:
        return "void"
    buffer = pack.exit_buffer * anchor
    if closes[t] < lower_signal - buffer:
        return EXIT_DOWN
    upper_end = upper_intercept + upper_slope
    if closes[t] > upper_signal + buffer and closes[c] <= upper_end + buffer:
        if pack.breakout_volume_filter:
            average = _mean(volumes[s : c + 1])
            if volumes[t] < pack.breakout_volume * average:
                return "void"
        return EXIT_UP
    return "forming"


def _choose(candidates: Sequence[_Candidate], overlap: float) -> List[_Candidate]:
    ordered = sorted(candidates, key=lambda cand: (cand.mean_error, -cand.efficiency, -cand.p, cand.a, cand.e))
    picked: List[_Candidate] = []
    for cand in ordered:
        if any(_jaccard(cand.a, cand.t, other.a, other.t) >= overlap for other in picked):
            continue
        picked.append(cand)
    return picked


def _overlaps(cand: _Candidate, selected: Sequence[_Candidate], overlap: float) -> bool:
    return any(_jaccard(cand.a, cand.t, other.a, other.t) >= overlap for other in selected)


def _jaccard(left_a: int, left_t: int, right_a: int, right_t: int) -> float:
    start = max(left_a, right_a)
    end = min(left_t, right_t)
    if end < start:
        return 0.0
    intersection = end - start + 1
    union = (left_t - left_a + 1) + (right_t - right_a + 1) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _apply_candidate(cand, objects, open_ids, events, extra_fits, bars, birth_rank, watchers):
    other_category = PENNANT if cand.category == FLAG else FLAG
    other_id = f"pause:{cand.a}:{cand.e}:{other_category}"
    if other_id in open_ids:
        _expire(objects[other_id], events, bars, cand.t, f"改成了{_NAMES[cand.category]}")
        open_ids.remove(other_id)
    oid = _object_id(cand)
    existing = objects.get(oid)
    if existing is not None and existing["status"] != "validated":
        return None
    if existing is None:
        if cand.t not in birth_rank:
            birth_rank[cand.t] = len(birth_rank) + 1
        existing = _new_object(cand, bars, birth_rank[cand.t])
        objects[oid] = existing
        open_ids.append(oid)
        events.append(
            _event(bars, cand.t, "validated", _NAMES[cand.category], oid, existing["revision"], None)
        )
    elif not existing["primary"]:
        existing["primary"] = True
        existing["alternate_of"] = None
    _append_fit(existing, cand, bars, extra_fits)
    if cand.exit_direction:
        existing["status"] = "broken"
        existing["exit"].append(
            {
                "available_index": cand.t,
                "available_time": bars[cand.t].timestamp.isoformat(),
                "direction": cand.exit_direction,
                "index": cand.t,
            }
        )
        reason = "向上，收盘离开上边界" if cand.exit_direction == EXIT_UP else "向下，收盘跌破下边界"
        events.append(
            _event(bars, cand.t, "breakout", reason, oid, existing["revision"], cand.t)
        )
        open_ids.remove(oid)
        if cand.exit_direction == EXIT_UP:
            fit = existing["fits"][-1]
            watchers.append(
                {
                    "id": oid,
                    "exit_index": cand.t,
                    "s": cand.s,
                    "c": cand.c,
                    "upper_intercept": fit["upper_intercept"],
                    "upper_slope": fit["upper_slope"],
                    "done": False,
                }
            )
    return existing


def _new_object(cand: _Candidate, bars, revision: int) -> dict:
    return {
        "id": _object_id(cand),
        "method": "pause",
        "scale": "multi",
        "category": cand.category,
        "status": "validated",
        "start_index": cand.a,
        "end_index": cand.c,
        "start_time": bars[cand.a].timestamp.isoformat(),
        "end_time": bars[cand.c].timestamp.isoformat(),
        "available_index": cand.t,
        "available_time": bars[cand.t].timestamp.isoformat(),
        "revision": revision,
        "primary": True,
        "alternate_of": None,
        "evidence": None,
        "constraints": None,
        "score": None,
        "stats": None,
        "pole_end": cand.e,
        "consolidation_start": cand.s,
        "anchor": cand.anchor,
        "pole_rise": cand.pole_rise,
        "efficiency": cand.efficiency,
        "rise_multiple": cand.pole_rise / cand.anchor,
        "fits": [],
        "exit": [],
        "pullbacks": [],
    }


def _append_fit(obj: dict, cand: _Candidate, bars, extra_fits: List[dict]) -> None:
    fit = dict(cand.fit)
    fit["available_time"] = bars[cand.t].timestamp.isoformat()
    fit["highs"] = [dict(point) for point in cand.fit["highs"]]
    fit["lows"] = [dict(point) for point in cand.fit["lows"]]
    obj["fits"].append(fit)
    obj["end_index"] = cand.c
    obj["end_time"] = bars[cand.c].timestamp.isoformat()
    row = dict(fit)
    row["object_id"] = obj["id"]
    row["highs"] = [dict(point) for point in fit["highs"]]
    row["lows"] = [dict(point) for point in fit["lows"]]
    extra_fits.append(row)


def _demote(objects, open_ids, selected: Sequence[_Candidate], t: int, overlap: float) -> None:
    chosen = {(cand.a, cand.e, cand.category) for cand in selected}
    for oid in open_ids:
        obj = objects[oid]
        if (obj["start_index"], obj["pole_end"], obj["category"]) in chosen:
            continue
        best = None
        best_score = None
        best_rank = None
        last_signal = int(obj["fits"][-1]["available_index"])
        for rank, cand in enumerate(selected):
            if cand.a == obj["start_index"] and cand.e == obj["pole_end"]:
                continue
            score = _jaccard(cand.a, t, obj["start_index"], last_signal)
            if score < overlap:
                continue
            if best is None or score > best_score or (score == best_score and rank < best_rank):
                best = cand
                best_score = score
                best_rank = rank
        if best is not None:
            obj["primary"] = False
            obj["alternate_of"] = _object_id(best)


def _expire_open(objects, open_ids, events, bars, t: int, reason_for) -> None:
    for oid in list(open_ids):
        reason = reason_for(objects[oid])
        if not reason:
            continue
        _expire(objects[oid], events, bars, t, reason)
        open_ids.remove(oid)


def _expire(obj: dict, events: List[dict], bars, t: int, reason: str) -> None:
    if obj["status"] != "validated":
        return
    obj["status"] = "expired"
    events.append(_event(bars, t, "expired", reason, obj["id"], obj["revision"], None))


def _watch_pullbacks(watchers, objects, bars, closes, t: int, limit: int) -> None:
    for watcher in watchers:
        if watcher["done"] or t <= watcher["exit_index"]:
            continue
        if t > watcher["exit_index"] + limit or bars[t].gap_before:
            watcher["done"] = True
            continue
        span = watcher["c"] - watcher["s"]
        if span <= 0:
            watcher["done"] = True
            continue
        x_value = (t - watcher["s"]) / span
        upper = watcher["upper_intercept"] + watcher["upper_slope"] * x_value
        if closes[t] < upper:
            objects[watcher["id"]]["pullbacks"].append(
                {
                    "available_index": t,
                    "available_time": bars[t].timestamp.isoformat(),
                    "index": t,
                }
            )
            watcher["done"] = True


def _event(bars, t: int, event_type: str, reason: str, object_id: str, revision: int, cross) -> dict:
    return {
        "event_type": event_type,
        "event_index": t,
        "event_time": bars[t].timestamp.isoformat(),
        "available_index": t,
        "available_time": bars[t].timestamp.isoformat(),
        "reason": reason,
        "reference_close": bars[t].close,
        "object_id": object_id,
        "object_kind": "pause",
        "revision": revision,
        "price_cross_index": cross,
    }


def _object_id(cand: _Candidate) -> str:
    return f"pause:{cand.a}:{cand.e}:{cand.category}"


def _draw(record: dict) -> bool:
    status = record.get("status")
    if status == "validated":
        return bool(record.get("primary"))
    return status in ("broken", "expired")


def _hover_lines(record: dict, fit: dict, category_name: str) -> List[str]:
    exits = record.get("exit") or []
    direction = exits[-1].get("direction") if exits else ""
    direction_name = {"up": "向上", "down": "向下"}.get(direction, "形成中")
    status = {"validated": "已验证", "broken": "已被收盘价突破", "expired": "过期"}.get(
        record.get("status"), record.get("status") or ""
    )
    return [
        f"停顿 · {category_name}",
        f"{status} · {direction_name}",
        f"{record.get('start_index')}–{fit.get('consolidation_end')}",
        f"P/A {_num(record.get('rise_multiple'))} · ER {_num(record.get('efficiency'))}",
        f"D {_num(fit.get('drawdown'))} · 宽度比 {_num(fit.get('width_ratio'))}",
        f"拟合误差/A {_num(fit.get('mean_error'))}",
    ]


def _price(geometric: float, axis: str) -> Optional[float]:
    value = math.exp(geometric) if axis == "log" else geometric
    if not math.isfinite(value) or value <= 0:
        return None
    return value


def _rgba(color: str, alpha: float) -> str:
    text = color.lstrip("#")
    return f"rgba({int(text[0:2], 16)}, {int(text[2:4], 16)}, {int(text[4:6], 16)}, {alpha})"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _num(value) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "—"
