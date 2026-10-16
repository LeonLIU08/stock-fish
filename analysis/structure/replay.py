"""按每根 K 线做前缀拟合，得到页面可以拖动的 as-of 帧。

每一帧只把截止到该根的行情交给 `build_snapshot`。后面的收盘价只放在公共部分里，供图上对照。
"""
from __future__ import annotations

from typing import Optional

from analysis.structure.series import StandardSeries
from analysis.structure.snapshot import build_snapshot, normalize_visible_methods, normalize_visible_scales
from analysis.structure.view_model import (
    AXIS_LABEL,
    COLORS,
    DIRECTION_LABEL,
    ROLE_LABEL,
    SCALE_CHART_LABEL,
    SCALE_STYLE,
    STATUS_LABEL,
    ZONE_LABEL,
    _close_point,
    _not_after,
    _parse_time,
    build_view_model,
)


def build_replay(
    series: StandardSeries,
    *,
    symbol: str,
    requested_start=None,
    requested_end=None,
    visible_scales=None,
    visible_methods=None,
    data_source: str = "",
    engine_version: Optional[str] = None,
) -> dict:
    """对每一根单独拟合。拟合完成后丢掉该日的完整快照，只留下图元和识别表。"""
    closes = [
        _close_point(
            {"index": bar.index, "time": bar.timestamp.isoformat(), "close": bar.close},
            series.interval,
        )
        for bar in series.bars
    ]
    frames = []
    for count in range(1, len(series) + 1):
        bar = closes[count - 1]
        try:
            snapshot = build_snapshot(
                series.prefix(count),
                symbol=symbol,
                requested_start=requested_start,
                requested_end=requested_end,
                as_of=None,
                visible_scales=visible_scales,
                visible_methods=visible_methods,
                data_source=data_source,
                engine_version=engine_version,
            )
        except Exception as exc:
            frames.append(_failed_frame(bar, series.price_axis, exc))
            continue
        frames.append(_frame_from_snapshot(snapshot, bar))
        del snapshot
    return {
        "price_axis": series.price_axis,
        "axis_label": AXIS_LABEL.get(series.price_axis, series.price_axis),
        "default_scales": normalize_visible_scales(visible_scales),
        "default_methods": _default_methods(series, visible_methods),
        "labels": {
            "scales": SCALE_CHART_LABEL,
            "roles": ROLE_LABEL,
            "statuses": STATUS_LABEL,
            "zones": ZONE_LABEL,
            "directions": DIRECTION_LABEL,
        },
        "colors": COLORS,
        "styles": SCALE_STYLE,
        "closes": closes,
        "frames": frames,
    }


def select_frame_index(frames, as_of) -> int:
    """`--as-of` 决定打开时落在哪一帧。晚于最后一根时用最后一帧，早于第一根时用第一帧。"""
    if not frames:
        raise ValueError("没有 as-of 帧")
    if not as_of:
        return len(frames) - 1
    limit = _parse_time(as_of)
    if limit is None:
        return len(frames) - 1
    chosen = None
    for index, frame in enumerate(frames):
        when = _parse_time(frame.get("time"))
        if when is not None and _not_after(when, limit):
            chosen = index
    if chosen is None:
        return 0
    return chosen


def compose_view_model(replay: dict, frame: dict) -> dict:
    """图用整段收盘价，结构用这一帧的前缀拟合。"""
    chart = frame["chart"]
    return {
        "price_axis": replay["price_axis"],
        "axis_label": replay["axis_label"],
        "default_scales": list(replay["default_scales"]),
        "labels": replay["labels"],
        "colors": replay["colors"],
        "styles": replay["styles"],
        "closes": replay["closes"],
        "pivots": chart["pivots"],
        "boundaries": chart["boundaries"],
        "zones": chart["zones"],
        "segments": chart["segments"],
        "as_of_index": chart["as_of_index"],
        "marks": chart["marks"],
        "default_methods": chart["default_methods"],
        "replaced_layers": chart["replaced_layers"],
        "method_objects": chart["method_objects"],
    }


def slim_replay(replay: dict) -> dict:
    """页面里只嵌每帧的图元和识别表。收盘价和样式留在 view model 里。"""
    frames = []
    for frame in replay["frames"]:
        frames.append(
            {
                "index": frame["index"],
                "label": frame["label"],
                "error": frame["error"],
                "chart": frame["chart"],
                "counts": frame["counts"],
                "sections": frame["sections"],
            }
        )
    return {"frames": frames}


def _frame_from_snapshot(snapshot: dict, bar: dict) -> dict:
    from analysis.structure.html_report import recognition_sections

    model = build_view_model(snapshot)
    index = int(bar["index"])
    sections = recognition_sections(snapshot)
    return {
        "index": index,
        "time": bar["time"],
        "label": bar["label"],
        "error": None,
        "chart": {
            "pivots": model["pivots"],
            "boundaries": model["boundaries"],
            "zones": model["zones"],
            "segments": model["segments"],
            "marks": model["marks"],
            "as_of_index": index,
            "replaced_layers": model["replaced_layers"],
            "method_objects": model["method_objects"],
            "default_methods": model["default_methods"],
        },
        "counts": sections["counts"],
        "sections": sections,
    }


def _failed_frame(bar: dict, price_axis: str, exc: BaseException) -> dict:
    from analysis.structure.html_report import recognition_sections

    message = str(exc) or exc.__class__.__name__
    sections = recognition_sections(_empty_snapshot(price_axis))
    index = int(bar["index"])
    return {
        "index": index,
        "time": bar["time"],
        "label": bar["label"],
        "error": message,
        "chart": {
            "pivots": [],
            "boundaries": [],
            "zones": [],
            "segments": [],
            "marks": [],
            "as_of_index": index,
            "replaced_layers": [],
            "method_objects": {},
            "default_methods": [],
        },
        "counts": sections["counts"],
        "sections": sections,
    }


def _empty_snapshot(price_axis: str) -> dict:
    return {
        "identity": {"price_axis": price_axis},
        "pivots": [],
        "temporary_pivots": [],
        "segments": [],
        "temporary_segments": [],
        "boundaries": [],
        "zones": [],
        "events": [],
        "pivot_stats": [],
        "relations": {},
        "methods": {},
        "params": {},
    }


def _default_methods(series: StandardSeries, visible_methods) -> list:
    from analysis.structure.config import enabled_methods

    enabled = enabled_methods(series.param_version)
    return normalize_visible_methods(visible_methods, enabled)
