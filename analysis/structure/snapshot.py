"""把一次运行收成唯一的 JSON 快照。

短、中、长的定义来自参数包。请求的起止日期只决定送进来的 K 线，
不改 k、门槛下限和最大跨度。
"""
from __future__ import annotations

import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

from analysis.structure.config import PARAM_VERSION, SCALE_NAMES, StructureParams
from analysis.structure.lifecycle import build_lifecycle
from analysis.structure.pivots import detect_pivots
from analysis.structure.segments import build_segments
from analysis.structure.series import StandardSeries
from analysis.structure.view_model import (
    AXIS_LABEL,
    OVERLAP_NOTE,
    READING_NOTE,
    SCALE_DEFINITION_NOTE,
    build_relations,
    pivot_segment_stats,
)
from analysis.structure.volatility import compute_volatility
from analysis.structure.zones import build_zones

_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_requested_window(
    *,
    start: Optional[date],
    end: Optional[date],
    years: Optional[float],
    today: date,
) -> tuple[date, date]:
    """把起止日期或回看年数收成一个闭区间。

    这个区间只用来选择 K 线。它不返回尺度参数，也不按区间长度改写短、中、长。
    """
    if start is not None and years is not None:
        raise ValueError("起止日期和回看年数请分开使用")
    if years is not None and years <= 0:
        raise ValueError("回看年数必须为正")
    resolved_end = end or today
    if start is not None:
        resolved_start = start
    else:
        span = 1.0 if years is None else years
        resolved_start = resolved_end - timedelta(days=int(round(span * 365)))
    if resolved_end < resolved_start:
        raise ValueError("结束日期不能早于开始日期")
    return resolved_start, resolved_end


def normalize_as_of(value, sample_time: Optional[datetime] = None) -> Optional[str]:
    """只写日期时包含当天全部 K 线。"""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
        date_only = False
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day, 23, 59, 59, 999999)
        date_only = True
    else:
        text = str(value).strip()
        date_only = len(text) == 10
        parsed = _parse_user_time(text)
        if date_only:
            parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    if sample_time is not None and (parsed.tzinfo is None) != (sample_time.tzinfo is None):
        if sample_time.tzinfo is not None and parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=sample_time.tzinfo)
    return parsed.isoformat()


def resolve_engine_version(explicit: Optional[str] = None) -> tuple[str, str]:
    """返回 git 短哈希。取不到时版本留空，并带上一句说明。"""
    if explicit is not None:
        note = "" if explicit else "取不到 git 短哈希，引擎代码版本留空。"
        return explicit, note
    try:
        raw = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        text = raw.decode().strip()
        if text:
            return text, ""
    except (OSError, subprocess.SubprocessError):
        pass
    return "", "取不到 git 短哈希，引擎代码版本留空。"


def normalize_visible_scales(value: Optional[Sequence[str] | str]) -> list[str]:
    """主图初次打开哪些尺度。三个尺度仍会全部拟合。"""
    if value is None:
        return ["mid"]
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",") if part.strip()]
    else:
        names = [str(part).strip() for part in value if str(part).strip()]
    if not names:
        return ["mid"]
    unknown = [name for name in names if name not in SCALE_NAMES]
    if unknown:
        raise ValueError(f"尺度必须是 short、mid、long，收到 {unknown}")
    ordered = []
    for name in names:
        if name not in ordered:
            ordered.append(name)
    return ordered


def build_snapshot(
    series: StandardSeries,
    *,
    symbol: str,
    requested_start: Optional[date | str] = None,
    requested_end: Optional[date | str] = None,
    as_of=None,
    visible_scales: Optional[Sequence[str] | str] = None,
    data_source: str = "",
    engine_version: Optional[str] = None,
) -> dict:
    """对已经结束的 K 线做一次完整拟合，并写成可复现的快照。"""
    if not str(symbol).strip():
        raise ValueError("标的不能为空")
    pivots = detect_pivots(series)
    segments = build_segments(series, pivots)
    zoned = build_zones(series, pivots)
    main_ids = {zone.id for zone in zoned.zones if zone.draw_on_main}
    lifecycle = build_lifecycle(series, pivots)
    volatility = compute_volatility(series)
    version, version_note = resolve_engine_version(engine_version)
    sample_time = series.bars[0].timestamp
    as_of_text = normalize_as_of(as_of, sample_time)
    params = series.params
    boundaries = [item.to_dict() for item in lifecycle.boundaries]
    zones = []
    for item in lifecycle.zones:
        payload = item.to_dict()
        payload["main_chart"] = item.id in main_ids
        zones.append(payload)
    confirmed = [item.to_dict() for item in pivots.confirmed]
    temporary = [item.to_dict() for item in pivots.temporary]
    segment_rows = [item.to_dict() for item in segments.segments]
    temporary_segments = [item.to_dict() for item in segments.temporary]
    events = [item.to_dict() for item in lifecycle.events]
    closes = [
        {
            "index": bar.index,
            "time": bar.timestamp.isoformat(),
            "close": bar.close,
            "gap_before": bar.gap_before,
        }
        for bar in series.bars
    ]
    gap_indexes = list(series.gap_indices)
    return {
        "kind": "trend_structure_snapshot",
        "reading": READING_NOTE,
        "scale_definition_note": SCALE_DEFINITION_NOTE,
        "identity": _identity(
            series,
            symbol=symbol.strip(),
            requested_start=requested_start,
            requested_end=requested_end,
            as_of_text=as_of_text,
            visible_scales=normalize_visible_scales(visible_scales),
            data_source=data_source,
            engine_version=version,
            engine_version_note=version_note,
            params=params,
        ),
        "quality": _quality(series, boundaries, zones, gap_indexes),
        "volatility": _volatility(series, volatility, confirmed),
        "pivot_stats": pivot_segment_stats(confirmed, temporary, segment_rows, series.price_axis),
        "pivots": confirmed,
        "temporary_pivots": temporary,
        "segments": segment_rows,
        "temporary_segments": temporary_segments,
        "boundaries": boundaries,
        "zones": zones,
        "events": events,
        "relations": _full_relations(boundaries, zones, segment_rows, lifecycle),
        "closes": closes,
        "params": params.to_dict(),
    }


def _identity(
    series: StandardSeries,
    *,
    symbol: str,
    requested_start,
    requested_end,
    as_of_text: Optional[str],
    visible_scales: Sequence[str],
    data_source: str,
    engine_version: str,
    engine_version_note: str,
    params: StructureParams,
) -> dict:
    return {
        "symbol": symbol,
        "interval": series.interval,
        "start_time": series.bars[0].timestamp.isoformat(),
        "end_time": series.bars[-1].timestamp.isoformat(),
        "requested_start": _date_text(requested_start),
        "requested_end": _date_text(requested_end),
        "param_version": series.param_version,
        "data_hash": series.data_hash,
        "price_basis": series.price_basis,
        "price_axis": series.price_axis,
        "price_axis_label": AXIS_LABEL.get(series.price_axis, series.price_axis),
        "engine_version": engine_version,
        "engine_version_note": engine_version_note,
        "as_of": as_of_text,
        "visible_scales": list(visible_scales),
        "data_source": data_source,
        "bar_count": len(series),
        "scale_definition": {
            "note": SCALE_DEFINITION_NOTE,
            "param_version": params.version or PARAM_VERSION,
            "scales": [scale.to_dict() for scale in params.scales],
        },
    }


def _quality(series: StandardSeries, boundaries: Sequence[dict], zones: Sequence[dict], gap_indexes: Sequence[int]) -> dict:
    gaps = []
    for index in gap_indexes:
        previous = series.bars[index - 1]
        current = series.bars[index]
        gaps.append(
            {
                "index": index,
                "time": current.timestamp.isoformat(),
                "previous_time": previous.timestamp.isoformat(),
            }
        )
    crossing = []
    for item in boundaries:
        if _range_crosses(int(item["start_index"]), int(item["end_index"]), gap_indexes):
            crossing.append(item["id"])
    for item in zones:
        if _range_crosses(int(item["effective_start"]), int(item["effective_end"]), gap_indexes):
            crossing.append(item["id"])
        elif _range_crosses(int(item["projected_start"]), int(item["projected_end"]), gap_indexes):
            crossing.append(item["id"])
    if not gaps:
        note = "没有间断，数据质量正常。趋势线和趋势区间都不跨越间断。"
        grade = "正常"
    elif crossing:
        note = "有间断，且这些对象的起止序号跨越了间断：" + "、".join(crossing) + "。"
        grade = "降级"
    else:
        places = "、".join(str(item["index"]) for item in gaps)
        note = f"有间断，数据质量降级。间断落在序号 {places}。趋势线和趋势区间的起止序号都不跨越这些间断。"
        grade = "降级"
    return {
        "bar_count": len(series),
        "gap_count": len(gaps),
        "gaps": gaps,
        "rejected_non_positive": 0,
        "grade": grade,
        "grade_note": note,
        "crossing_ids": crossing,
    }


def _volatility(series: StandardSeries, volatility: Sequence[Optional[float]], pivots: Sequence[dict]) -> dict:
    unit = "对数收益" if series.price_axis == "log" else "价格差"
    if series.price_axis == "log":
        floor_note = "门槛下限的单位是对数收益。"
    else:
        floor_note = "参数里的门槛下限仍是对数收益口径，进入每段时按当时收盘价换成价格差。表里的实际门槛中位数已经是价格差。"
    published = [value for value in volatility if value is not None]
    rows = []
    for scale in series.params.scales:
        used = [
            float(item["threshold"])
            for item in pivots
            if item.get("scale") == scale.name and item.get("threshold") is not None
        ]
        rows.append(
            {
                "scale": scale.name,
                "k": scale.k,
                "delta_min": scale.delta_min,
                "max_span": scale.max_span,
                "volatility_median": _median(published),
                "threshold_median": _median(used),
                "unit": unit,
                "delta_min_note": floor_note,
            }
        )
    return {"scales": rows, "unit": unit}


def _full_relations(boundaries: Sequence[dict], zones: Sequence[dict], segments: Sequence[dict], lifecycle) -> dict:
    relations = build_relations(boundaries, zones, segments)
    relations["overlaps"] = [item.to_dict() for item in lifecycle.overlaps]
    relations["overlap_note"] = OVERLAP_NOTE
    return relations


def _range_crosses(start: int, end: int, gaps: Sequence[int]) -> bool:
    return any(start < gap <= end for gap in gaps)


def _median(values: Sequence[float]) -> Optional[float]:
    nums = sorted(float(value) for value in values)
    count = len(nums)
    if count == 0:
        return None
    mid = count // 2
    if count % 2:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2.0


def _date_text(value) -> Optional[str]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _parse_user_time(text: str) -> datetime:
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"无法解析 as-of 时间 {text!r}") from exc
