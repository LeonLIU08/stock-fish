"""把快照换成价格坐标上的点列。

只读快照里已经写好的斜率、截距和收盘价。这里不检测拐点，也不重新拟合。
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable, List, Optional, Sequence

from analysis.structure.config import PRICE_AXIS_LOG, SCALE_NAMES

READING_NOTE = (
    "分数是当前参数下的拟合质量；突破是收盘价相对边界的状态变化；"
    "本页面不给出成交价格和持仓建议。"
)
SCALE_DEFINITION_NOTE = (
    "短、中、长三个尺度由参数包里的反转倍数 k、门槛下限和最大序号跨度定义。"
    "请求的起止日期只决定送入引擎的已完成 K 线，不改写这些定义。"
)
OVERLAP_NOTE = "不合并为一条证据"

SCALE_LABEL = {"short": "短", "mid": "中", "long": "长"}
SCALE_CHART_LABEL = {"short": "短尺度", "mid": "中尺度", "long": "长尺度"}
ROLE_LABEL = {
    "support": "支撑",
    "resistance": "压力",
    "high": "高点",
    "low": "低点",
    "pending": "待定",
}
STATUS_LABEL = {
    "validated": "已验证",
    "candidate": "候选",
    "rejected": "拒绝",
    "broken": "已被收盘价突破",
    "expired": "过期",
}
ZONE_LABEL = {
    "channel": "通道",
    "convergence": "收敛区间",
    "sideways": "横向区间",
    "other": "其他边界对",
}
EVENT_LABEL = {
    "validated": "已验证",
    "breakout": "突破",
    "expired": "过期",
}
DIRECTION_LABEL = {"up": "上升", "down": "下降", "flat": "走平"}
CONSTRAINT_LABEL = {
    "two_pivots": "至少两个同类拐点",
    "slope_sign": "斜率符号",
    "hard_break": "硬破坏深度",
    "no_skipped_break": "没有跳过破坏拐点",
    "min_touch_clusters": "最少接触簇",
    "positive_width": "宽度为正",
    "lower_touches": "下侧接触",
    "upper_touches": "上侧接触",
    "label_geometry": "几何标签",
}
AXIS_LABEL = {PRICE_AXIS_LOG: "对数价格", "uniform": "均匀价格"}
MAIN_ZONE_LABELS = {"channel", "convergence", "sideways"}
_EVENT_STATUS = {"validated": "validated", "breakout": "broken", "expired": "expired"}
_DRAWABLE_STATUS = {"validated", "broken", "expired"}

SCALE_STYLE = {
    "short": {"width": 1.25, "type": "dotted"},
    "mid": {"width": 2.5, "type": "solid"},
    "long": {"width": 1.7, "type": "dashed"},
}
COLORS = {
    "close": "#E6E1D6",
    "support": "#7EB6FF",
    "resistance": "#F0A05A",
    "high": "#F07167",
    "low": "#3DDC97",
    "pending": "#B7B1A6",
    "channel": "rgba(126, 182, 255, 0.16)",
    "convergence": "rgba(196, 148, 255, 0.18)",
    "sideways": "rgba(240, 176, 96, 0.16)",
    "channel_stroke": "rgba(126, 182, 255, 0.85)",
    "convergence_stroke": "rgba(196, 148, 255, 0.9)",
    "sideways_stroke": "rgba(240, 176, 96, 0.9)",
    "breakout": "#FF5C7A",
    "segment": "rgba(230, 225, 214, 0.45)",
    "focus": "#F3D58A",
    "as_of": "#E0A45A",
}


def price_on_line(slope: float, intercept: float, index: int, price_axis: str) -> float:
    """把几何直线上的一点换回实际股价。对数轴先取指数。"""
    geometric = float(slope) * int(index) + float(intercept)
    if price_axis == PRICE_AXIS_LOG:
        return math.exp(geometric)
    return geometric


def price_axis_distance(left: float, right: float, price_axis: str) -> float:
    """纵轴上的距离。对数轴比的是倍数，均匀轴比的是价差。"""
    if price_axis == PRICE_AXIS_LOG:
        if left <= 0 or right <= 0:
            raise ValueError("对数轴距离需要正价格")
        return abs(math.log(float(right)) - math.log(float(left)))
    return abs(float(right) - float(left))


def pivot_segment_stats(
    pivots: Sequence[dict],
    temporary: Sequence[dict],
    segments: Sequence[dict],
    price_axis: str,
) -> List[dict]:
    """每个尺度一行。临时拐点单独计数，不进高点和低点。"""
    unit = "对数收益" if price_axis == PRICE_AXIS_LOG else "价格差"
    rows = []
    for scale in SCALE_NAMES:
        confirmed = [item for item in pivots if item.get("scale") == scale]
        delays = [
            int(item["confirm_index"]) - int(item["extreme_index"])
            for item in confirmed
            if item.get("confirm_index") is not None and item.get("extreme_index") is not None
        ]
        own_segments = [item for item in segments if item.get("scale") == scale]
        spans = [int(item["bar_span"]) for item in own_segments if item.get("bar_span") is not None]
        changes = [abs(float(item["change"])) for item in own_segments if item.get("change") is not None]
        rows.append(
            {
                "scale": scale,
                "high_count": sum(1 for item in confirmed if item.get("role") == "high"),
                "low_count": sum(1 for item in confirmed if item.get("role") == "low"),
                "temporary_count": sum(1 for item in temporary if item.get("scale") == scale),
                "confirm_delay_median": _median(delays),
                "confirm_delay_max": max(delays) if delays else None,
                "segment_count": len(own_segments),
                "span_median": _median(spans),
                "abs_change_median": _median(changes),
                "change_unit": unit,
            }
        )
    return rows


def display_snapshot(snapshot: dict) -> dict:
    """页面上该看的那一份。

    `as_of` 只拿掉当时还不可用的对象，并把状态退回到当时最后一条事件。
    收盘价和波动统计仍覆盖全部已加载 K 线。
    """
    limit = _as_of_limit(snapshot)
    events = [event for event in _list(snapshot, "events") if _visible(event, limit)]

    def adapt(record: dict) -> Optional[dict]:
        if not _visible(record, limit):
            return None
        status = _status_at(record, events, limit)
        if status == record.get("status"):
            return record
        return _with_status(record, status)

    pivots = _adapt(_list(snapshot, "pivots"), adapt)
    boundaries = _adapt(_list(snapshot, "boundaries"), adapt)
    zones = _adapt(_list(snapshot, "zones"), adapt)
    segments = _adapt(_list(snapshot, "segments"), adapt)
    if _show_temporary(snapshot, limit):
        temporary_pivots = list(_list(snapshot, "temporary_pivots"))
        temporary_segments = list(_list(snapshot, "temporary_segments"))
    else:
        temporary_pivots = []
        temporary_segments = []

    shown = dict(snapshot)
    shown["pivots"] = pivots
    shown["temporary_pivots"] = temporary_pivots
    shown["segments"] = segments
    shown["temporary_segments"] = temporary_segments
    shown["boundaries"] = boundaries
    shown["zones"] = zones
    shown["events"] = events
    shown["relations"] = _relations(snapshot, boundaries, zones, segments)
    shown["pivot_stats"] = pivot_segment_stats(
        pivots,
        temporary_pivots,
        segments,
        _price_axis(snapshot),
    )
    return shown


def build_view_model(snapshot: dict) -> dict:
    """从快照得到图上的点。端点价格是实际股价，一条边界只有两个端点。"""
    shown = display_snapshot(snapshot)
    price_axis = _price_axis(snapshot)
    closes = [_close_point(bar, _interval(snapshot)) for bar in _list(snapshot, "closes")]
    stop = _as_of_stop_index(snapshot)
    boundaries = []
    for item in shown["boundaries"]:
        drawn = _boundary_view(item, shown["events"], price_axis, closes, stop)
        if drawn is not None:
            boundaries.append(drawn)
    zones = []
    for item in shown["zones"]:
        drawn = _zone_view(item, shown["events"], price_axis, stop)
        if drawn is not None:
            zones.append(drawn)
    return {
        "price_axis": price_axis,
        "axis_label": AXIS_LABEL.get(price_axis, price_axis),
        "default_scales": list(_visible_scales(snapshot)),
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
        "pivots": [_pivot_view(item, confirmed=True) for item in shown["pivots"]]
        + [_pivot_view(item, confirmed=False) for item in shown["temporary_pivots"]],
        "boundaries": boundaries,
        "zones": zones,
        "segments": [_segment_view(item) for item in shown["segments"] if _segment_view(item)],
        "as_of_index": stop,
    }


def build_relations(boundaries: Sequence[dict], zones: Sequence[dict], segments: Sequence[dict]) -> dict:
    """按主结果和备选标识归组。不同尺度的重叠沿用区间上已经写好的说明。"""
    return _relations({"relations": {"overlaps": [], "overlap_note": OVERLAP_NOTE}}, boundaries, zones, segments)


def _relations(snapshot: dict, boundaries: Sequence[dict], zones: Sequence[dict], segments: Sequence[dict]) -> dict:
    stored = (snapshot.get("relations") or {}) if isinstance(snapshot, dict) else {}
    visible_zones = {item.get("id") for item in zones}
    overlaps = []
    for item in stored.get("overlaps") or []:
        ids = item.get("zone_ids") or []
        if ids and all(zone_id in visible_zones for zone_id in ids):
            overlaps.append(item)
    visible_segments = {item.get("id") for item in segments}
    contains = []
    for item in segments:
        kept = [name for name in item.get("contains") or [] if name in visible_segments]
        if kept:
            contains.append({"id": item.get("id"), "scale": item.get("scale"), "contains": kept})
    return {
        "boundary_groups": _groups(boundaries, "role"),
        "zone_groups": _groups(zones, "label"),
        "overlaps": overlaps,
        "segment_contains": contains,
        "overlap_note": stored.get("overlap_note") or OVERLAP_NOTE,
    }


def _groups(records: Sequence[dict], kind_key: str) -> List[dict]:
    groups = []
    for primary in records:
        if not primary.get("primary"):
            continue
        alternates = [item.get("id") for item in records if item.get("alternate_of") == primary.get("id")]
        if not alternates:
            continue
        groups.append(
            {
                "primary_id": primary.get("id"),
                "alternate_ids": alternates,
                "scale": primary.get("scale"),
                kind_key: primary.get(kind_key),
            }
        )
    return groups


def _boundary_view(record: dict, events: Sequence[dict], price_axis: str, closes: Sequence[dict], stop: Optional[int]) -> Optional[dict]:
    if not record.get("primary", True):
        return None
    status = record.get("status")
    if status not in _DRAWABLE_STATUS:
        return None
    if record.get("slope") is None or record.get("intercept") is None:
        return None
    start = int(record["start_index"])
    end = int(record["end_index"])
    breakout = _breakout(record, events) if status == "broken" else None
    solid_end = end
    projection_end = None
    if breakout is not None:
        break_index = int(breakout["index"])
        if break_index < solid_end:
            solid_end = break_index
        elif break_index > end:
            projection_end = break_index
    elif status == "validated":
        horizon = _forward_end(closes, end, stop)
        if horizon > end:
            projection_end = horizon
    if solid_end < start:
        return None
    return {
        "id": record.get("id"),
        "scale": record.get("scale"),
        "role": record.get("role"),
        "status": status,
        "layer": "expired" if status == "expired" else "current",
        "revision": record.get("revision"),
        "normalized_slope": record.get("normalized_slope"),
        "slope": record.get("slope"),
        "intercept": record.get("intercept"),
        "solid": _endpoints(record, start, solid_end, price_axis),
        "projection": None if projection_end is None else _endpoints(record, end, projection_end, price_axis),
        "breakout": breakout,
        "start_time": record.get("start_time"),
        "end_time": record.get("end_time"),
        "available_time": record.get("available_time"),
    }


def zone_drawn(record: dict) -> bool:
    """主图画出全部已成立的横向、通道和收敛。突破和过期仍保留。"""
    if not record.get("primary", True):
        return False
    if record.get("label") not in MAIN_ZONE_LABELS:
        return False
    return record.get("status") in _DRAWABLE_STATUS


def _zone_view(record: dict, events: Sequence[dict], price_axis: str, stop: Optional[int]) -> Optional[dict]:
    if not zone_drawn(record):
        return None
    status = record.get("status")
    if record.get("lower_slope") is None or record.get("upper_slope") is None:
        return None
    start = int(record["effective_start"])
    end = int(record["effective_end"])
    breakout = _breakout(record, events) if status == "broken" else None
    if breakout is not None:
        end = min(end, int(breakout["index"]))
    if stop is not None and status == "validated":
        end = min(end, stop) if end > start else end
    if end < start:
        return None
    return {
        "id": record.get("id"),
        "scale": record.get("scale"),
        "label": record.get("label"),
        "status": status,
        "layer": "expired" if status == "expired" else "current",
        "revision": record.get("revision"),
        "lower_id": record.get("lower_id"),
        "upper_id": record.get("upper_id"),
        "lower": _rail(record, "lower", start, end, price_axis),
        "upper": _rail(record, "upper", start, end, price_axis),
        "breakout": breakout,
        "available_time": record.get("available_time"),
    }


def _endpoints(record: dict, start: int, end: int, price_axis: str) -> List[List[float]]:
    slope = record["slope"]
    intercept = record["intercept"]
    return [
        [start, price_on_line(slope, intercept, start, price_axis)],
        [end, price_on_line(slope, intercept, end, price_axis)],
    ]


def _rail(record: dict, side: str, start: int, end: int, price_axis: str) -> List[List[float]]:
    slope = record[f"{side}_slope"]
    intercept = record[f"{side}_intercept"]
    return [
        [start, price_on_line(slope, intercept, start, price_axis)],
        [end, price_on_line(slope, intercept, end, price_axis)],
    ]


def _breakout(record: dict, events: Sequence[dict]) -> Optional[dict]:
    found = None
    for event in events:
        if event.get("object_id") != record.get("id") or event.get("event_type") != "breakout":
            continue
        found = event
    if found is None or found.get("event_index") is None:
        return None
    return {
        "index": int(found["event_index"]),
        "price": found.get("reference_close"),
        "time": found.get("available_time"),
        "price_cross_index": found.get("price_cross_index"),
    }


def _pivot_view(record: dict, confirmed: bool) -> dict:
    kind = "pivot" if confirmed else "temporary"
    return {
        "id": f"{kind}:{record.get('scale')}:{record.get('role')}:{record.get('extreme_index')}",
        "index": record.get("extreme_index"),
        "price": record.get("price"),
        "scale": record.get("scale"),
        "role": record.get("role"),
        "confirmed": confirmed,
        "extreme_time": record.get("extreme_time"),
        "confirm_time": record.get("confirm_time") if confirmed else None,
        "available_time": record.get("available_time") if confirmed else None,
        "threshold": record.get("threshold"),
        "volatility": record.get("volatility"),
    }


def _segment_view(record: dict) -> Optional[dict]:
    start = record.get("start") or {}
    end = record.get("end") or {}
    if start.get("extreme_index") is None or end.get("extreme_index") is None:
        return None
    if start.get("price") is None or end.get("price") is None:
        return None
    return {
        "id": record.get("id"),
        "scale": record.get("scale"),
        "direction": record.get("direction"),
        "points": [
            [int(start["extreme_index"]), float(start["price"])],
            [int(end["extreme_index"]), float(end["price"])],
        ],
    }


def _forward_end(closes: Sequence[dict], end_index: int, stop: Optional[int]) -> int:
    limit = int(end_index)
    for bar in closes:
        index = int(bar["index"])
        if index <= end_index:
            continue
        if bar.get("gap_before"):
            break
        if stop is not None and index > stop:
            break
        limit = index
    return limit


def _close_point(bar: dict, interval: str) -> dict:
    time_text = str(bar.get("time") or "")
    return {
        "index": int(bar["index"]),
        "time": time_text,
        "label": _tick_label(time_text, interval),
        "close": float(bar["close"]),
    }


def _tick_label(time_text: str, interval: str) -> str:
    if "T" not in time_text:
        return time_text[:10]
    day, clock = time_text.split("T", 1)
    clock = clock.split("+", 1)[0].replace("Z", "")
    if interval == "1d" or clock.startswith("00:00:00"):
        return day
    return f"{day[5:]} {clock[:5]}"


def _as_of_limit(snapshot: dict) -> Optional[datetime]:
    identity = snapshot.get("identity") or {}
    return _parse_time(identity.get("as_of"))


def _as_of_stop_index(snapshot: dict) -> Optional[int]:
    limit = _as_of_limit(snapshot)
    if limit is None:
        return None
    last = None
    for bar in _list(snapshot, "closes"):
        when = _parse_time(bar.get("time"))
        if when is not None and _not_after(when, limit):
            last = int(bar["index"])
    return last


def _show_temporary(snapshot: dict, limit: Optional[datetime]) -> bool:
    if limit is None:
        return True
    closes = _list(snapshot, "closes")
    if not closes:
        return False
    last = _parse_time(closes[-1].get("time"))
    return last is not None and _not_after(last, limit)


def _visible(record: dict, limit: Optional[datetime]) -> bool:
    if limit is None:
        return True
    available = _parse_time(record.get("available_time"))
    if available is None:
        return False
    return _not_after(available, limit)


def _status_at(record: dict, events: Sequence[dict], limit: Optional[datetime]) -> Optional[str]:
    if limit is None:
        return record.get("status")
    relevant = [event for event in events if event.get("object_id") == record.get("id")]
    if not relevant:
        return record.get("status")
    return _EVENT_STATUS.get(relevant[-1].get("event_type"), record.get("status"))


def _with_status(record: dict, status: Optional[str]) -> dict:
    updated = dict(record)
    updated["status"] = status
    if "label" in record:
        eligible = record["main_chart"] if "main_chart" in record else bool(record.get("draw_on_main"))
        updated["draw_on_main"] = bool(eligible and status == "validated" and record.get("label") in MAIN_ZONE_LABELS)
    return updated


def _adapt(records: Sequence[dict], adapt) -> List[dict]:
    adapted = []
    for record in records:
        item = adapt(record)
        if item is not None:
            adapted.append(item)
    return adapted


def _list(snapshot: dict, key: str) -> Sequence[dict]:
    value = snapshot.get(key) or []
    return value if isinstance(value, list) else list(value)


def _price_axis(snapshot: dict) -> str:
    identity = snapshot.get("identity") or {}
    return identity.get("price_axis") or PRICE_AXIS_LOG


def _interval(snapshot: dict) -> str:
    identity = snapshot.get("identity") or {}
    return identity.get("interval") or "1d"


def _visible_scales(snapshot: dict) -> Sequence[str]:
    identity = snapshot.get("identity") or {}
    scales = identity.get("visible_scales") or ["mid"]
    return [scale for scale in scales if scale in SCALE_NAMES] or ["mid"]


def _parse_time(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _not_after(left: datetime, right: datetime) -> bool:
    if (left.tzinfo is None) != (right.tzinfo is None):
        if left.tzinfo is None:
            left = left.replace(tzinfo=right.tzinfo)
        else:
            right = right.replace(tzinfo=left.tzinfo)
    return left <= right


def _median(values: Iterable[float]) -> Optional[float]:
    nums = sorted(float(value) for value in values)
    count = len(nums)
    if count == 0:
        return None
    mid = count // 2
    if count % 2:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2.0
