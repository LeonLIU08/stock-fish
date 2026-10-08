"""判断方法的契约：对象外壳、图元、表格和一次运行的结果。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

MARK_KINDS = (
    "label_bar",
    "band",
    "segment",
    "polyline",
    "step_line",
    "marker",
    "text",
)
BASE_INPUTS = frozenset(
    {"closes", "pivots", "volatility", "segments", "boundaries", "zones", "events"}
)
KIND_RUN = "run"
KIND_INSTANCE = "instance"
RUN_STATUSES = ("completed", "skipped", "failed")


@dataclass(frozen=True)
class CategorySpec:
    key: str
    name: str
    color: str


@dataclass(frozen=True)
class ColumnSpec:
    """页面表格的一列。path 用点号取值，-1 表示列表最后一项。"""

    name: str
    path: str
    format: str
    hover: bool = False


@dataclass(frozen=True)
class TableSpec:
    key: str
    title: str
    columns: Tuple[ColumnSpec, ...]
    appendix: bool = False
    source: str = "objects"


@dataclass(frozen=True)
class MethodSample:
    symbol: str
    title: str
    blurb: str
    levels: Tuple[float, ...]
    price_axis: str = "log"


@dataclass
class MethodResult:
    objects: List[dict] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)
    extra: Dict[str, List[dict]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def pivot_ref(pivot: Any) -> Dict[str, Any]:
    return {
        "scale": pivot.scale,
        "role": pivot.role,
        "extreme_index": int(pivot.extreme_index),
    }


def make_mark(
    *,
    kind: str,
    object_id: str,
    method: str,
    scale: str,
    category: str,
    color: str,
    legend: str,
    start_index: int,
    end_index: int,
    available_index: int,
    points: Optional[Sequence[Sequence[float]]] = None,
    hover: Optional[Sequence[str]] = None,
    line_type: str = "solid",
    show: str = "always",
    text: str = "",
    anchor: str = "price",
    index: Optional[int] = None,
    price: Optional[float] = None,
) -> dict:
    if kind not in MARK_KINDS:
        raise ValueError(f"未知图元 {kind!r}")
    return {
        "kind": kind,
        "object_id": object_id,
        "method": method,
        "scale": scale,
        "category": category,
        "color": color,
        "legend": legend,
        "start_index": int(start_index),
        "end_index": int(end_index),
        "available_index": int(available_index),
        "points": [list(point) for point in points] if points else [],
        "hover": list(hover or []),
        "line_type": line_type,
        "show": show,
        "text": text,
        "anchor": anchor,
        "index": index,
        "price": price,
    }


def lookup_path(record: Any, path: str) -> Any:
    current = record
    for part in str(path).split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            if part == "-1":
                return None
            current = current.get(part)
            continue
        if isinstance(current, list):
            if not current:
                return None
            if part == "-1":
                current = current[-1]
            elif part.isdigit():
                index = int(part)
                if index >= len(current):
                    return None
                current = current[index]
            else:
                return None
            continue
        return None
    return current
