"""方法注册表。启动时检查键、原料和环；运行时一个方法失败不影响其他方法。"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from analysis.structure.config import StructureParams
from analysis.structure.context import StructureContext
from analysis.structure.methods.base import BASE_INPUTS, RUN_STATUSES, MethodResult


def registered_methods() -> tuple:
    """按登记顺序返回方法。延迟导入，避免和快照、页面互相在模块顶层缠住。"""
    from analysis.structure.methods.direction import DirectionMethod

    return (DirectionMethod(),)


def validate_registry(methods: Sequence[object]) -> List[object]:
    """键不重复、原料找得到、没有环。返回按依赖排好的顺序，无依赖的保持登记顺序。"""
    by_key: Dict[str, object] = {}
    rank: Dict[str, int] = {}
    for index, method in enumerate(methods):
        key = method.key
        if key in by_key:
            raise ValueError(f"方法键重复: {key}")
        if key in BASE_INPUTS:
            raise ValueError(f"方法键 {key} 与基础原料重名")
        by_key[key] = method
        rank[key] = index
    graph: Dict[str, List[str]] = {}
    for method in methods:
        deps = []
        for requirement in method.requires:
            if requirement in BASE_INPUTS:
                continue
            if requirement not in by_key:
                raise ValueError(f"{method.key} 的原料 {requirement} 找不到")
            deps.append(requirement)
        graph[method.key] = deps
    color: Dict[str, int] = {}

    def visit(key: str) -> None:
        color[key] = 1
        for dep in graph[key]:
            state = color.get(dep, 0)
            if state == 1:
                raise ValueError(f"方法依赖成环: {key} → {dep}")
            if state != 2:
                visit(dep)
        color[key] = 2

    for method in methods:
        if color.get(method.key) != 2:
            visit(method.key)

    remaining = {method.key: method for method in methods}
    ordered: List[object] = []
    while remaining:
        ready = [
            key
            for key in remaining
            if all(dep not in remaining for dep in graph[key])
        ]
        if not ready:
            raise ValueError("方法依赖成环")
        ready.sort(key=lambda key: rank[key])
        ordered.append(remaining.pop(ready[0]))
    return ordered


def method_param_dict(params: StructureParams, method: object) -> dict:
    raw = getattr(params, method.key, None)
    if raw is None:
        return method.default_params()
    if hasattr(raw, "to_dict"):
        return raw.to_dict()
    return dict(raw)


def run_methods(
    context: StructureContext,
    params: StructureParams,
    enabled: Sequence[str],
    methods: Optional[Sequence[object]] = None,
) -> dict:
    """按依赖运行启用的方法。失败的写明原因，依赖它的标为跳过。"""
    chosen = tuple(methods) if methods is not None else registered_methods()
    ordered = validate_registry(chosen)
    enabled_set = set(enabled)
    unknown = sorted(enabled_set.difference(method.key for method in ordered))
    if unknown:
        raise ValueError(f"参数版本启用了未登记的方法: {', '.join(unknown)}")
    records: Dict[str, dict] = {}
    events: List[dict] = []
    failed = set()
    for method in ordered:
        if method.key not in enabled_set:
            continue
        blocked = [item for item in method.requires if item in failed]
        if blocked:
            records[method.key] = _record(
                method,
                {},
                "skipped",
                f"依赖的 {blocked[0]} 失败，跳过",
            )
            failed.add(method.key)
            continue
        pack = method_param_dict(params, method)
        try:
            method.check_params(pack)
            upstream = {
                item: records[item]
                for item in method.requires
                if item in records
            }
            result = method.judge(context, pack, upstream)
            if not isinstance(result, MethodResult):
                raise TypeError(f"{method.key} 没有返回方法结果")
            records[method.key] = _record(method, pack, "completed", "", result)
            events.extend(result.events)
        except Exception as exc:
            reason = str(exc) or exc.__class__.__name__
            records[method.key] = _record(method, pack, "failed", reason)
            failed.add(method.key)
    return {"records": records, "events": events, "failed": failed}


def merge_events(base: Sequence[dict], extra: Sequence[dict], method_order: Sequence[str]) -> List[dict]:
    """同一根上先排第一版事件，再按方法登记顺序，同一方法内按对象标识。"""
    order = {key: index for index, key in enumerate(method_order)}
    decorated = []
    for index, event in enumerate(base):
        decorated.append((int(event.get("available_index") or 0), 0, "", index, event))
    for index, event in enumerate(extra):
        kind = str(event.get("object_kind") or "")
        group = 1 + order.get(kind, len(order))
        decorated.append(
            (
                int(event.get("available_index") or 0),
                group,
                str(event.get("object_id") or ""),
                index,
                event,
            )
        )
    decorated.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [item[-1] for item in decorated]


def method_failure_text(snapshot: dict) -> str:
    """失败和因此被跳过的方法。没有问题就返回空字符串。"""
    lines = []
    methods = snapshot.get("methods") or {}
    for key, record in methods.items():
        status = record.get("status")
        if status not in ("failed", "skipped"):
            continue
        name = record.get("name") or key
        reason = record.get("reason") or status
        if status == "failed":
            lines.append(f"{name}失败：{reason}")
        else:
            lines.append(f"{name}跳过：{reason}")
    return "；".join(lines)


def _record(method: object, pack: dict, status: str, reason: str, result: Optional[MethodResult] = None) -> dict:
    if status not in RUN_STATUSES:
        raise ValueError(f"未知运行结果 {status}")
    result = result or MethodResult()
    return {
        "key": method.key,
        "name": method.name,
        "version": method.version,
        "kind": method.kind,
        "section": method.section,
        "requires": list(method.requires),
        "categories": [
            {"key": item.key, "name": item.name, "color": item.color}
            for item in method.categories
        ],
        "replaces": list(method.replaces),
        "params": pack,
        "status": status,
        "reason": reason,
        "objects": list(result.objects),
        "extra": {key: list(rows) for key, rows in result.extra.items()},
        "notes": list(result.notes),
    }
