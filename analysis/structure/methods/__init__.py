"""趋势类别的判断方法。每种图形一个方法，共用取数、拐点和页面。"""

from analysis.structure.methods.registry import (
    merge_events,
    method_failure_text,
    registered_methods,
    run_methods,
    validate_registry,
)

__all__ = [
    "merge_events",
    "method_failure_text",
    "registered_methods",
    "run_methods",
    "validate_registry",
]
