"""
AnalysisStore — 每次分析结果的永久归档

目录: memory/analysis/{symbol}/{timestamp}/
  - state.json
  - employee_reports.json  (大师模式)
  - cio_decision.json      (大师模式)
  - prediction.json
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from memory import ANALYSIS_DIR


def _safe_timestamp(ts: str) -> str:
    """把 ISO 时间戳变成目录名安全字符串。"""
    cleaned = re.sub(r"[^0-9T\-]", "-", str(ts).replace(":", "-"))
    return cleaned.strip("-") or "unknown"


class AnalysisStore:
    """分析结果归档。无进程内状态，可随时 new 一个实例。"""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else ANALYSIS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def save(
        self,
        symbol: str,
        state: Dict[str, Any],
        timestamp: str,
        employee_reports: Any = None,
        cio_decision: Any = None,
        prediction: Any = None,
    ) -> Path:
        run_dir = self.base_dir / symbol / _safe_timestamp(timestamp)
        with self._lock:
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(run_dir / "state.json", state)
            if employee_reports is not None:
                self._write_json(run_dir / "employee_reports.json", employee_reports)
            if cio_decision is not None:
                self._write_json(run_dir / "cio_decision.json", cio_decision)
            if prediction is not None:
                self._write_json(run_dir / "prediction.json", prediction)
        logger.debug(f"AnalysisStore 已归档 {symbol} @ {timestamp}")
        return run_dir

    def list_all_symbols(self) -> List[str]:
        if not self.base_dir.exists():
            return []
        return sorted(
            d.name
            for d in self.base_dir.iterdir()
            if d.is_dir() and not d.name.startswith("__")
        )

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, default=str, indent=2)
        tmp.rename(path)
