"""把 Alpha158 算好后按月落盘。滚动训练只读取用到的月份。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_baseline import cache_dir, limit_instruments, universe_expression


def panel_dir(freeze_dirname: str) -> Path:
    return cache_dir(freeze_dirname) / "alpha158_panel"


def _signature(cfg: dict) -> dict:
    segments = cfg["segments"]
    return {
        "label": cfg["label"]["expr"],
        "column": cfg["label"]["column"],
        "universe": universe_expression(cfg["universe"]),
        "start": str(segments["warmup"]["start"])[:10],
        "end": str(segments["test"]["end"])[:10],
        "instruments": "tradable",
    }


def _read_meta(cfg: dict) -> dict | None:
    root = panel_dir(cfg["data"]["freeze_dirname"])
    meta_path = root / "meta.json"
    if not meta_path.exists() or not any(root.glob("*.parquet")):
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def panel_ready(cfg: dict) -> bool:
    meta = _read_meta(cfg)
    return bool(meta) and meta.get("signature") == _signature(cfg)


def extendable_end(cfg: dict) -> str | None:
    """标签和股票池没变、只是测试终点往后延时，返回已有缓存的最后一天。"""
    meta = _read_meta(cfg)
    if not meta:
        return None
    old = meta.get("signature") or {}
    new = _signature(cfg)
    if any(old.get(key) != value for key, value in new.items() if key != "end"):
        return None
    if "end" not in old or pd.Timestamp(new["end"]) <= pd.Timestamp(old["end"]):
        return None
    return str(old["end"])[:10]


def _flatten(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[list[str]]]:
    columns = [[str(group), str(name)] for group, name in frame.columns]
    flat = frame.copy()
    flat.columns = [f"{group}__{name}" for group, name in columns]
    flat = flat.reset_index()
    return flat, columns


def _restore(flat: pd.DataFrame, columns: list[list[str]]) -> pd.DataFrame:
    names = [f"{group}__{name}" for group, name in columns]
    frame = flat.set_index(["datetime", "instrument"])[names]
    frame.columns = pd.MultiIndex.from_tuples([tuple(item) for item in columns])
    return frame.sort_index()


def write_panel(frame: pd.DataFrame, cfg: dict) -> Path:
    root = panel_dir(cfg["data"]["freeze_dirname"])
    root.mkdir(parents=True, exist_ok=True)
    flat, columns = _flatten(frame)
    flat["datetime"] = pd.to_datetime(flat["datetime"])
    flat["month"] = flat["datetime"].dt.to_period("M").astype(str)
    for month, part in flat.groupby("month", sort=True):
        part.drop(columns=["month"]).to_parquet(root / f"{month}.parquet", index=False)
    meta = {"signature": _signature(cfg), "columns": columns, "rows": int(len(flat))}
    (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return root


def append_panel(frame: pd.DataFrame, cfg: dict, after: str) -> Path:
    """只把已有缓存最后一天之后的月份写进去。更早的月份保持原文件。"""
    root = panel_dir(cfg["data"]["freeze_dirname"])
    meta = _read_meta(cfg)
    if meta is None:
        raise RuntimeError(f"没有可续写的 Alpha158 缓存: {root}")
    flat, columns = _flatten(frame)
    if columns != meta["columns"]:
        raise RuntimeError("新计算的 Alpha158 列和已有缓存不一致")
    flat["datetime"] = pd.to_datetime(flat["datetime"])
    flat = flat[flat["datetime"] > pd.Timestamp(after)]
    if flat.empty:
        raise RuntimeError(f"{after} 之后没有新的 Alpha158 行")
    flat["month"] = flat["datetime"].dt.to_period("M").astype(str)
    for month, part in flat.groupby("month", sort=True):
        part.drop(columns=["month"]).to_parquet(root / f"{month}.parquet", index=False)
    meta["signature"] = _signature(cfg)
    meta["rows"] = int(meta.get("rows") or 0) + int(len(flat))
    (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return root


def load_panel(cfg: dict, start: str, end: str) -> pd.DataFrame:
    root = panel_dir(cfg["data"]["freeze_dirname"])
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    start_month = pd.Timestamp(start).to_period("M")
    end_month = pd.Timestamp(end).to_period("M")
    files = []
    for path in sorted(root.glob("*.parquet")):
        month = pd.Period(path.stem, freq="M")
        if start_month <= month <= end_month:
            files.append(path)
    if not files:
        raise FileNotFoundError(f"没有 {start} ~ {end} 的 Alpha158 缓存: {root}")
    flat = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    flat["datetime"] = pd.to_datetime(flat["datetime"])
    flat = flat[(flat["datetime"] >= pd.Timestamp(start)) & (flat["datetime"] <= pd.Timestamp(end))]
    # 特征按股票单独计算，缓存可以比名单大。读入后立刻丢掉名单外的行，
    # 后面的标签标准化、训练和选股都只看见这份名单。
    return limit_instruments(_restore(flat, meta["columns"]), cfg)


class PanelDataset:
    """从已落盘的特征里按日期切片。学习集在切片上做截面标准化。"""

    def __init__(self, frame: pd.DataFrame, segments: dict):
        self._frame = frame
        self.segments = {
            key: (pd.Timestamp(value[0]), pd.Timestamp(value[1])) for key, value in segments.items()
        }

    def _slice(self, segment: str) -> pd.DataFrame:
        start, end = self.segments[segment]
        dates = self._frame.index.get_level_values("datetime")
        return self._frame.loc[(dates >= start) & (dates <= end)].copy()

    def _learn(self, frame: pd.DataFrame) -> pd.DataFrame:
        from qlib.data.dataset.processor import CSZScoreNorm, DropnaLabel

        from processors import DropInvalidLabel

        frame = DropnaLabel()(frame)
        frame = CSZScoreNorm(fields_group="label")(frame)
        return DropInvalidLabel()(frame)

    def prepare(self, segments, col_set=None, data_key=None):
        from qlib.data.dataset.handler import DataHandlerLP

        if isinstance(segments, (list, tuple)):
            return [self.prepare(segment, col_set=col_set, data_key=data_key) for segment in segments]
        frame = self._slice(segments)
        if data_key == DataHandlerLP.DK_L:
            frame = self._learn(frame)
        if col_set is None or isinstance(col_set, (list, tuple)):
            return frame
        return frame[col_set]
