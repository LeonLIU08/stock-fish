"""港股每手股数。只给回测下单用，不进入训练样本。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def floor_to_lot(amount: float, lot: int) -> float:
    """股数向下取整到整手。不足一手则为 0。"""
    if lot <= 1:
        return float(amount)
    if amount <= 0:
        return 0.0
    return float(int(amount + 1e-6) // int(lot) * int(lot))


def cache_path(freeze_dirname: str) -> Path:
    from run_baseline import cache_dir

    return cache_dir(freeze_dirname) / "lot_sizes.csv"


def _read_cache(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype={"code": str})
    found = {}
    for code, lot in zip(frame["code"], frame["lot_size"]):
        text = str(code).strip().zfill(5)
        if not text.isdigit():
            continue
        size = int(lot)
        if size >= 1:
            found[text] = size
    return found


def _write_cache(path: Path, lots: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"code": code, "lot_size": lots[code]} for code in sorted(lots)]
    pd.DataFrame(rows).to_csv(path, index=False)


def fetch_missing(codes: list[str]) -> dict[str, int]:
    from analysis.backtest.trend.lots import fetch_hk_lot_sizes

    found = fetch_hk_lot_sizes(codes)
    return {code: int(lot) for code, (lot, _name) in found.items() if int(lot) >= 1}


def load_lot_sizes(codes: list[str], freeze_dirname: str, default_lot: int = 100) -> tuple[dict[str, int], int]:
    """返回代码到每手股数，以及用默认手数补上的只数。"""
    wanted = []
    for code in codes:
        text = str(code).strip().upper().zfill(5)
        if text.isdigit() and text not in wanted:
            wanted.append(text)
    path = cache_path(freeze_dirname)
    cached = _read_cache(path)
    missing = [code for code in wanted if code not in cached]
    if missing:
        try:
            fetched = fetch_missing(missing)
        except Exception as exc:
            print(f"每手股数查询失败，缺的代码按 {default_lot} 股: {exc}", flush=True)
            fetched = {}
        if fetched:
            cached.update(fetched)
            _write_cache(path, cached)
            print(f"写入每手股数 {len(fetched)} 只 -> {path}", flush=True)
    lots = {}
    fallback = 0
    for code in wanted:
        if code in cached:
            lots[code] = cached[code]
        else:
            lots[code] = int(default_lot)
            fallback += 1
    return lots, fallback


def tradable_codes(config_path: Path) -> tuple[str, list[str]]:
    from run_baseline import load_config, provider_dir

    cfg = load_config(config_path)
    freeze = cfg["data"]["freeze_dirname"]
    inst = provider_dir(cfg) / "instruments" / "tradable.txt"
    codes = []
    for line in inst.read_text(encoding="utf-8").splitlines():
        if line.strip():
            codes.append(line.split("\t", 1)[0].strip())
    return freeze, codes


if __name__ == "__main__":
    config = ROOT / "qlib-zh" / "hk" / "baseline.yaml"
    freeze, codes = tradable_codes(config)
    lots, fallback = load_lot_sizes(codes, freeze)
    print(f"每手股数 {len(lots)} 只，其中 {fallback} 只用默认 100 股", flush=True)
