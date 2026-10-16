"""不复权日线和除权事件。

Yahoo 的 auto_adjust=False 仍按拆股调整过价格和股息。这里用拆股记录乘回去，
得到当时的成交价和每股股息，再交给比例复权。拿不到这组成交价时直接失败，
不退回前复权。
"""
from __future__ import annotations

import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd

from analysis.backtest.bars import CACHE_DIR, to_yahoo_symbol
from analysis.backtest.trend.prices import CorporateAction

_COLUMNS = ("datetime", "open", "high", "low", "close", "volume")


def restore_traded_prices(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    dividends: Sequence[float],
    splits: Sequence[float],
) -> tuple:
    """把已经按拆股调整过的价格和股息乘回当时的成交口径。拆股日本身用拆后价格。"""
    count = len(closes)
    if not (len(opens) == len(highs) == len(lows) == len(dividends) == len(splits) == count):
        raise ValueError("价格、股息和拆股的长度必须一致")
    raw_open = [0.0] * count
    raw_high = [0.0] * count
    raw_low = [0.0] * count
    raw_close = [0.0] * count
    raw_dividend = [0.0] * count
    scale = 1.0
    for index in range(count - 1, -1, -1):
        raw_open[index] = float(opens[index]) * scale
        raw_high[index] = float(highs[index]) * scale
        raw_low[index] = float(lows[index]) * scale
        raw_close[index] = float(closes[index]) * scale
        raw_dividend[index] = float(dividends[index]) * scale
        split = float(splits[index] or 0.0)
        if split > 0 and split != 1:
            scale *= split
    return raw_open, raw_high, raw_low, raw_close, raw_dividend


def enclose_ohlc(frame: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """把最高价扩到能包住开盘价和收盘价，最低价收到能包住两者。

    非正价格不改，留给后面的检查跳过这只股票。返回新表和被改过的根数。
    """
    out = frame.copy()
    opened = out["open"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    close = out["close"].astype(float)
    usable = (opened > 0) & (high > 0) & (low > 0) & (close > 0)
    enclosed_high = pd.concat([high, opened, close], axis=1).max(axis=1)
    enclosed_low = pd.concat([low, opened, close], axis=1).min(axis=1)
    changed = usable & ((enclosed_high > high) | (enclosed_low < low))
    out.loc[usable, "high"] = enclosed_high[usable]
    out.loc[usable, "low"] = enclosed_low[usable]
    return out, int(changed.sum())


def actions_from_raw(dividends: Sequence[float], splits: Sequence[float]) -> Tuple[CorporateAction, ...]:
    actions = []
    for index, (dividend, split) in enumerate(zip(dividends, splits)):
        if dividend > 0:
            actions.append(CorporateAction(index, "dividend", float(dividend)))
        if split > 0 and split != 1:
            actions.append(CorporateAction(index, "split", float(split)))
    return tuple(actions)


def load_raw_bars(symbol: str, start: date, end: date) -> Tuple[pd.DataFrame, Tuple[CorporateAction, ...], str]:
    """取不复权 OHLCV 和除权事件。失败时说明原因，不改用前复权缓存。"""
    yahoo = to_yahoo_symbol(symbol)
    cache = CACHE_DIR / f"raw_{symbol}_{start.isoformat()}_{end.isoformat()}.pkl"
    cached = _read_cache(cache)
    if cached is not None:
        frame, actions = cached
        return frame, tuple(actions), f"cache:raw:{yahoo}"
    try:
        frame, actions = _download_yahoo(yahoo, start, end)
    except Exception as exc:
        raise RuntimeError(f"{symbol} 没有不复权 K 线和除权事件，不能做趋势回测：{exc}") from exc
    if frame.empty:
        raise RuntimeError(f"{symbol} 没有不复权 K 线和除权事件，不能做趋势回测")
    _write_cache(cache, (frame, actions))
    return frame, actions, f"yahoo:raw:{yahoo}"


def _download_yahoo(yahoo: str, start: date, end: date):
    import yfinance as yf

    history = yf.Ticker(yahoo).history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=True,
    )
    if history is None or history.empty:
        raise RuntimeError("Yahoo 没有返回日线")
    history = history.reset_index()
    if "Date" in history.columns:
        history = history.rename(columns={"Date": "datetime"})
    elif "Datetime" in history.columns:
        history = history.rename(columns={"Datetime": "datetime"})
    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column not in history.columns:
            raise RuntimeError(f"Yahoo 缺少 {column}")
    dividends = history["Dividends"] if "Dividends" in history.columns else 0.0
    splits = history["Stock Splits"] if "Stock Splits" in history.columns else 0.0
    restored = restore_traded_prices(
        history["Open"].astype(float).tolist(),
        history["High"].astype(float).tolist(),
        history["Low"].astype(float).tolist(),
        history["Close"].astype(float).tolist(),
        pd.Series(dividends).fillna(0).astype(float).tolist(),
        pd.Series(splits).fillna(0).astype(float).tolist(),
    )
    frame = pd.DataFrame({
        "datetime": history["datetime"],
        "open": restored[0],
        "high": restored[1],
        "low": restored[2],
        "close": restored[3],
        "volume": history["Volume"].astype(float).tolist(),
    })
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame[(frame["close"] > 0) & (frame["open"] > 0)].reset_index(drop=True)
    split_values = pd.Series(splits).fillna(0).astype(float).tolist()
    # 复权事件跟还原后的成交序列对齐。下载表被丢掉无效行时，事件仍按原序号会错位。
    # 日线里无效行很少；这里用还原后的股息和拆股，长度与成交价一致。
    actions = actions_from_raw(restored[4], split_values[: len(restored[4])])
    if len(frame) != len(restored[0]):
        raise RuntimeError("去掉无效 K 线后，除权事件无法和成交价对齐")
    missing = [column for column in _COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"不复权 K 线缺少 {', '.join(missing)}")
    return frame, actions


def _read_cache(path: Path):
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _write_cache(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
