"""Load daily / minute OHLCV for HK names and Hang Seng benchmarks."""
from __future__ import annotations

import hashlib
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from loguru import logger

from analysis.backtest.config import BacktestConfig

HK_TZ = "Asia/Hong_Kong"
CACHE_DIR = Path("data/backtest_cache")

_YF_INTERVAL = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "60m": "60m",
    "1d": "1d",
}

# Yahoo intraday history windows (calendar days). Beyond this the API truncates.
_YF_MAX_CALENDAR_DAYS = {
    "1m": 7,
    "5m": 60,
    "15m": 60,
    "60m": 730,
    "1d": 3650,
}


def to_yahoo_hk(symbol: str) -> str:
    """00700 / HK00700 / 00700.HK -> 0700.HK"""
    raw = (symbol or "").strip().upper()
    if raw.startswith("^"):
        return raw
    if raw.endswith(".HK"):
        digits = re.sub(r"\D", "", raw[:-3])
    elif raw.startswith("HK"):
        digits = re.sub(r"\D", "", raw[2:])
    else:
        digits = re.sub(r"\D", "", raw)
    if not digits:
        raise ValueError(f"无法解析港股代码: {symbol}")
    return f"{(digits.lstrip('0') or '0').zfill(4)}.HK"


def _ensure_hk_tz(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series)
    if getattr(ts.dt, "tz", None) is None:
        # Daily dates are session dates in HK; naive minute stamps from Yahoo
        # are usually already exchange-local or UTC — treat naive as HK.
        return ts.dt.tz_localize(HK_TZ)
    return ts.dt.tz_convert(HK_TZ)


def _normalize_ohlcv(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount"])

    out = df.copy()
    rename = {}
    for src, dest in {
        "date": "datetime",
        "Date": "datetime",
        "time": "datetime",
        "时间": "datetime",
        "日期": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "turnover": "amount",
    }.items():
        if src in out.columns and dest not in out.columns:
            rename[src] = dest
    if rename:
        out = out.rename(columns=rename)

    if datetime_col != "datetime" and datetime_col in out.columns:
        out = out.rename(columns={datetime_col: "datetime"})

    if "datetime" not in out.columns:
        if out.index.name in ("date", "datetime", "Date") or isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            first = out.columns[0]
            if first != "datetime":
                out = out.rename(columns={first: "datetime"})
        else:
            raise ValueError("K 线缺少 datetime/date 列")

    out["datetime"] = _ensure_hk_tz(out["datetime"])
    for col in ("open", "high", "low", "close"):
        if col not in out.columns:
            raise ValueError(f"K 线缺少 {col} 列")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["volume"] = pd.to_numeric(out.get("volume", 0), errors="coerce").fillna(0)
    if "amount" not in out.columns:
        out["amount"] = out["close"] * out["volume"]
    else:
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0)

    out = out.dropna(subset=["datetime", "open", "high", "low", "close"])
    out = out[out["close"] > 0]
    out = out.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    return out[["datetime", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)


def _cache_path(key: str) -> Path:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", key)[:80]
    return CACHE_DIR / f"{safe}_{digest}.pkl"


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_pickle(path)
        return _normalize_ohlcv(df)
    except Exception as e:
        logger.warning(f"回测缓存读取失败 {path}: {e}")
        return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(path)
    except Exception as e:
        logger.warning(f"回测缓存写入失败 {path}: {e}")


class BarLoader:
    """Fetch bars with Longbridge → Yahoo → AkShare failover and disk cache."""

    def __init__(self) -> None:
        self._manager = None

    def _get_manager(self):
        if self._manager is None:
            from market_data.data_fetchers.base import DataFetcherManager
            self._manager = DataFetcherManager()
        return self._manager

    def load_symbol(
        self,
        symbol: str,
        config: BacktestConfig,
        warmup_bars: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, str]:
        bars = warmup_bars if warmup_bars is not None else 20
        padded_start = config.start - timedelta(days=config.warmup_calendar_days(bars))
        return self._load(
            cache_key=f"{symbol}_{config.interval}_{padded_start}_{config.end}",
            fetchers=(
                lambda: self._from_longbridge(symbol, padded_start, config.end, config.interval),
                lambda: self._from_yahoo(to_yahoo_hk(symbol), padded_start, config.end, config.interval),
                lambda: self._from_akshare_hk(symbol, padded_start, config.end, config.interval),
                lambda: self._from_manager_daily(symbol, padded_start, config.end, config.interval),
            ),
            start=padded_start,
            end=config.end,
        )

    def load_benchmark(
        self,
        key: str,
        config: BacktestConfig,
        warmup_bars: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, str]:
        spec = (config.benchmarks or {}).get(key)
        if not spec:
            raise KeyError(f"未知基准 {key}")
        bars = warmup_bars if warmup_bars is not None else 20
        padded_start = config.start - timedelta(days=config.warmup_calendar_days(bars))
        symbols = (spec["yahoo"],) + tuple(spec.get("fallbacks") or ())
        fetchers = [
            (lambda yahoo=yahoo: self._from_yahoo(yahoo, padded_start, config.end, "1d"))
            for yahoo in symbols
        ]
        return self._load(
            cache_key=f"bench_{key}_1d_{padded_start}_{config.end}",
            fetchers=fetchers,
            start=padded_start,
            end=config.end,
        )

    def _load(self, cache_key: str, fetchers, start: date, end: date) -> Tuple[pd.DataFrame, str]:
        path = _cache_path(cache_key)
        cached = _read_cache(path)
        if cached is not None and len(cached) > 0 and coverage_ok(cached, start, end):
            logger.info(f"K 线缓存命中 {cache_key}: {len(cached)} bars")
            return cached, "cache"

        errors = []
        for fetcher in fetchers:
            try:
                result = fetcher()
            except Exception as e:
                errors.append(str(e))
                logger.warning(f"K 线源失败 ({cache_key}): {e}")
                continue
            if result is None:
                continue
            df, source = result
            if df is None or df.empty:
                errors.append(f"{source}: empty")
                continue
            df = _normalize_ohlcv(df)
            if df.empty:
                errors.append(f"{source}: empty after normalize")
                continue
            if not coverage_ok(df, start, end):
                span = f"{df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}"
                errors.append(f"{source}: coverage too short ({span})")
                logger.warning(f"K 线覆盖不足，跳过 {source}: {span}")
                continue
            _write_cache(path, df)
            logger.info(f"K 线加载成功 {cache_key}: source={source}, bars={len(df)}")
            return df, source

        raise RuntimeError(f"无法获取 K 线 {cache_key}: {'; '.join(errors) or 'no source'}")

    def _from_longbridge(
        self, symbol: str, start: date, end: date, interval: str
    ) -> Optional[Tuple[pd.DataFrame, str]]:
        manager = self._get_manager()
        fetcher = manager._get_fetcher_by_name("LongbridgeFetcher", capability="daily_data")
        if fetcher is None or not hasattr(fetcher, "fetch_history_bars"):
            return None
        df = manager._call_fetcher_method(
            fetcher,
            "fetch_history_bars",
            symbol,
            start.isoformat(),
            end.isoformat(),
            interval,
        )
        if df is None or df.empty:
            return None
        return df, "longbridge"

    def _from_yahoo(
        self, yahoo_symbol: str, start: date, end: date, interval: str
    ) -> Optional[Tuple[pd.DataFrame, str]]:
        import yfinance as yf

        yf_interval = _YF_INTERVAL.get(interval)
        if not yf_interval:
            return None

        span = (end - start).days
        max_span = _YF_MAX_CALENDAR_DAYS.get(interval, 7)
        if interval != "1d" and span > max_span:
            logger.warning(
                f"Yahoo {yf_interval} 最长约 {max_span} 天，请求 {span} 天，数据可能被截断"
            )

        end_exclusive = end + timedelta(days=1)
        df = yf.download(
            tickers=yahoo_symbol,
            start=start.isoformat(),
            end=end_exclusive.isoformat(),
            interval=yf_interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        # yfinance uses Datetime or Date depending on interval
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "datetime"})
        elif "Date" in df.columns:
            df = df.rename(columns={"Date": "datetime"})
        return df, f"yahoo:{yahoo_symbol}"

    def _from_akshare_hk(
        self, symbol: str, start: date, end: date, interval: str
    ) -> Optional[Tuple[pd.DataFrame, str]]:
        import akshare as ak

        code = re.sub(r"\D", "", symbol).zfill(5)
        start_s = start.strftime("%Y%m%d")
        end_s = end.strftime("%Y%m%d")

        if interval == "1d":
            df = ak.stock_hk_hist(
                symbol=code,
                period="daily",
                start_date=start_s,
                end_date=end_s,
                adjust="qfq",
            )
            source = "akshare_hk_daily"
        else:
            period = {"1m": "1", "5m": "5", "15m": "15", "60m": "60"}.get(interval)
            if period is None or not hasattr(ak, "stock_hk_hist_min_em"):
                return None
            df = ak.stock_hk_hist_min_em(
                symbol=code,
                period=period,
                adjust="qfq",
                start_date=f"{start.isoformat()} 09:00:00",
                end_date=f"{end.isoformat()} 16:10:00",
            )
            source = "akshare_hk_min"
        if df is None or df.empty:
            return None
        return df, source

    def _from_manager_daily(
        self, symbol: str, start: date, end: date, interval: str
    ) -> Optional[Tuple[pd.DataFrame, str]]:
        if interval != "1d":
            return None
        manager = self._get_manager()
        df, source = manager.get_daily_data(
            symbol,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            days=max((end - start).days, 30),
        )
        if df is None or df.empty:
            return None
        return df, f"manager:{source}"


def coverage_ok(df: pd.DataFrame, start: date, end: date, min_frac: float = 0.5) -> bool:
    """Reject truncated feeds (e.g. Yahoo 1m only keeps ~7 days)."""
    if df is None or df.empty:
        return False
    first = pd.Timestamp(df["datetime"].iloc[0])
    last = pd.Timestamp(df["datetime"].iloc[-1])
    if first.tzinfo is not None:
        first = first.tz_convert(HK_TZ)
        last = last.tz_convert(HK_TZ)
    got = max((last.date() - first.date()).days, 1)
    need = max((end - start).days, 1)
    return got >= need * min_frac


def slice_eval_window(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Keep bars whose HK calendar date is inside [start, end]."""
    if df.empty:
        return df
    ts = df["datetime"]
    if getattr(ts.dt, "tz", None) is None:
        day = ts.dt.date
    else:
        day = ts.dt.tz_convert(HK_TZ).dt.date
    return df[(day >= start) & (day <= end)].copy().reset_index(drop=True)


def coverage_note(df: pd.DataFrame, start: date, end: date, interval: str) -> str:
    if df.empty:
        return "无数据"
    first_ts = pd.Timestamp(df["datetime"].iloc[0])
    last_ts = pd.Timestamp(df["datetime"].iloc[-1])
    if first_ts.tzinfo is not None:
        first = first_ts.tz_convert(HK_TZ).date()
        last = last_ts.tz_convert(HK_TZ).date()
        n_days = df["datetime"].dt.tz_convert(HK_TZ).dt.date.nunique()
    else:
        first = first_ts.date()
        last = last_ts.date()
        n_days = pd.to_datetime(df["datetime"]).dt.date.nunique()
    expected = max(int((end - start).days * 5 / 7), 1)
    return (
        f"{first} ~ {last}, {len(df)} bars / {n_days} 个交易日"
        f"（区间约 {expected} 个交易日, interval={interval}）"
    )
