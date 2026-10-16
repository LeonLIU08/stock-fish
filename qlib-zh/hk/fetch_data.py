"""拉取港股日线，预处理后写成 qlib bin。

读取 qlib-zh/hk/baseline.yaml。中间结果是 pickle，放在 ~/.qlib/hk_staging/<freeze_dirname>/，
qlib 目录放在 ~/.qlib/qlib_data/<freeze_dirname>/。已成功的股票会跳过，中断后可续跑。

    bash scripts/fetch_hk_qlib_data.sh --symbols 00700,02800
    bash scripts/fetch_hk_qlib_data.sh
    bash scripts/fetch_hk_qlib_data.sh --dump-only

默认股票池是东财当前港股列表，另加强制加入日历标的。退市股票若接口已经不提供，这里不会出现。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "qlib-zh" / "hk" / "baseline.yaml"
HK_TZ = "Asia/Hong_Kong"
FIELDS = ["open", "high", "low", "close", "volume", "vwap", "factor"]
BAR_COLUMNS = ["date", *FIELDS, "source"]
NAME_DROP = re.compile(r"窝轮|牛熊证|界内证")
# 港股普通股和 ETF 代码在 00001–09999。10000 及以上是窝轮、牛熊证等。
EQUITY_CODE_LIMIT = 10000
TRANSIENT_MARKERS = (
    "RemoteDisconnected",
    "Connection aborted",
    "Connection reset",
    "ConnectionError",
    "timed out",
    "Timeout",
    "Temporary failure",
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置不是映射: {path}")
    return cfg


def freeze_name(cfg: dict) -> str:
    name = str(cfg["data"]["freeze_dirname"]).strip()
    if not name:
        raise ValueError("data.freeze_dirname 为空")
    return name


def qlib_dir(cfg: dict) -> Path:
    return Path.home() / ".qlib" / "qlib_data" / freeze_name(cfg)


def staging_dir(cfg: dict) -> Path:
    return Path.home() / ".qlib" / "hk_staging" / freeze_name(cfg)


def as_date(value) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def normalize_code(symbol: str, width: int) -> str:
    digits = re.sub(r"\D", "", str(symbol).strip())
    if not digits or len(digits) > width:
        raise ValueError(f"无法解析港股代码: {symbol}")
    return digits.zfill(width)


def to_yahoo_hk(symbol: str, width: int) -> str:
    code = normalize_code(symbol, width)
    return f"{(code.lstrip('0') or '0').zfill(4)}.HK"


def to_hk_dates(values) -> pd.Series:
    ts = pd.to_datetime(values, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(HK_TZ)
    return ts.dt.normalize().dt.tz_localize(None)


def _finite_positive(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    return num.where(np.isfinite(num))


def clean_bars(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """统一成复权价、成交量、VWAP、复权因子。"""
    out = df.copy()
    if "date" not in out.columns:
        raise ValueError("缺少 date")
    out["date"] = to_hk_dates(out["date"])
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            raise ValueError(f"缺少 {col}")
        out[col] = _finite_positive(out[col])
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out[out["close"] > 0]
    out = out[out["open"] > 0]
    out = out[out["high"] > 0]
    out = out[out["low"] > 0]
    out = out[out["high"] >= out["low"]]
    out["volume"] = out["volume"].fillna(0)
    out = out[out["volume"] >= 0]

    if "factor" in out.columns:
        factor = _finite_positive(out["factor"]).fillna(1.0)
    else:
        factor = pd.Series(1.0, index=out.index)
    factor = factor.where(factor > 0, 1.0)
    out["factor"] = factor

    amount = _finite_positive(out["amount"]) if "amount" in out.columns else None
    if amount is None:
        amount = out["close"] * out["volume"]
    else:
        amount = amount.fillna(out["close"] * out["volume"])
    out["vwap"] = np.where(out["volume"] > 0, amount / out["volume"], out["close"])
    out["vwap"] = _finite_positive(out["vwap"]).fillna(out["close"])
    out["source"] = source
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    return out[BAR_COLUMNS].reset_index(drop=True)


_LB_MANAGER = None


def fetch_longbridge(symbol: str, start: date, end: date) -> Optional[pd.DataFrame]:
    global _LB_MANAGER
    from market_data.data_fetchers.base import DataFetcherManager

    if _LB_MANAGER is None:
        _LB_MANAGER = DataFetcherManager()
    manager = _LB_MANAGER
    fetcher = manager._get_fetcher_by_name("LongbridgeFetcher", capability="daily_data")
    if fetcher is None or not hasattr(fetcher, "fetch_history_bars"):
        return None
    raw = manager._call_fetcher_method(
        fetcher,
        "fetch_history_bars",
        symbol,
        start.isoformat(),
        end.isoformat(),
        "1d",
    )
    if raw is None or raw.empty:
        return None
    frame = raw.rename(columns={"datetime": "date"})
    frame["factor"] = 1.0
    return clean_bars(frame, "longbridge_qfq")


def fetch_yahoo(symbol: str, start: date, end: date, width: int) -> Optional[pd.DataFrame]:
    import yfinance as yf

    ticker = to_yahoo_hk(symbol, width)
    raw = yf.download(
        tickers=ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    if "Date" in raw.columns:
        raw = raw.rename(columns={"Date": "date"})
    elif "Datetime" in raw.columns:
        raw = raw.rename(columns={"Datetime": "date"})
    raw = raw.rename(columns={col: str(col).lower().replace(" ", "_") for col in raw.columns})
    close = pd.to_numeric(raw["close"], errors="coerce")
    adj = pd.to_numeric(raw.get("adj_close"), errors="coerce")
    factor = (adj / close).where(close > 0)
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    for col in ("open", "high", "low", "close"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce") * factor
    raw["factor"] = factor
    raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
    return clean_bars(raw, f"yahoo_adj:{ticker}")


def fetch_akshare(symbol: str, start: date, end: date, width: int) -> Optional[pd.DataFrame]:
    import akshare as ak

    code = normalize_code(symbol, width)
    raw = ak.stock_hk_hist(
        symbol=code,
        period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust="qfq",
    )
    if raw is None or raw.empty:
        return None
    renamed = raw.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
        }
    )
    renamed["factor"] = 1.0
    return clean_bars(renamed, "akshare_qfq")


FETCHERS = {
    "longbridge": fetch_longbridge,
    "yahoo": fetch_yahoo,
    "akshare": fetch_akshare,
}


def is_transient(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(marker in text for marker in TRANSIENT_MARKERS)


def fetch_symbol(symbol: str, start: date, end: date, width: int, sources: Iterable[str]) -> pd.DataFrame:
    errors = []
    for name in sources:
        fetcher = FETCHERS.get(name)
        if fetcher is None:
            errors.append(f"{name}: 未知来源")
            continue
        for attempt in range(3):
            try:
                if name == "longbridge":
                    frame = fetcher(symbol, start, end)
                else:
                    frame = fetcher(symbol, start, end, width)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                if not is_transient(exc) or attempt == 2:
                    logger.warning(f"{symbol} {name} 失败: {exc}")
                    break
                delay = 2 * (attempt + 1)
                logger.warning(f"{symbol} {name} 断线，{delay}s 后重试 ({attempt + 1}/3)")
                time.sleep(delay)
                continue
            if frame is None or frame.empty:
                errors.append(f"{name}: empty")
                break
            return frame
    raise RuntimeError("; ".join(errors) or "no source")


def read_symbol_file(path: Path, width: int) -> list[str]:
    codes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        codes.append(normalize_code(text.split()[0], width))
    return codes


def load_em_symbols(width: int) -> list[str]:
    import akshare as ak

    spot = None
    for attempt in range(3):
        try:
            spot = ak.stock_hk_spot_em()
            break
        except Exception as exc:
            if not is_transient(exc) or attempt == 2:
                raise
            delay = 5 * (attempt + 1)
            logger.warning(f"东财列表断线，{delay}s 后重试 ({attempt + 1}/3): {exc}")
            time.sleep(delay)
    if spot is None:
        raise RuntimeError("东财港股列表为空")
    if "代码" not in spot.columns:
        raise RuntimeError(f"东财港股列表缺少代码列: {list(spot.columns)}")
    names = spot["名称"] if "名称" in spot.columns else pd.Series("", index=spot.index)
    codes = []
    skipped_products = 0
    for raw, name in zip(spot["代码"], names):
        text = str(name or "")
        if NAME_DROP.search(text):
            skipped_products += 1
            continue
        try:
            code = normalize_code(raw, width)
        except ValueError:
            continue
        if int(code) >= EQUITY_CODE_LIMIT:
            skipped_products += 1
            continue
        codes.append(code)
    logger.info(f"东财列表去掉窝轮等 {skipped_products} 只，保留 {len(codes)} 只")
    return codes


def resolve_symbols(args, cfg: dict) -> list[str]:
    width = int(cfg["data"]["code_width"])
    calendar = normalize_code(cfg["data"]["calendar_instrument"], width)
    if args.symbols:
        picked = [normalize_code(item, width) for item in args.symbols.split(",") if item.strip()]
    elif args.universe == "hkconnect":
        picked = read_symbol_file(ROOT / "scripts" / "hkconnect_06Oct.txt", width)
    elif args.universe == "file":
        if not args.symbols_file:
            raise ValueError("--universe file 需要 --symbols-file")
        picked = read_symbol_file(Path(args.symbols_file), width)
    else:
        cached = staging_dir(cfg) / "symbols.txt"
        try:
            logger.info("从东财读取当前港股列表")
            picked = load_em_symbols(width)
        except Exception as exc:
            if not cached.exists():
                raise
            logger.warning(f"东财列表失败，改用已保存名单 {cached}: {exc}")
            picked = read_symbol_file(cached, width)
        picked = [code for code in picked if int(code) < EQUITY_CODE_LIMIT]
    ordered = []
    seen = set()
    for code in [calendar, *picked]:
        if code not in seen:
            ordered.append(code)
            seen.add(code)
    if args.limit is not None:
        rest = [code for code in ordered if code != calendar]
        ordered = [calendar] + rest[: max(args.limit - 1, 0)]
        if calendar not in ordered:
            ordered.insert(0, calendar)
    return ordered


def bar_path(staging: Path, code: str) -> Path:
    return staging / "bars" / f"{code}.pkl"


def fresh_enough(path: Path, end: date) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_pickle(path)
    except Exception:
        return False
    if frame.empty:
        return False
    last = pd.to_datetime(frame["date"]).max().date()
    return last >= end - timedelta(days=7)


def write_bars(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pkl.tmp")
    frame.to_pickle(tmp)
    tmp.replace(path)


def append_failure(path: Path, code: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if new_file:
            handle.write("code,error\n")
        escaped = message.replace('"', "'").replace("\n", " ")
        handle.write(f'{code},"{escaped}"\n')


def fetch_all(codes: list[str], cfg: dict, args) -> tuple[int, int]:
    data = cfg["data"]
    start = as_date(data["download_start"])
    end = as_date(data["download_end"]) or date.today()
    if start is None:
        raise ValueError("data.download_start 为空")
    width = int(data["code_width"])
    if args.sources:
        sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    else:
        # Longbridge 日线按 400 天分片。全市场先走单次请求的来源，它只作失败后的补齐。
        configured = [item for item in data["sources"] if item != "longbridge"]
        if "longbridge" in data["sources"]:
            configured.append("longbridge")
            logger.info("Longbridge 放到最后：日线接口按 400 天分片。要用它优先，请传 --sources")
        sources = configured
    staging = staging_dir(cfg)
    failures = staging / "failures.csv"
    ok = 0
    failed = 0
    logger.info(f"下载 {start.isoformat()} ~ {end.isoformat()}，来源 {sources}，共 {len(codes)} 只")
    for index, code in enumerate(codes, start=1):
        path = bar_path(staging, code)
        if not args.refresh and fresh_enough(path, end):
            ok += 1
            logger.info(f"[{index}/{len(codes)}] {code} 已有，跳过")
            continue
        try:
            frame = fetch_symbol(code, start, end, width, sources)
            if frame.empty:
                raise RuntimeError("empty after clean")
            write_bars(path, frame)
        except Exception as exc:
            failed += 1
            append_failure(failures, code, str(exc))
            logger.error(f"[{index}/{len(codes)}] {code} 失败: {exc}")
            continue
        ok += 1
        logger.info(
            f"[{index}/{len(codes)}] {code} {frame['source'].iloc[0]} "
            f"{frame['date'].iloc[0].date().isoformat()} ~ {frame['date'].iloc[-1].date().isoformat()} "
            f"bars={len(frame)}"
        )
        if args.sleep > 0 and index < len(codes):
            time.sleep(args.sleep)
    return ok, failed


def iter_staged_bars(staging: Path):
    bars = staging / "bars"
    if not bars.exists():
        return
    for path in sorted(bars.glob("*.pkl")):
        code = path.stem
        if code.isdigit() and int(code) >= EQUITY_CODE_LIMIT:
            continue
        frame = pd.read_pickle(path)
        frame["date"] = pd.to_datetime(frame["date"])
        if frame.empty:
            continue
        yield code, frame


def build_calendar(frames: dict[str, pd.DataFrame], calendar_code: str) -> list[pd.Timestamp]:
    frame = frames.get(calendar_code)
    if frame is None or frame.empty:
        raise RuntimeError(f"日历标的 {calendar_code} 没有行情，无法生成交易日")
    days = sorted(pd.to_datetime(frame["date"]).dt.normalize().unique())
    return [pd.Timestamp(day) for day in days]


def dump_symbol(code: str, frame: pd.DataFrame, calendar: list[pd.Timestamp], features_dir: Path) -> Optional[tuple[str, str]]:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data = data.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    data = data.sort_values("date")
    if data.empty:
        return None
    calendar_df = pd.DataFrame({"date": calendar})
    window = calendar_df[
        (calendar_df["date"] >= data["date"].min()) & (calendar_df["date"] <= data["date"].max())
    ]
    if window.empty:
        return None
    aligned = data.set_index("date").reindex(window["date"])
    start_at = calendar.index(aligned.index.min())
    symbol_dir = features_dir / code.lower()
    symbol_dir.mkdir(parents=True, exist_ok=True)
    for field in FIELDS:
        values = pd.to_numeric(aligned[field], errors="coerce").to_numpy(dtype="float64")
        payload = np.hstack([np.float64(start_at), values]).astype("<f")
        payload.tofile(symbol_dir / f"{field}.day.bin")
    usable = aligned["close"].dropna()
    if usable.empty:
        return None
    return usable.index.min().strftime("%Y-%m-%d"), usable.index.max().strftime("%Y-%m-%d")


def dump_qlib(cfg: dict) -> dict:
    width = int(cfg["data"]["code_width"])
    calendar_code = normalize_code(cfg["data"]["calendar_instrument"], width)
    excluded = {normalize_code(code, width) for code in cfg.get("universe", {}).get("exclude", [])}
    staging = staging_dir(cfg)
    target = qlib_dir(cfg)
    calendar_frame = None
    for code, frame in iter_staged_bars(staging):
        if code == calendar_code:
            calendar_frame = frame
            break
    if calendar_frame is None:
        raise RuntimeError(f"日历标的 {calendar_code} 没有行情，无法生成交易日")
    calendar = build_calendar({calendar_code: calendar_frame}, calendar_code)
    calendars_dir = target / "calendars"
    instruments_dir = target / "instruments"
    features_dir = target / "features"
    calendars_dir.mkdir(parents=True, exist_ok=True)
    instruments_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    (calendars_dir / "day.txt").write_text(
        "\n".join(day.strftime("%Y-%m-%d") for day in calendar) + "\n",
        encoding="utf-8",
    )

    rows = []
    tradable = []
    health: dict[str, int] = {}
    written = 0
    for code, frame in iter_staged_bars(staging):
        span = dump_symbol(code, frame, calendar, features_dir)
        if span is None:
            logger.warning(f"{code} 与交易日历没有重叠，跳过")
            continue
        rows.append((code.upper(), span[0], span[1]))
        if code not in excluded and code != calendar_code:
            tradable.append((code.upper(), span[0], span[1]))
            years = pd.to_datetime(frame["date"]).dt.year
            close = pd.to_numeric(frame["close"], errors="coerce")
            for year in sorted(years[close.notna()].unique()):
                key = str(int(year))
                health[key] = health.get(key, 0) + 1
        written += 1
        if written % 200 == 0:
            logger.info(f"已写入 {written} 只")
    if not rows:
        raise RuntimeError(f"没有可写入的行情: {staging / 'bars'}")
    pd.DataFrame(rows, columns=["symbol", "start", "end"]).to_csv(
        instruments_dir / "all.txt", sep="\t", header=False, index=False
    )
    pd.DataFrame(tradable, columns=["symbol", "start", "end"]).to_csv(
        instruments_dir / "tradable.txt", sep="\t", header=False, index=False
    )
    min_names = int(cfg["data"]["checks"]["min_names_per_year"])
    report = {
        "freeze_dirname": freeze_name(cfg),
        "qlib_dir": str(target),
        "staging_dir": str(staging),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "calendar_instrument": calendar_code,
        "calendar_start": calendar[0].strftime("%Y-%m-%d"),
        "calendar_end": calendar[-1].strftime("%Y-%m-%d"),
        "calendar_days": len(calendar),
        "symbols": int(len(rows)),
        "tradable_symbols": int(len(tradable)),
        "partial_download": True,
        "min_names_per_year": min_names,
        "names_by_year": dict(sorted(health.items())),
        "years_below_min": [year for year, count in health.items() if count < min_names],
    }
    (target / "meta.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(
        f"写入 {target}，日历 {report['calendar_start']} ~ {report['calendar_end']}，"
        f"{report['calendar_days']} 天，{report['symbols']} 只"
    )
    if report["years_below_min"]:
        logger.warning(f"这些年份入池不足 {min_names}: {report['years_below_min']}")
    return report


def yearly_counts(frames: dict[str, pd.DataFrame], calendar_code: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for code, frame in frames.items():
        if code == calendar_code:
            continue
        years = pd.to_datetime(frame["date"]).dt.year
        close = pd.to_numeric(frame["close"], errors="coerce")
        for year in sorted(years[close.notna()].unique()):
            key = str(int(year))
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def run_self_test() -> None:
    calendar = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "open": [10.0, 12.0],
            "high": [11.0, 13.0],
            "low": [9.0, 11.0],
            "close": [10.5, 12.5],
            "volume": [100.0, 0.0],
            "vwap": [10.4, 12.5],
            "factor": [1.0, 1.0],
            "source": ["test", "test"],
        }
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        span = dump_symbol("00700", frame, calendar, root)
        assert span == ("2024-01-02", "2024-01-04"), span
        payload = np.fromfile(root / "00700" / "close.day.bin", dtype="<f4")
        assert payload.shape == (4,), payload.shape
        assert int(payload[0]) == 0
        assert abs(float(payload[1]) - 10.5) < 1e-4
        assert np.isnan(payload[2])
        assert abs(float(payload[3]) - 12.5) < 1e-4
    logger.info("bin 自检通过")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拉取港股日线并写成 qlib bin")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--universe", choices=["em", "hkconnect", "file"], default="em")
    parser.add_argument("--symbols-file", type=Path, default=None)
    parser.add_argument("--symbols", default="", help="逗号分隔，指定后不再读取股票池")
    parser.add_argument("--limit", type=int, default=None, help="只拉前 N 只，日历标的始终保留")
    parser.add_argument("--sources", default="", help="覆盖配置里的 data.sources，逗号分隔")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--refresh", action="store_true", help="已有 parquet 也重新下载")
    parser.add_argument("--dump-only", action="store_true", help="只用已下载的 parquet 重写 qlib bin")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    cfg = load_config(args.config)
    staging = staging_dir(cfg)
    staging.mkdir(parents=True, exist_ok=True)
    if not args.dump_only:
        codes = resolve_symbols(args, cfg)
        (staging / "symbols.txt").write_text("\n".join(codes) + "\n", encoding="utf-8")
        ok, failed = fetch_all(codes, cfg, args)
        logger.info(f"下载结束，成功或跳过 {ok}，失败 {failed}")
        if ok == 0:
            return 1
    report = dump_qlib(cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
