"""拉取中证500日线，写成工程目录里的 qlib bin。

读取 qlib-zh/cn/baseline.yaml。中间 pickle 在 qlib-zh/cn/data/staging/，
qlib 目录在 qlib-zh/cn/data/cn_data/。已成功的股票会跳过，中断后可续跑。

    /usr/bin/python3 qlib-zh/cn/fetch_data.py
    /usr/bin/python3 qlib-zh/cn/fetch_data.py --dump-only
    /usr/bin/python3 qlib-zh/cn/fetch_data.py --symbols 600519,000001
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
HK_DIR = ROOT / "qlib-zh" / "hk"
if str(HK_DIR) not in sys.path:
    sys.path.insert(0, str(HK_DIR))

from fetch_data import clean_bars, dump_symbol  # noqa: E402

DEFAULT_CONFIG = ROOT / "qlib-zh" / "cn" / "baseline.yaml"
INDEX_CODE = "000905"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置不是映射: {path}")
    return cfg


def data_root(cfg: dict) -> Path:
    uri = str(cfg["runtime"]["provider_uri"])
    if uri.startswith("/work/"):
        qlib_path = ROOT / uri[len("/work/") :]
    else:
        qlib_path = Path(uri)
    return qlib_path.parent


def staging_dir(cfg: dict) -> Path:
    return data_root(cfg) / "staging"


def qlib_dir(cfg: dict) -> Path:
    return data_root(cfg) / "cn_data"


def read_codes(path: Path) -> list[str]:
    codes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        digits = "".join(ch for ch in text.split()[0] if ch.isdigit()).zfill(6)
        if len(digits) != 6:
            raise ValueError(f"无法解析 A 股代码: {text}")
        codes.append(digits)
    if not codes:
        raise ValueError(f"股票名单是空的: {path}")
    return codes


def qlib_symbol(code: str, *, index: bool = False) -> str:
    if index or code.startswith(("5", "6", "9")):
        prefix = "SH"
    elif code.startswith(("0", "2", "3")):
        prefix = "SZ"
    elif code.startswith(("4", "8")):
        prefix = "BJ"
    else:
        raise ValueError(f"无法判断交易所: {code}")
    return f"{prefix}{code}"


def baostock_symbol(code: str, *, index: bool = False) -> str:
    symbol = qlib_symbol(code, index=index)
    return f"{symbol[:2].lower()}.{symbol[2:]}"


def as_date(value) -> date:
    if value is None or value == "":
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def trade_dates(bs, start: date, end: date) -> list[pd.Timestamp]:
    rs = bs.query_trade_dates(start_date=start.isoformat(), end_date=end.isoformat())
    if rs.error_code != "0":
        raise RuntimeError(f"交易日历失败: {rs.error_msg}")
    days = []
    while rs.next():
        row = rs.get_row_data()
        if len(row) >= 2 and row[1] == "1":
            days.append(pd.Timestamp(row[0]))
    if not days:
        raise RuntimeError("交易日历为空")
    return days


def fetch_bars(bs, code: str, start: date, end: date, *, index: bool = False) -> pd.DataFrame:
    last_error = "empty"
    for attempt in range(3):
        rs = bs.query_history_k_data_plus(
            baostock_symbol(code, index=index),
            "date,open,high,low,close,volume,amount",
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            frequency="d",
            adjustflag="2",
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if rs.error_code != "0":
            last_error = rs.error_msg
            time.sleep(1.5 * (attempt + 1))
            continue
        if not rows:
            raise RuntimeError("BaoStock 没有返回行情")
        frame = pd.DataFrame(rows, columns=rs.fields)
        frame["factor"] = 1.0
        cleaned = clean_bars(frame, "baostock_qfq")
        if cleaned.empty:
            raise RuntimeError("清洗后没有行情")
        return cleaned
    raise RuntimeError(last_error)


def fresh_enough(path: Path, end: date) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_pickle(path)
    except Exception:
        return False
    if frame.empty or "date" not in frame.columns:
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


def dump_qlib(cfg: dict, calendar: list[pd.Timestamp]) -> dict:
    staging = staging_dir(cfg) / "bars"
    target = qlib_dir(cfg)
    features = target / "features"
    instruments = target / "instruments"
    calendars = target / "calendars"
    for path in (features, instruments, calendars):
        path.mkdir(parents=True, exist_ok=True)
    (calendars / "day.txt").write_text(
        "\n".join(day.strftime("%Y-%m-%d") for day in calendar) + "\n",
        encoding="utf-8",
    )
    rows = []
    tradable = []
    health: dict[str, int] = {}
    index_symbol = qlib_symbol(INDEX_CODE, index=True)
    for path in sorted(staging.glob("*.pkl")):
        code = path.stem
        symbol = qlib_symbol(code, index=(code == INDEX_CODE))
        frame = pd.read_pickle(path)
        frame["date"] = pd.to_datetime(frame["date"])
        span = dump_symbol(symbol, frame, calendar, features)
        if span is None:
            logger.warning(f"{symbol} 与交易日历没有重叠，跳过")
            continue
        rows.append((symbol, span[0], span[1]))
        if symbol != index_symbol:
            tradable.append((symbol, span[0], span[1]))
            years = pd.to_datetime(frame["date"]).dt.year
            close = pd.to_numeric(frame["close"], errors="coerce")
            for year in sorted(years[close.notna()].unique()):
                key = str(int(year))
                health[key] = health.get(key, 0) + 1
    if not rows:
        raise RuntimeError(f"没有可写入的行情: {staging}")
    pd.DataFrame(rows, columns=["symbol", "start", "end"]).to_csv(
        instruments / "all.txt", sep="\t", header=False, index=False
    )
    pd.DataFrame(tradable, columns=["symbol", "start", "end"]).to_csv(
        instruments / "tradable.txt", sep="\t", header=False, index=False
    )
    report = {
        "qlib_dir": str(target),
        "staging_dir": str(staging.parent),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "calendar_start": calendar[0].strftime("%Y-%m-%d"),
        "calendar_end": calendar[-1].strftime("%Y-%m-%d"),
        "calendar_days": len(calendar),
        "symbols": len(rows),
        "tradable_symbols": len(tradable),
        "names_by_year": dict(sorted(health.items())),
        "price_adjustment": "qfq",
        "factor": "prices are already forward-adjusted; factor is 1",
    }
    (target / "meta.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        f"写入 {target}，日历 {report['calendar_start']} ~ {report['calendar_end']}，"
        f"{report['calendar_days']} 天，{report['symbols']} 只"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="拉取中证500日线并写成 qlib bin")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--symbols", default="", help="逗号分隔的六位代码，默认读配置里的名单")
    parser.add_argument("--dump-only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    start = as_date(cfg["data"]["download_start"])
    end = as_date(cfg["data"].get("download_end"))
    symbols_file = ROOT / str(cfg["universe"]["symbols_file"])
    codes = [item.strip().zfill(6) for item in args.symbols.split(",") if item.strip()] or read_codes(symbols_file)
    if INDEX_CODE not in codes:
        codes = [INDEX_CODE, *codes]
    staging = staging_dir(cfg)
    bars = staging / "bars"
    failures = staging / "failures.csv"
    logger.info(f"区间 {start.isoformat()} ~ {end.isoformat()}，{len(codes)} 只，目录 {data_root(cfg)}")

    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {login.error_msg}")
    try:
        calendar = trade_dates(bs, start, end)
        if not args.dump_only:
            ok = 0
            failed = 0
            for index, code in enumerate(codes, start=1):
                path = bars / f"{code}.pkl"
                if not args.refresh and fresh_enough(path, end):
                    ok += 1
                    logger.info(f"[{index}/{len(codes)}] {code} 已有，跳过")
                    continue
                try:
                    frame = fetch_bars(bs, code, start, end, index=(code == INDEX_CODE))
                except Exception as exc:
                    failed += 1
                    append_failure(failures, code, str(exc))
                    logger.error(f"[{index}/{len(codes)}] {code} 失败: {exc}")
                    continue
                write_bars(path, frame)
                ok += 1
                logger.info(
                    f"[{index}/{len(codes)}] {qlib_symbol(code, index=(code == INDEX_CODE))} "
                    f"{frame['date'].iloc[0].date().isoformat()} ~ {frame['date'].iloc[-1].date().isoformat()} "
                    f"bars={len(frame)}"
                )
            logger.info(f"下载完成，成功 {ok}，失败 {failed}")
        report = dump_qlib(cfg, calendar)
    finally:
        bs.logout()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
