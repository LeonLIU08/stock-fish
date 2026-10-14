"""港股缺每手股数时，从 Longbridge static_info 取当前值。"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from loguru import logger

from analysis.backtest.trend.markets import parse_symbol

_BATCH = 50


def fetch_hk_lot_sizes(codes: Sequence[str]) -> Dict[str, Tuple[int, Optional[str]]]:
    """返回规范代码到 (每手股数, 中文名)。没查到的代码不出现。"""
    from market_data.data_fetchers.longbridge_fetcher import LongbridgeFetcher, _to_longbridge_symbol

    wanted: Dict[str, str] = {}
    for code in codes:
        parsed = parse_symbol(code)
        remote = _to_longbridge_symbol(parsed.code)
        if remote:
            wanted[remote] = parsed.code
    if not wanted:
        return {}
    fetcher = LongbridgeFetcher()
    ctx = fetcher._get_ctx()
    if ctx is None:
        logger.warning("Longbridge 不可用，无法补充港股每手股数")
        return {}
    found: Dict[str, Tuple[int, Optional[str]]] = {}
    symbols = list(wanted)
    for start in range(0, len(symbols), _BATCH):
        batch = symbols[start:start + _BATCH]
        try:
            infos = ctx.static_info(batch) or []
        except Exception as exc:
            logger.warning(f"Longbridge static_info 失败 ({len(batch)} 只): {exc}")
            continue
        for info in infos:
            remote = str(getattr(info, "symbol", "") or "")
            code = wanted.get(remote)
            if code is None:
                try:
                    code = parse_symbol(remote).code
                except ValueError:
                    continue
            lot = getattr(info, "lot_size", None)
            if isinstance(lot, bool) or not isinstance(lot, int) or lot < 1:
                continue
            name = _name(info)
            found[code] = (lot, name)
        logger.info(f"Longbridge 每手股数 {min(start + _BATCH, len(symbols))}/{len(symbols)}")
    return found


def _name(info) -> Optional[str]:
    for attr in ("name_cn", "name_hk", "name_en"):
        text = str(getattr(info, attr, "") or "").strip()
        if text:
            return text
    return None
