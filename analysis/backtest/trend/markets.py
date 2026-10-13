"""港股、美股代码识别。

费用计算在成交模型里做。这里只把代码收成规范形式，并挡住 A 股。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

HK = "hk"
US = "us"

_US_BODY = re.compile(r"[A-Z]{1,5}(?:\.[A-Z])?")


@dataclass(frozen=True)
class ParsedSymbol:
    code: str
    market: str

    @property
    def currency(self) -> str:
        if self.market == HK:
            return "HKD"
        return "USD"


def parse_symbol(raw: str) -> ParsedSymbol:
    """港股收成 5 位数字，美股收成大写代码。6 位数字视为 A 股并拒绝。"""
    text = str(raw or "").strip().upper()
    if not text:
        raise ValueError("股票代码为空")
    if text.endswith((".SH", ".SZ", ".SS", ".BJ")):
        raise ValueError(f"{raw} 看起来是 A 股代码，第一期只做港股和美股")

    market = None
    body = text
    if text.endswith(".HK"):
        market = HK
        body = text[:-3]
    elif text.endswith(".US"):
        market = US
        body = text[:-3]

    if market == HK or (market is None and body.isdigit()):
        if not body.isdigit():
            raise ValueError(f"无法识别的港股代码: {raw}")
        if len(body) == 6:
            raise ValueError(f"{body} 是 6 位数字，第一期不做 A 股")
        if not 1 <= len(body) <= 5:
            raise ValueError(f"无法识别的港股代码: {raw}")
        return ParsedSymbol(body.zfill(5), HK)

    if not _US_BODY.fullmatch(body):
        raise ValueError(f"无法识别的代码: {raw}")
    return ParsedSymbol(body, US)
