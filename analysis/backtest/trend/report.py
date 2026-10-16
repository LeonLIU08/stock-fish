"""趋势回测在现有报告之外的对照表。各账户并列，金额不相加。"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def comparison_markdown(results: Sequence[dict], runs: Sequence[dict]) -> str:
    lines = [
        "## 账户对照",
        "",
        "每个账户独立。表里不把成交次数、盈亏或不同币种的金额加在一起。港元和美元只并列收益率。",
        "",
    ]
    header = ["股票", "币种"]
    for run in runs:
        label = run["label"]
        header.extend([
            f"{label} 成交",
            f"{label} 成本后收益",
            f"{label} 最大回撤",
            f"{label} 胜率",
            f"{label} 盈亏/计划风险",
        ])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for symbol, rows in _by_symbol(results):
        first = rows[0]
        cells = [f"{first.get('name', symbol)} `{symbol}`", str(first.get("currency") or "")]
        located = {row.get("run"): row for row in rows}
        for run in runs:
            row = located.get(run["key"])
            cells.extend(_cells(row))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def comparison_html(
    results: Sequence[dict],
    runs: Sequence[dict],
    output_dir: Optional[Path] = None,
) -> str:
    heads = "".join(
        "<th>成交</th><th>成本后收益</th><th>最大回撤</th><th>胜率</th><th>盈亏/计划风险</th>"
        for _run in runs
    )
    groups = "".join(f"<th colspan='5'>{html.escape(run['label'])}</th>" for run in runs)
    body = []
    for symbol, rows in _by_symbol(results):
        first = rows[0]
        located = {row.get("run"): row for row in rows}
        body.append(
            "<tr>"
            f"<td>{_stock_label(first, symbol)}</td>"
            f"<td>{html.escape(str(first.get('currency') or ''))}</td>"
            f"{_cells_html(located, runs, output_dir)}</tr>"
        )
    return (
        "<h2>账户对照</h2>"
        "<p>每个账户独立。表里不把金额加在一起，港元和美元只并列收益率。点成本后收益，打开该账户的报告。</p>"
        "<table><thead>"
        f"<tr><th rowspan='2'>股票</th><th rowspan='2'>币种</th>{groups}</tr>"
        f"<tr>{heads}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def write_summary_html(
    output_dir: Path,
    results: Sequence[dict],
    runs: Sequence[dict],
    *,
    generated_at: str,
    config: Optional[dict] = None,
    failures: Sequence[dict] = (),
) -> Path:
    """在结果目录写 index.html。点某只股票的某个策略，打开它的 report.html。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = "".join(
        _stock_card(symbol, rows, runs, output_dir)
        for symbol, rows in _by_symbol(results)
    )
    table = comparison_html(results, runs, output_dir)
    cfg = config or {}
    structure = cfg.get("structure") or {}
    scale = structure.get("scale") or ""
    count = len({row["symbol"] for row in results})
    skipped = len(failures)
    document = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>趋势回测汇总</title>
  <style>
    body {{ background:#0f1419; color:#e2e8f0; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif; margin:0; padding:24px; }}
    h1,h2 {{ color:#f8fafc; }}
    a {{ color:#7dd3fc; }}
    .muted {{ color:#94a3b8; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(340px, 1fr)); gap:12px; margin:16px 0 28px; }}
    article.card {{ background:#151b22; border:1px solid #1e293b; border-radius:10px; padding:14px 16px; }}
    article.card h2 {{ margin:0; font-size:1.05rem; }}
    article.card ul {{ list-style:none; margin:10px 0 0; padding:0; }}
    article.card li a, article.card li span.row {{
      display:grid; grid-template-columns:1.5fr 0.7fr 0.9fr 0.5fr; gap:8px; align-items:baseline;
      padding:7px 0; border-top:1px solid #1e293b; text-decoration:none; color:inherit;
    }}
    article.card li a:hover {{ color:#7dd3fc; }}
    .up {{ color:#f87171; }}
    .down {{ color:#4ade80; }}
    .failed {{ margin-top: 1.6rem; }}
    .failed li {{ margin: 0.7rem 0; }}
    table {{ border-collapse:collapse; width:100%; margin:12px 0 24px; font-size:13px; }}
    th,td {{ border:1px solid #334155; padding:8px 10px; text-align:right; }}
    th {{ background:#1e293b; text-align:center; }}
    td:nth-child(-n+2), th:nth-child(-n+2) {{ text-align:left; }}
  </style>
</head>
<body>
  <h1>趋势回测汇总</h1>
  <p class="muted">{html.escape(generated_at)} · {html.escape(str(cfg.get('start') or ''))} ~ {html.escape(str(cfg.get('end') or ''))} · 尺度 {html.escape(str(scale))} · {count} 只</p>
  <p>点某只股票下的策略，打开该账户的 report.html。完成 {count} 只，跳过 {skipped} 只。</p>
  <div class="grid">
{cards}  </div>
  {table}
  {_failures_html(failures)}
  <p><a href="summary.md">Markdown 总览</a> · <a href="summary.json">JSON</a></p>
</body>
</html>
"""
    path = output_dir / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def _failures_html(failures: Sequence[dict]) -> str:
    if not failures:
        return ""
    items = "".join(
        "<li><strong>"
        f"{html.escape(str(item.get('name') or item['symbol']))} {html.escape(str(item['symbol']))}"
        "</strong>"
        f"<p class=\"muted\">{html.escape(str(item.get('error') or ''))}</p></li>"
        for item in failures
    )
    return f"<section class=\"failed\"><h2>这次跳过</h2><ul>{items}</ul></section>"


def _by_symbol(results: Sequence[dict]):
    order: List[str] = []
    grouped: Dict[str, List[dict]] = {}
    for row in results:
        symbol = row["symbol"]
        if symbol not in grouped:
            order.append(symbol)
            grouped[symbol] = []
        grouped[symbol].append(row)
    return [(symbol, grouped[symbol]) for symbol in order]


def _cells(row: dict | None) -> List[str]:
    if row is None:
        return ["—", "—", "—", "—", "—"]
    metrics = row.get("metrics") or {}
    audit = row.get("audit") or {}
    return [
        str(metrics.get("n_trades", 0)),
        _pct(metrics.get("total_return")),
        _pct(metrics.get("max_drawdown")),
        _pct(metrics.get("win_rate")) if metrics.get("win_rate") is not None else "—",
        _num(audit.get("mean_r_multiple")),
    ]


def _cells_html(located: dict, runs: Sequence[dict], output_dir: Optional[Path]) -> str:
    parts: List[str] = []
    for run in runs:
        row = located.get(run["key"])
        href = _report_href(row, output_dir) if row else None
        for index, value in enumerate(_cells(row)):
            text = html.escape(value)
            if href and index == 1:
                text = f'<a href="{html.escape(href, quote=True)}">{text}</a>'
            parts.append(f"<td>{text}</td>")
    return "".join(parts)


def _stock_label(row: dict, symbol: str) -> str:
    name = str(row.get("name") or symbol)
    return f"{html.escape(name)} {html.escape(symbol)}"


def _stock_card(symbol: str, rows: Sequence[dict], runs: Sequence[dict], output_dir: Path) -> str:
    first = rows[0]
    located = {row.get("run"): row for row in rows}
    items = []
    for run in runs:
        row = located.get(run["key"])
        label = html.escape(str(run["label"]))
        if row is None:
            items.append(f'<li><span class="row"><strong>{label}</strong><span>—</span><span></span><span></span></span></li>')
            continue
        metrics = row.get("metrics") or {}
        ret = metrics.get("total_return")
        line = (
            f"<strong>{label}</strong>"
            f"<span class=\"{_tone(ret)}\">{html.escape(_pct(ret))}</span>"
            f"<span>回撤 {html.escape(_pct(metrics.get('max_drawdown')))}</span>"
            f"<span>{html.escape(str(metrics.get('n_trades', 0)))} 笔</span>"
        )
        href = _report_href(row, output_dir)
        if href:
            items.append(f'<li><a href="{html.escape(href, quote=True)}">{line}</a></li>')
        else:
            items.append(f'<li><span class="row">{line}</span></li>')
    return (
        '<article class="card">'
        f"<h2>{_stock_label(first, symbol)}</h2>"
        f"<p class=\"muted\">{html.escape(str(first.get('currency') or ''))}</p>"
        f"<ul>{''.join(items)}</ul>"
        "</article>\n"
    )


def _report_href(row: Optional[dict], output_dir: Optional[Path]) -> Optional[str]:
    if not row:
        return None
    raw = row.get("report_html")
    symbol = row.get("symbol")
    run = row.get("run")
    fallback = f"{symbol}_{run}/report.html" if symbol and run else None
    if not raw:
        return fallback
    path = Path(str(raw))
    if output_dir is None:
        return path.name
    try:
        return path.resolve().relative_to(Path(output_dir).resolve()).as_posix()
    except ValueError:
        return fallback or path.name


def _tone(value) -> str:
    if value is None:
        return ""
    number = float(value)
    if number > 0:
        return "up"
    if number < 0:
        return "down"
    return ""


def _pct(value) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"


def _num(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2f}"
