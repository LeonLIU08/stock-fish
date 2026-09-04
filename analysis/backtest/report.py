"""Write JSON + Markdown + HTML backtest reports with charts."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis.backtest.config import BacktestConfig
from analysis.backtest.engine import Trade


def _pct(v) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.2f}%"


def _num(v, digits=2) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def trade_to_dict(t: Trade) -> Dict[str, Any]:
    ts = t.timestamp
    ts_s = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    return {
        "timestamp": ts_s,
        "side": t.side,
        "price": round(t.price, 4),
        "shares": t.shares,
        "notional": round(t.notional, 2),
        "fee": round(t.fee, 2),
        "cash_after": round(t.cash_after, 2),
        "equity_after": round(t.equity_after, 2),
        "equity_gross_after": round(getattr(t, "equity_gross_after", t.equity_after), 2),
        "reason": t.reason,
        "skipped": t.skipped,
    }


def _img_b64(path: Optional[Path]) -> str:
    if path is None or not Path(path).is_file():
        return ""
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(Path(path).name)}" />'


def _metric_cell(block: Dict[str, Any], key: str, pct: bool = False) -> str:
    val = (block or {}).get(key)
    if pct:
        return _pct(val)
    if key == "sharpe":
        return _num(val, 2)
    if key == "n_trades":
        return str(val if val is not None else "n/a")
    return _num(val, 2) if val is not None else "n/a"


def render_kpi_markdown(split: Dict[str, Any]) -> List[str]:
    rows = [
        ("全样本 / 成本后", split.get("full_post_cost")),
        ("全样本 / 成本前", split.get("full_pre_cost")),
        ("样本内 / 成本后", split.get("is_post_cost")),
        ("样本内 / 成本前", split.get("is_pre_cost")),
        ("样本外 / 成本后", split.get("oos_post_cost")),
        ("样本外 / 成本前", split.get("oos_pre_cost")),
    ]
    lines = [
        "",
        f"- 样本内截止: {split.get('is_end')}；样本外起点: {split.get('oos_start')}",
        "",
        "| 分段 | 收益 | 年化 | 最大回撤 | 夏普 | 成交次数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, block in rows:
        lines.append(
            f"| {label} | {_metric_cell(block, 'total_return', True)} | "
            f"{_metric_cell(block, 'annualized_return', True)} | "
            f"{_metric_cell(block, 'max_drawdown', True)} | "
            f"{_metric_cell(block, 'sharpe')} | "
            f"{_metric_cell(block, 'n_trades')} |"
        )
    return lines


def _chart_rel(chart_path: Optional[str], output_dir: Path) -> Optional[str]:
    if not chart_path:
        return None
    path = Path(chart_path)
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return path.name


def render_markdown(payload: Dict[str, Any], config: BacktestConfig, output_dir: Optional[Path] = None) -> str:
    cfg = payload.get("config", {})
    names = config.symbol_names
    strategy_cfg = cfg.get("strategy") or {}
    strategy_name = strategy_cfg.get("name", config.strategy_name)
    lines: List[str] = [
        "# 策略回测报告",
        "",
        f"- 生成时间: {payload.get('generated_at', '')}",
        f"- 配置文件: `{cfg.get('config_path') or 'in-memory'}`",
        f"- 策略: `{strategy_name}`",
        f"- 区间: {cfg.get('start')} ~ {cfg.get('end')}（时区 {cfg.get('timezone')}）",
        f"- K 线周期: `{cfg.get('interval')}`",
        f"- 每股初始资金: {cfg.get('capital')}",
        f"- 每日最多成交: {cfg.get('max_trades_per_day')} 次",
        f"- 方向: 只做多，{cfg.get('signal_on')} 出信号，{cfg.get('fill_on')} 成交，{cfg.get('position_mode')}",
        f"- 成本: 佣金 {cfg.get('commission_rate')} + 印花税 {cfg.get('stamp_duty_rate')}（双边）",
        "",
        "## 基准（买入持有）",
        "",
    ]
    benches = payload.get("benchmarks", {})
    for key, spec in (config.benchmarks or {}).items():
        b = benches.get(key, {})
        lines.append(
            f"- {spec.get('name', key)} (`{key}`): 收益 {_pct(b.get('buy_hold_return'))}"
            f"，数据 {b.get('coverage', 'n/a')}，来源 `{b.get('source', 'n/a')}`"
        )
    lines.append(
        "- 均匀定投（每只股票独立）：起点收盘价可买手数 N，间隔 = 交易日数 / N，"
        "每隔该间隔买入 1 手（开盘成交，含佣金/印花税，剩余现金留在账户）"
    )

    lines += ["", "## 分股成绩（成本后 · 全样本）", ""]
    bench_keys = list((config.benchmarks or {}).keys())
    extra_headers = "".join(f" 相对{config.benchmarks[k].get('name', k)}超额 |" for k in bench_keys)
    lines.append(
        "| 股票 | 参数方案 | 策略收益 | 年化 | 最大回撤 | 夏普 | 交易次数 | 胜率 |"
        + extra_headers
        + " 自身买入持有 | 均匀定投 | 相对定投超额 |"
    )
    lines.append("|" + "---|" * (8 + len(bench_keys) + 3))

    for row in payload.get("results", []):
        m = row.get("metrics", {})
        name = names.get(row["symbol"], row["symbol"])
        extras = "".join(f" {_pct(m.get(f'excess_vs_{k}'))} |" for k in bench_keys)
        variant_label = row.get("variant_label") or row.get("variant") or row.get("scheme", "")
        lines.append(
            f"| {name} `{row['symbol']}` | {variant_label} | {_pct(m.get('total_return'))} | "
            f"{_pct(m.get('annualized_return'))} | {_pct(m.get('max_drawdown'))} | "
            f"{_num(m.get('sharpe'), 2)} | {m.get('n_trades', 0)} | "
            f"{_pct(m.get('win_rate')) if m.get('win_rate') is not None else 'n/a'} |"
            f"{extras} {_pct(m.get('buy_hold_return'))} | {_pct(m.get('dca_return'))} | "
            f"{_pct(m.get('excess_vs_dca'))} |"
        )

    for row in payload.get("results", []):
        name = names.get(row["symbol"], row["symbol"])
        variant_label = row.get("variant_label") or row.get("variant") or row.get("scheme", "")
        lines += [
            "",
            f"### {name} `{row['symbol']}` · {variant_label}",
            "",
            f"- K 线: {row.get('coverage')}，来源 `{row.get('bar_source')}`",
            f"- 期末权益: {row.get('metrics', {}).get('final_equity')}  "
            f"现金 {row.get('metrics', {}).get('final_cash')}  "
            f"持仓 {row.get('metrics', {}).get('final_shares')} 股",
            f"- 累计费用: {row.get('metrics', {}).get('cumulative_fees')}",
            f"- 均匀定投: 起点价 {row.get('metrics', {}).get('dca_start_price')}，"
            f"可买 {row.get('metrics', {}).get('dca_n_lots')} 手，"
            f"{row.get('metrics', {}).get('dca_n_trading_days')} 个交易日 / N = 间隔 "
            f"{row.get('metrics', {}).get('dca_interval')}，"
            f"实买 {row.get('metrics', {}).get('dca_n_lots_bought')} 手 / "
            f"{row.get('metrics', {}).get('dca_n_buy_days')} 天，"
            f"收益 {_pct(row.get('metrics', {}).get('dca_return'))}",
            f"- 图表: `{row.get('report_html', '')}`",
            "",
        ]
        lines += render_kpi_markdown(row.get("split") or {})

        if output_dir is not None:
            chart_titles = [
                ("kline", "日K线 + 均线 + 买卖点"),
                ("nav", "净值曲线（策略/基准，对数）+ 超额（对数）+ 回撤水下图"),
                ("heatmap", "逐年逐月收益热力图"),
                ("rolling", "滚动 12 个月收益、滚动 Sharpe、滚动 Beta"),
                ("position", "持仓数量 / 换手率 / 杠杆"),
                ("factor", "因子暴露时序（风格漂移代理）"),
            ]
            lines += ["", "#### 图表", ""]
            charts = row.get("charts") or {}
            for key, title in chart_titles:
                rel = _chart_rel(charts.get(key), output_dir)
                lines.append(f"**{title}**")
                if rel:
                    lines.append(f"![{title}]({rel})")
                else:
                    lines.append("无数据")
                lines.append("")

        lines += ["", "#### 数据缺口", ""]
        for gap in row.get("gaps") or []:
            lines.append(f"- {gap}")

        lines += ["", "#### 极端交易日", ""]
        extremes = row.get("extremes") or {}
        for label, key in (("最好", "best"), ("最差", "worst")):
            lines.append(f"**{label}**")
            lines.append("| 日期 | 当日收益 | 成交 |")
            lines.append("|---|---:|---|")
            for item in extremes.get(key) or []:
                trades = item.get("trades") or []
                detail = "；".join(f"{t.get('side')} {t.get('shares')}@{t.get('price')}" for t in trades) or "无"
                lines.append(f"| {item.get('date')} | {_pct(item.get('return'))} | {detail} |")
            lines.append("")

        trips = row.get("round_trips") or []
        lines.append("#### 开平仓回合")
        if not trips:
            lines.append("无完整开平仓回合。")
        else:
            lines.append("| 开仓 | 平仓 | 股数 | 成本前盈亏 | 成本后盈亏 | 净收益率 |")
            lines.append("|---|---|---:|---:|---:|---:|")
            for trip in trips:
                lines.append(
                    f"| {trip.get('entry')} | {trip.get('exit')} | {trip.get('shares')} "
                    f"| {trip.get('pnl_gross')} | {trip.get('pnl_net')} | {_pct(trip.get('return_net'))} |"
                )
            lines.append("")

        trades = row.get("trades") or []
        lines.append("#### 交易明细")
        if not trades:
            lines.append("无成交。")
        else:
            lines.append("| 时间 | 方向 | 价格 | 股数 | 费用 | 成本后权益 |")
            lines.append("|---|---|---:|---:|---:|---:|")
            for t in trades:
                lines.append(
                    f"| {t.get('timestamp', '')} | {t.get('side')} | {t.get('price')} "
                    f"| {t.get('shares')} | {t.get('fee')} | {t.get('equity_after')} |"
                )
    lines.append("")
    return "\n".join(lines)


def render_html_case(row: Dict[str, Any], config: BacktestConfig, chart_paths: Dict[str, Optional[Path]]) -> str:
    name = config.symbol_names.get(row["symbol"], row["symbol"])
    split = row.get("split") or {}
    kpi_rows = [
        ("全样本", "成本后", split.get("full_post_cost")),
        ("全样本", "成本前", split.get("full_pre_cost")),
        ("样本内", "成本后", split.get("is_post_cost")),
        ("样本内", "成本前", split.get("is_pre_cost")),
        ("样本外", "成本后", split.get("oos_post_cost")),
        ("样本外", "成本前", split.get("oos_pre_cost")),
    ]
    kpi_html = "".join(
        "<tr>"
        f"<td>{html.escape(a)}</td><td>{html.escape(b)}</td>"
        f"<td>{_metric_cell(block, 'total_return', True)}</td>"
        f"<td>{_metric_cell(block, 'annualized_return', True)}</td>"
        f"<td>{_metric_cell(block, 'max_drawdown', True)}</td>"
        f"<td>{_metric_cell(block, 'sharpe')}</td>"
        f"<td>{_metric_cell(block, 'n_trades')}</td>"
        "</tr>"
        for a, b, block in kpi_rows
    )

    def section(title: str, key: str) -> str:
        img = _img_b64(chart_paths.get(key))
        if not img:
            return f"<h2>{html.escape(title)}</h2><p class='muted'>无数据</p>"
        return f"<h2>{html.escape(title)}</h2><div class='chart'>{img}</div>"

    extremes = row.get("extremes") or {}
    ext_html = ""
    for label, key in (("最好", "best"), ("最差", "worst")):
        ext_html += f"<h3>{label}</h3><table><thead><tr><th>日期</th><th>当日收益</th><th>成交</th></tr></thead><tbody>"
        for item in extremes.get(key) or []:
            trades = item.get("trades") or []
            detail = "；".join(f"{t.get('side')} {t.get('shares')}@{t.get('price')}" for t in trades) or "无"
            ext_html += (
                f"<tr><td>{html.escape(str(item.get('date')))}</td>"
                f"<td>{_pct(item.get('return'))}</td>"
                f"<td>{html.escape(detail)}</td></tr>"
            )
        ext_html += "</tbody></table>"

    trade_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(t.get('timestamp', '')))}</td>"
        f"<td>{html.escape(str(t.get('side')))}</td>"
        f"<td>{t.get('price')}</td><td>{t.get('shares')}</td>"
        f"<td>{t.get('fee')}</td><td>{t.get('equity_after')}</td>"
        "</tr>"
        for t in (row.get("trades") or [])
    ) or "<tr><td colspan='6'>无成交</td></tr>"

    trip_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(t.get('entry', '')))}</td>"
        f"<td>{html.escape(str(t.get('exit', '')))}</td>"
        f"<td>{t.get('shares')}</td>"
        f"<td>{t.get('pnl_gross')}</td>"
        f"<td>{t.get('pnl_net')}</td>"
        f"<td>{_pct(t.get('return_net'))}</td>"
        "</tr>"
        for t in (row.get("round_trips") or [])
    ) or "<tr><td colspan='6'>无完整开平仓回合</td></tr>"

    gaps = "".join(f"<li>{html.escape(g)}</li>" for g in (row.get("gaps") or []))
    variant_label = row.get("variant_label") or row.get("variant") or row.get("scheme", "")
    m = row.get("metrics") or {}
    dca_interval = m.get("dca_interval")
    dca_interval_s = "n/a" if dca_interval is None else str(dca_interval)
    dca_note = (
        f"均匀定投：起点价 {html.escape(str(m.get('dca_start_price', 'n/a')))}，"
        f"可买 {html.escape(str(m.get('dca_n_lots', 'n/a')))} 手，"
        f"{html.escape(str(m.get('dca_n_trading_days', 'n/a')))} 个交易日 / N = 间隔 {html.escape(dca_interval_s)}，"
        f"实买 {html.escape(str(m.get('dca_n_lots_bought', 'n/a')))} 手 / "
        f"{html.escape(str(m.get('dca_n_buy_days', 'n/a')))} 天，"
        f"收益 {_pct(m.get('dca_return'))}，相对定投超额 {_pct(m.get('excess_vs_dca'))}"
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(name)} {html.escape(variant_label)} 回测</title>
  <style>
    body {{ background:#0f1419; color:#e2e8f0; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif; margin:0; padding:24px; }}
    h1,h2,h3 {{ color:#f8fafc; }}
    a {{ color:#7dd3fc; }}
    .muted {{ color:#94a3b8; }}
    table {{ border-collapse:collapse; width:100%; margin:12px 0 24px; font-size:13px; }}
    th,td {{ border:1px solid #334155; padding:8px 10px; text-align:left; }}
    th {{ background:#1e293b; }}
    td:nth-child(n+3) {{ text-align:right; }}
    .chart img {{ width:100%; max-width:1100px; border:1px solid #1e293b; border-radius:8px; }}
    .card {{ background:#151b22; border:1px solid #1e293b; border-radius:10px; padding:16px 20px; margin-bottom:20px; }}
  </style>
</head>
<body>
  <h1>{html.escape(name)} `{html.escape(row['symbol'])}` · {html.escape(variant_label)}</h1>
  <p class="muted">{html.escape(str(row.get('coverage', '')))} · 来源 {html.escape(str(row.get('bar_source', '')))}</p>
  <div class="card">
    <h2>关键指标（成本前/成本后，样本内/样本外）</h2>
    <p class="muted">样本内截止 {html.escape(str(split.get('is_end')))}，样本外起点 {html.escape(str(split.get('oos_start')))}</p>
    <p class="muted">{dca_note}</p>
    <table>
      <thead><tr><th>样本</th><th>成本</th><th>收益</th><th>年化</th><th>最大回撤</th><th>夏普</th><th>成交次数</th></tr></thead>
      <tbody>{kpi_html}</tbody>
    </table>
  </div>
  {section("日K线 + 均线 + 买卖点", "kline")}
  {section("净值曲线（策略/基准/超额，对数坐标）+ 回撤水下图", "nav")}
  {section("逐年逐月收益热力图", "heatmap")}
  {section("滚动 12 个月收益、滚动 Sharpe、滚动 Beta", "rolling")}
  {section("持仓数量 / 换手率 / 杠杆", "position")}
  {section("因子暴露时序（风格漂移代理）", "factor")}
  <div class="card">
    <h2>数据缺口</h2>
    <ul>{gaps}</ul>
  </div>
  <div class="card">
    <h2>极端交易日复盘</h2>
    {ext_html}
  </div>
  <div class="card">
    <h2>开平仓回合</h2>
    <table>
      <thead><tr><th>开仓</th><th>平仓</th><th>股数</th><th>成本前盈亏</th><th>成本后盈亏</th><th>净收益率</th></tr></thead>
      <tbody>{trip_rows}</tbody>
    </table>
  </div>
  <div class="card">
    <h2>交易明细</h2>
    <table>
      <thead><tr><th>时间</th><th>方向</th><th>价格</th><th>股数</th><th>费用</th><th>成本后权益</th></tr></thead>
      <tbody>{trade_rows}</tbody>
    </table>
  </div>
</body>
</html>
"""


def write_case_html(row: Dict[str, Any], config: BacktestConfig, out_dir: Path, chart_paths: Dict[str, Optional[Path]]) -> Path:
    path = out_dir / "report.html"
    path.write_text(render_html_case(row, config, chart_paths), encoding="utf-8")
    return path


def write_index_html(payload: Dict[str, Any], config: BacktestConfig, output_dir: Path) -> Path:
    rows = []
    for row in payload.get("results", []):
        m = row.get("metrics") or {}
        href = Path(row.get("report_html") or "").name
        if row.get("report_html"):
            rel = Path(row["report_html"])
            try:
                href = str(rel.relative_to(output_dir))
            except ValueError:
                href = rel.name
        name = config.symbol_names.get(row["symbol"], row["symbol"])
        variant_label = row.get("variant_label") or row.get("variant") or row.get("scheme", "")
        rows.append(
            "<tr>"
            f"<td><a href='{html.escape(href)}'>{html.escape(name)} {html.escape(row['symbol'])}</a></td>"
            f"<td>{html.escape(variant_label)}</td>"
            f"<td>{_pct(m.get('total_return'))}</td>"
            f"<td>{_pct(m.get('max_drawdown'))}</td>"
            f"<td>{_num(m.get('sharpe'), 2)}</td>"
            f"<td>{m.get('n_trades', 0)}</td>"
            "</tr>"
        )
    strategy_cfg = (payload.get("config") or {}).get("strategy") or {}
    strategy_name = strategy_cfg.get("name", config.strategy_name)
    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/><title>策略回测</title>
<style>
body {{ background:#0f1419; color:#e2e8f0; font-family:-apple-system,sans-serif; padding:24px; }}
a {{ color:#7dd3fc; }} table {{ border-collapse:collapse; width:100%; }}
th,td {{ border:1px solid #334155; padding:8px 10px; }} th {{ background:#1e293b; }}
</style></head>
<body>
<h1>策略回测索引</h1>
<p>{html.escape(str(payload.get('generated_at', '')))} · 策略 {html.escape(str(strategy_name))} · {html.escape(str((payload.get('config') or {}).get('start')))} ~ {html.escape(str((payload.get('config') or {}).get('end')))}</p>
<table><thead><tr><th>股票</th><th>参数方案</th><th>收益</th><th>最大回撤</th><th>夏普</th><th>成交</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p><a href="summary.md">Markdown 总览</a> · <a href="summary.json">JSON</a></p>
</body></html>
"""
    path = output_dir / "index.html"
    path.write_text(html_doc, encoding="utf-8")
    return path


def write_reports(output_dir: Path, payload: Dict[str, Any], config: BacktestConfig) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(payload, config, output_dir), encoding="utf-8")
    write_index_html(payload, config, output_dir)
    return md_path
