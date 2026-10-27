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
    payload = {
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
    if getattr(t, "action", ""):
        payload.update({
            "action": t.action,
            "tactic": getattr(t, "tactic", ""),
            "stop_price": None if t.stop_price is None else round(float(t.stop_price), 4),
            "planned_risk": t.planned_risk,
            "position_fraction": t.position_fraction,
            "basis": getattr(t, "basis", ""),
            "one_lot": bool(getattr(t, "one_lot", False)),
            "shares_after": t.shares_after,
        })
    return payload


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


def _trend_trades(trades: List[Dict[str, Any]]) -> bool:
    return any(item.get("action") or item.get("basis") for item in trades)


def _trend_trips(trips: List[Dict[str, Any]]) -> bool:
    return any(item.get("exit_reason") for item in trips)


def _pct_or_blank(value) -> str:
    if value is None:
        return ""
    return _pct(value)


def _audit_markdown(row: Dict[str, Any]) -> str:
    audit = row.get("audit")
    if not audit:
        return ""
    cancels = audit.get("cancels") or {}
    blocked = audit.get("blocked") or {}
    lines = [
        "#### 衰竭减仓",
        "",
        f"- 本 run {'启用' if audit.get('exhaustion_enabled') else '未启用'}衰竭减仓",
    ]
    if audit.get("exhaustion_enabled"):
        lines.append(f"- 减仓次数: {audit.get('exhaustion_reduces', 0)}")
    lines += [
        "",
        "#### 取消与跳过",
        "",
        f"- 跳空: {cancels.get('跳空', 0)}",
        f"- 开盘价不高于止损: {cancels.get('开盘价不高于止损', 0)}",
        f"- 止损距离: {cancels.get('止损距离', 0)}",
        f"- 股数为 0 / 风险预算: {cancels.get('风险预算', 0)}",
        f"- 股数为 0 / 现金: {cancels.get('现金', 0)}",
        f"- 股数为 0 / 占用上限: {cancels.get('占用上限', 0)}",
        f"- 环境不允许: {blocked.get('环境不允许', 0)}",
        f"- 按一手规则成交: {audit.get('one_lot_fills', 0)}",
        "",
        f"- 最大连续亏损次数: {audit.get('max_consecutive_losses', 0)}",
    ]
    return "\n".join(lines)


def _audit_html(row: Dict[str, Any]) -> str:
    text = _audit_markdown(row)
    if not text:
        return ""
    parts = ["<div class='card'>"]
    for line in text.splitlines():
        if line.startswith("#### "):
            parts.append(f"<h2>{html.escape(line[5:])}</h2>")
        elif line.startswith("- "):
            parts.append(f"<p>{html.escape(line[2:])}</p>")
    parts.append("</div>")
    return "".join(parts)


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
    comparison = payload.get("comparison_markdown")
    if comparison:
        lines += ["", comparison]
    failures = payload.get("failures") or []
    if failures:
        lines += ["", "## 跳过", "", "这些股票没有写入账户结果。", ""]
        for item in failures:
            name = item.get("name") or item.get("symbol")
            lines.append(f"- {name} `{item.get('symbol')}`：{item.get('error')}")

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
        elif _trend_trips(trips):
            lines.append("| 开仓 | 平仓 | 股数 | 成本前盈亏 | 成本后盈亏 | 净收益率 | 退出原因 | 计划风险 | 一手规则 | 实际亏损大于计划风险 |")
            lines.append("|---|---|---:|---:|---:|---:|---|---:|---|---|")
            for trip in trips:
                lines.append(
                    f"| {trip.get('entry')} | {trip.get('exit')} | {trip.get('shares')} "
                    f"| {trip.get('pnl_gross')} | {trip.get('pnl_net')} | {_pct(trip.get('return_net'))} "
                    f"| {trip.get('exit_reason') or ''} | {_pct_or_blank(trip.get('planned_risk'))} "
                    f"| {'是' if trip.get('one_lot') else '否'} "
                    f"| {'是' if trip.get('loss_exceeds_plan') else '否'} |"
                )
            lines.append("")
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
        elif _trend_trades(trades):
            lines.append("| 时间 | 动作 | 策略 | 价格 | 股数 | 仓位占比 | 计划风险 | 生效止损 | 依据 |")
            lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
            for t in trades:
                lines.append(
                    f"| {t.get('timestamp', '')} | {t.get('action') or t.get('side')} | {t.get('tactic', '')} "
                    f"| {t.get('price')} | {t.get('shares')} | {_pct_or_blank(t.get('position_fraction'))} "
                    f"| {_pct_or_blank(t.get('planned_risk'))} | {t.get('stop_price') if t.get('stop_price') is not None else ''} "
                    f"| {t.get('basis') or t.get('reason') or ''} |"
                )
        else:
            lines.append("| 时间 | 方向 | 价格 | 股数 | 费用 | 成本后权益 |")
            lines.append("|---|---|---:|---:|---:|---:|")
            for t in trades:
                lines.append(
                    f"| {t.get('timestamp', '')} | {t.get('side')} | {t.get('price')} "
                    f"| {t.get('shares')} | {t.get('fee')} | {t.get('equity_after')} |"
                )
        audit_md = _audit_markdown(row)
        if audit_md:
            lines += ["", audit_md]
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
        path = chart_paths.get(key)
        if path is None or not Path(path).is_file():
            return f"<h2>{html.escape(title)}</h2><p class='muted'>无数据</p>"
        name = Path(path).name
        img = (
            f'<img src="{html.escape(name, quote=True)}" '
            f'alt="{html.escape(title)}" />'
        )
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

    trades = row.get("trades") or []
    trend_trades = _trend_trades(trades)
    if trend_trades:
        trade_head = "<tr><th>时间</th><th>动作</th><th>策略</th><th>价格</th><th>股数</th><th>仓位占比</th><th>计划风险</th><th>生效止损</th><th>依据</th></tr>"
        trade_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(t.get('timestamp', '')))}</td>"
            f"<td>{html.escape(str(t.get('action') or t.get('side')))}</td>"
            f"<td>{html.escape(str(t.get('tactic', '')))}</td>"
            f"<td>{t.get('price')}</td><td>{t.get('shares')}</td>"
            f"<td>{_pct_or_blank(t.get('position_fraction'))}</td>"
            f"<td>{_pct_or_blank(t.get('planned_risk'))}</td>"
            f"<td>{'' if t.get('stop_price') is None else t.get('stop_price')}</td>"
            f"<td>{html.escape(str(t.get('basis') or t.get('reason') or ''))}</td>"
            "</tr>"
            for t in trades
        ) or "<tr><td colspan='9'>无成交</td></tr>"
    else:
        trade_head = "<tr><th>时间</th><th>方向</th><th>价格</th><th>股数</th><th>费用</th><th>成本后权益</th></tr>"
        trade_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(t.get('timestamp', '')))}</td>"
            f"<td>{html.escape(str(t.get('side')))}</td>"
            f"<td>{t.get('price')}</td><td>{t.get('shares')}</td>"
            f"<td>{t.get('fee')}</td><td>{t.get('equity_after')}</td>"
            "</tr>"
            for t in trades
        ) or "<tr><td colspan='6'>无成交</td></tr>"

    trips = row.get("round_trips") or []
    if _trend_trips(trips):
        trip_head = (
            "<tr><th>开仓</th><th>平仓</th><th>股数</th><th>成本前盈亏</th><th>成本后盈亏</th>"
            "<th>净收益率</th><th>退出原因</th><th>计划风险</th><th>一手规则</th><th>实际亏损大于计划风险</th></tr>"
        )
        trip_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(t.get('entry', '')))}</td>"
            f"<td>{html.escape(str(t.get('exit', '')))}</td>"
            f"<td>{t.get('shares')}</td>"
            f"<td>{t.get('pnl_gross')}</td>"
            f"<td>{t.get('pnl_net')}</td>"
            f"<td>{_pct(t.get('return_net'))}</td>"
            f"<td>{html.escape(str(t.get('exit_reason') or ''))}</td>"
            f"<td>{_pct_or_blank(t.get('planned_risk'))}</td>"
            f"<td>{'是' if t.get('one_lot') else '否'}</td>"
            f"<td>{'是' if t.get('loss_exceeds_plan') else '否'}</td>"
            "</tr>"
            for t in trips
        ) or "<tr><td colspan='10'>无完整开平仓回合</td></tr>"
    else:
        trip_head = "<tr><th>开仓</th><th>平仓</th><th>股数</th><th>成本前盈亏</th><th>成本后盈亏</th><th>净收益率</th></tr>"
        trip_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(t.get('entry', '')))}</td>"
            f"<td>{html.escape(str(t.get('exit', '')))}</td>"
            f"<td>{t.get('shares')}</td>"
            f"<td>{t.get('pnl_gross')}</td>"
            f"<td>{t.get('pnl_net')}</td>"
            f"<td>{_pct(t.get('return_net'))}</td>"
            "</tr>"
            for t in trips
        ) or "<tr><td colspan='6'>无完整开平仓回合</td></tr>"
    preamble = "".join(f"<p class='muted'>{html.escape(str(line))}</p>" for line in (row.get("preamble") or []))
    back = ""
    if row.get("summary_href"):
        back = f"<p><a href=\"{html.escape(str(row['summary_href']), quote=True)}\">返回汇总</a></p>"
    audit_html = _audit_html(row)

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
  {back}
  <h1>{html.escape(name)} `{html.escape(row['symbol'])}` · {html.escape(variant_label)}</h1>
  <p class="muted">{html.escape(str(row.get('coverage', '')))} · 来源 {html.escape(str(row.get('bar_source', '')))}</p>
  {preamble}
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
      <thead>{trip_head}</thead>
      <tbody>{trip_rows}</tbody>
    </table>
  </div>
  <div class="card">
    <h2>交易明细</h2>
    <table>
      <thead>{trade_head}</thead>
      <tbody>{trade_rows}</tbody>
    </table>
  </div>
  {audit_html}
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
{payload.get('comparison_html') or ''}
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
