"""把基线结果写成单文件 HTML，打开即可看训练损失、IC 和回测曲线。"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


def _fmt_pct(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.2f}%"


def _fmt_num(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.4f}"


def _polyline(values: list[float], width: int, height: int, pad: int = 28) -> str:
    finite = [v for v in values if v == v]
    if len(finite) < 2:
        return ""
    lo, hi = min(finite), max(finite)
    if hi == lo:
        hi = lo + 1.0
    span = hi - lo
    inner_w = width - pad * 2
    inner_h = height - pad * 2
    coords = []
    usable = [(i, v) for i, v in enumerate(values) if v == v]
    for i, v in usable:
        x = pad + inner_w * (i / max(len(values) - 1, 1))
        y = pad + inner_h * (1 - (v - lo) / span)
        coords.append(f"{x:.1f},{y:.1f}")
    return " ".join(coords)


def _chart(series: list[tuple[str, list[float], str]], width: int = 880, height: int = 260) -> str:
    labels = "".join(
        f'<span class="swatch" style="background:{html.escape(color)}"></span>{html.escape(name)}'
        for name, _values, color in series
        if any(v == v for v in _values)
    )
    lines = []
    for _name, values, color in series:
        points = _polyline(values, width, height)
        if points:
            lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{points}"/>'
            )
    return (
        f'<div class="chart"><svg viewBox="0 0 {width} {height}" role="img">{"".join(lines)}</svg>'
        f'<div class="legend">{labels}</div></div>'
    )


def _indexed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        return frame.set_index("datetime")
    frame.iloc[:, 0] = pd.to_datetime(frame.iloc[:, 0])
    return frame.set_index(frame.columns[0])


def _equity(frame: pd.DataFrame) -> pd.DataFrame:
    ret = frame["return"].astype(float)
    if "cost" in frame.columns:
        ret = ret - frame["cost"].astype(float)
    bench = frame["bench"].astype(float) if "bench" in frame.columns else 0.0
    out = pd.DataFrame(
        {
            "strategy": (1.0 + ret).cumprod() - 1.0,
            "benchmark": (1.0 + bench).cumprod() - 1.0,
        },
        index=frame.index,
    )
    out["drawdown"] = (1.0 + out["strategy"]) / (1.0 + out["strategy"]).cummax() - 1.0
    return out


def _loss_block(curves: dict) -> str:
    if not curves:
        return "<p class='muted'>这次训练没有记下损失曲线。</p>"
    series = []
    palette = {"train": "#4c6ef5", "valid": "#f08c00"}
    for name, metrics in curves.items():
        values = metrics.get("l2") or next(iter(metrics.values()), [])
        series.append((str(name), [float(v) for v in values], palette.get(name, "#495057")))
    return "<h3>训练损失</h3>" + _chart(series)


def _segment_block(name: str, title: str, model_dir: Path) -> str:
    ic_path = model_dir / f"{name}_ic.csv"
    bt_path = model_dir / f"{name}_backtest.csv"
    parts = [f"<h3>{html.escape(title)}</h3>"]
    if ic_path.exists():
        ic = _indexed(ic_path)
        parts.append("<h4>Rank IC</h4>")
        parts.append(_chart([("Rank IC", ic["rank_ic"].astype(float).tolist(), "#364fc7")]))
    if bt_path.exists():
        equity = _equity(_indexed(bt_path))
        parts.append("<h4>累计收益</h4>")
        parts.append(
            _chart(
                [
                    ("组合", equity["strategy"].tolist(), "#c92a2a"),
                    ("基准", equity["benchmark"].tolist(), "#868e96"),
                ]
            )
        )
        parts.append("<h4>回撤</h4>")
        parts.append(_chart([("回撤", equity["drawdown"].tolist(), "#e03131")]))
    return "\n".join(parts)


def write_report(path: Path, summary: dict, model_dirs: dict[str, Path]) -> None:
    sections = []
    for model in summary.get("models", []):
        name = model["model"]
        rows = []
        for segment, metrics in model.get("metrics", {}).items():
            rows.append(
                "<tr>"
                f"<td>{html.escape(segment)}</td>"
                f"<td>{_fmt_num(metrics.get('rank_ic'))}</td>"
                f"<td>{_fmt_num(metrics.get('rank_icir'))}</td>"
                f"<td>{_fmt_pct(metrics.get('interval_return'))}</td>"
                f"<td>{_fmt_pct(metrics.get('benchmark_return'))}</td>"
                f"<td>{_fmt_pct(metrics.get('excess_return'))}</td>"
                f"<td>{_fmt_pct(metrics.get('max_drawdown'))}</td>"
                f"<td>{_fmt_num(metrics.get('turnover'))}</td>"
                "</tr>"
            )
        segments = model.get("segments") or {}
        seg_text = "，".join(
            f"{key} {value[0]} ~ {value[1]}" for key, value in segments.items()
        )
        model_dir = model_dirs[name]
        body = _loss_block(model.get("train_curves") or {})
        body += _segment_block("valid", "验证集", model_dir)
        body += _segment_block("test", "测试集", model_dir)
        sections.append(
            f"<section><h2>{html.escape(name)}</h2>"
            f"<p class='muted'>{html.escape(seg_text)}</p>"
            "<table><thead><tr><th>段</th><th>Rank IC</th><th>Rank ICIR</th>"
            "<th>区间收益</th><th>基准</th><th>超额</th><th>最大回撤</th><th>换手</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>{body}</section>"
        )
    document = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>港股 Alpha158 基线</title>
<style>
body {{ margin: 0; font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #212529; background: #f8f9fa; }}
main {{ max-width: 960px; margin: 0 auto; padding: 32px 20px 64px; }}
h1 {{ font-size: 28px; margin-bottom: 4px; }}
h2 {{ font-size: 22px; margin: 0 0 8px; }}
h3 {{ margin: 28px 0 8px; font-size: 18px; }}
h4 {{ margin: 18px 0 6px; font-size: 15px; color: #495057; }}
section {{ background: #fff; border: 1px solid #e9ecef; border-radius: 12px; padding: 20px 22px 28px; margin-top: 18px; }}
table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: right; padding: 8px 10px; border-bottom: 1px solid #f1f3f5; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: #495057; font-weight: 600; }}
.muted {{ color: #868e96; }}
.chart {{ margin-top: 8px; }}
svg {{ width: 100%; height: auto; background: #f8f9fa; border-radius: 8px; }}
.legend {{ display: flex; gap: 16px; margin-top: 8px; color: #495057; font-size: 13px; }}
.swatch {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}
</style>
</head>
<body>
<main>
<h1>港股 Alpha158 基线</h1>
<p class="muted">生成于 {html.escape(str(summary.get("generated_at", "")))}</p>
{''.join(sections)}
</main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
