"""把基线结果写成单文件 HTML，打开即可看训练损失、IC 和回测曲线。"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd

_WIDTH = 880
_HEIGHT = 312
_PAD_L = 76
_PAD_R = 18
_PAD_T = 16
_PAD_B = 42


def _fmt_pct(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.2f}%"


def _fmt_num(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.4f}"


def _fmt_param(value) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return "—"
    return str(value)


_PARAM_LABELS = {
    "loss": "损失",
    "learning_rate": "学习率",
    "num_boost_round": "提升轮数上限",
    "early_stopping_rounds": "早停轮数",
    "num_leaves": "叶子数",
    "max_depth": "最大深度",
    "num_threads": "线程数",
    "colsample_bytree": "列采样",
    "subsample": "行采样",
    "lambda_l1": "L1",
    "lambda_l2": "L2",
    "estimator": "估计器",
    "eval_metric": "评估指标",
    "eta": "学习率",
    "n_estimators": "树的数量",
    "nthread": "线程数",
    "thread_count": "线程数",
    "grow_policy": "生长策略",
    "bootstrap_type": "采样方式",
}


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _frac_digits(step: float) -> int:
    step = abs(step)
    if step == 0 or step >= 10:
        return 0
    digits = 0
    scaled = step
    while digits < 4 and abs(scaled - round(scaled)) > 1e-6 * max(1.0, abs(scaled)):
        scaled *= 10
        digits += 1
    return digits


def _fmt_y(value: float, kind: str, step: float) -> str:
    if abs(value) < abs(step) * 1e-6:
        value = 0.0
    if kind == "pct":
        return f"{value * 100:.{_frac_digits(abs(step) * 100)}f}%"
    return f"{value:.{_frac_digits(step)}f}"


def _nice_step(span: float, count: int = 4) -> float:
    raw = span / max(count, 1)
    if raw <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(raw))
    residual = raw / magnitude
    if residual <= 1:
        nice = 1
    elif residual <= 2:
        nice = 2
    elif residual <= 2.5:
        nice = 2.5
    elif residual <= 5:
        nice = 5
    else:
        nice = 10
    return nice * magnitude


def _ticks(lo: float, hi: float, count: int = 4) -> tuple[list[float], float]:
    if hi <= lo:
        hi = lo + 1.0
    step = _nice_step(hi - lo, count)
    start = math.ceil((lo - step * 1e-8) / step) * step
    values: list[float] = []
    cursor = start
    for _ in range(24):
        if cursor > hi + step * 1e-6:
            break
        if cursor >= lo - step * 1e-6:
            values.append(0.0 if abs(cursor) < step * 1e-6 else cursor)
        cursor += step
    if len(values) < 2:
        values = [lo, hi]
        step = (hi - lo) or 1.0
    elif len(values) < 4:
        finer, finer_step = _ticks_raw(lo, hi, count * 2)
        if 4 <= len(finer) <= 6:
            return finer, finer_step
    elif len(values) > 6:
        return _ticks_raw(lo, hi, max(count - 1, 2))
    return values, step


def _ticks_raw(lo: float, hi: float, count: int) -> tuple[list[float], float]:
    step = _nice_step(hi - lo, count)
    start = math.ceil((lo - step * 1e-8) / step) * step
    values: list[float] = []
    cursor = start
    for _ in range(24):
        if cursor > hi + step * 1e-6:
            break
        if cursor >= lo - step * 1e-6:
            values.append(0.0 if abs(cursor) < step * 1e-6 else cursor)
        cursor += step
    if len(values) < 2:
        return [lo, hi], (hi - lo) or 1.0
    return values, step


def _even_ticks(n: int, count: int = 5) -> list[int]:
    if n <= 0:
        return []
    count = min(count, n)
    raw = [round(i * (n - 1) / max(count - 1, 1)) for i in range(count)]
    ticks: list[int] = []
    for index in raw:
        if not ticks or ticks[-1] != index:
            ticks.append(index)
    return ticks


def _round_ticks(n: int, count: int = 5) -> list[int]:
    """迭代轴用整齐的轮次，而不是把下标平均切开。"""
    if n <= 1:
        return [0] if n == 1 else []
    if n <= count:
        return list(range(n))
    step = max(1, int(round(_nice_step((n - 1) / (count - 1), 1))))
    ticks = {0, n - 1}
    cursor = step
    while cursor < n:
        ticks.add(cursor - 1)
        cursor += step
    return sorted(ticks)


def _y_range(series: list[tuple[str, list[float], str]]) -> tuple[float, float] | None:
    finite = [v for _name, values, _color in series for v in values if _is_finite(v)]
    if len(finite) < 2:
        return None
    lo, hi = min(finite), max(finite)
    if hi == lo:
        pad = abs(lo) * 0.05 or 1.0
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def _x_at(index: int, n: int, left: float, right: float) -> float:
    return left + (right - left) * (index / max(n - 1, 1))


def _y_at(value: float, lo: float, hi: float, top: float, bottom: float) -> float:
    return top + (bottom - top) * (1 - (value - lo) / (hi - lo))


def _polyline(
    values: list[float],
    n: int,
    lo: float,
    hi: float,
    box: tuple[float, float, float, float],
) -> str:
    left, top, right, bottom = box
    coords = []
    for index, value in enumerate(values):
        if not _is_finite(value):
            continue
        coords.append(f"{_x_at(index, n, left, right):.1f},{_y_at(value, lo, hi, top, bottom):.1f}")
    return " ".join(coords)


def _chart(
    series: list[tuple[str, list[float], str]],
    x_labels: list[str] | None = None,
    y_kind: str = "num",
    width: int = _WIDTH,
    height: int = _HEIGHT,
) -> str:
    legend = "".join(
        f'<span class="swatch" style="background:{html.escape(color)}"></span>{html.escape(name)}'
        for name, values, color in series
        if any(_is_finite(v) for v in values)
    )
    y_range = _y_range(series)
    if y_range is None:
        return (
            f'<div class="chart"><svg viewBox="0 0 {width} {height}" role="img"></svg>'
            f'<div class="legend">{legend}</div></div>'
        )

    lo, hi = y_range
    n = max([len(values) for _name, values, _color in series] + [len(x_labels or []), 1])
    left, top = float(_PAD_L), float(_PAD_T)
    right, bottom = float(width - _PAD_R), float(height - _PAD_B)
    box = (left, top, right, bottom)
    y_ticks, y_step = _ticks(lo, hi)
    x_ticks = _even_ticks(n) if x_labels else _round_ticks(n)

    grid: list[str] = []
    marks: list[str] = []
    texts: list[str] = []
    for tick in y_ticks:
        y = _y_at(tick, lo, hi, top, bottom)
        if top + 0.5 < y < bottom - 0.5:
            grid.append(
                f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{right:.1f}" y2="{y:.1f}" '
                'stroke="#ced4da" stroke-width="1" stroke-dasharray="4 4"/>'
            )
        marks.append(
            f'<line x1="{left - 4:.1f}" y1="{y:.1f}" x2="{left:.1f}" y2="{y:.1f}" '
            'stroke="#868e96" stroke-width="1"/>'
        )
        texts.append(
            f'<text x="{left - 8:.1f}" y="{y:.1f}" text-anchor="end" dominant-baseline="central" '
            f'fill="#868e96" font-size="11">{html.escape(_fmt_y(tick, y_kind, y_step))}</text>'
        )
    for index in x_ticks:
        x = _x_at(index, n, left, right)
        if left + 0.5 < x < right - 0.5:
            grid.append(
                f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}" '
                'stroke="#ced4da" stroke-width="1" stroke-dasharray="4 4"/>'
            )
        marks.append(
            f'<line x1="{x:.1f}" y1="{bottom:.1f}" x2="{x:.1f}" y2="{bottom + 4:.1f}" '
            'stroke="#868e96" stroke-width="1"/>'
        )
        if x_labels and index < len(x_labels) and x_labels[index]:
            label = x_labels[index]
        else:
            label = str(index + 1)
        if index == x_ticks[0]:
            anchor = "start"
        elif index == x_ticks[-1]:
            anchor = "end"
        else:
            anchor = "middle"
        texts.append(
            f'<text x="{x:.1f}" y="{bottom + 8:.1f}" text-anchor="{anchor}" dominant-baseline="hanging" '
            f'fill="#868e96" font-size="11">{html.escape(label)}</text>'
        )
    axes = (
        f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{bottom:.1f}" '
        'stroke="#868e96" stroke-width="1"/>'
        f'<line x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" y2="{bottom:.1f}" '
        'stroke="#868e96" stroke-width="1"/>'
    )
    lines = []
    for _name, values, color in series:
        points = _polyline(values, n, lo, hi, box)
        if points:
            lines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{points}"/>'
            )
    body = (
        f'<g shape-rendering="crispEdges">{"".join(grid)}{axes}{"".join(marks)}</g>'
        f'{"".join(lines)}{"".join(texts)}'
    )
    return (
        f'<div class="chart"><svg viewBox="0 0 {width} {height}" role="img">{body}</svg>'
        f'<div class="legend">{legend}</div></div>'
    )


def _date_labels(index) -> list[str]:
    return [pd.Timestamp(value).strftime("%Y-%m-%d") for value in index]


def _indexed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        return frame.set_index("datetime")
    frame.iloc[:, 0] = pd.to_datetime(frame.iloc[:, 0])
    return frame.set_index(frame.columns[0])


def _cash_weight(frame: pd.DataFrame) -> list[float] | None:
    if "cash" not in frame.columns or "account" not in frame.columns:
        return None
    account = frame["account"].astype(float)
    cash = frame["cash"].astype(float)
    weight = cash / account.where(account != 0)
    if not any(_is_finite(value) for value in weight):
        return None
    return [float(value) if _is_finite(value) else float("nan") for value in weight]


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
    if not ic_path.exists() and not bt_path.exists():
        return ""
    parts = [f"<h3>{html.escape(title)}</h3>"]
    if ic_path.exists():
        ic = _indexed(ic_path)
        parts.append("<h4>Rank IC</h4>")
        parts.append(
            _chart(
                [("Rank IC", ic["rank_ic"].astype(float).tolist(), "#364fc7")],
                x_labels=_date_labels(ic.index),
            )
        )
    if bt_path.exists():
        frame = _indexed(bt_path)
        equity = _equity(frame)
        dates = _date_labels(equity.index)
        parts.append("<h4>累计收益</h4>")
        parts.append(
            _chart(
                [
                    ("组合", equity["strategy"].tolist(), "#c92a2a"),
                    ("基准", equity["benchmark"].tolist(), "#868e96"),
                ],
                x_labels=dates,
                y_kind="pct",
            )
        )
        parts.append("<h4>回撤</h4>")
        parts.append(
            _chart(
                [("回撤", equity["drawdown"].tolist(), "#e03131")],
                x_labels=dates,
                y_kind="pct",
            )
        )
        cash = _cash_weight(frame)
        if cash is not None:
            parts.append("<h4>当日现金比重</h4>")
            parts.append(
                _chart(
                    [("现金", cash, "#0ca678")],
                    x_labels=dates,
                    y_kind="pct",
                )
            )
    return "\n".join(parts)


def _kv_table(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{html.escape(key)}</td>"
        f"<td>{html.escape(value)}</td>"
        "</tr>"
        for key, value in rows
        if value != ""
    )
    return f"<table class='kv'><tbody>{body}</tbody></table>"


def _universe_section(summary: dict) -> str:
    universe = summary.get("universe") or {}
    symbols_file = universe.get("symbols_file") or summary.get("symbols_file")
    if not universe and not symbols_file:
        return ""
    symbol_count = universe.get("symbol_count", summary.get("symbol_count"))
    amount = universe.get("min_median_amount")
    amount_text = "—" if amount is None else f"{int(amount):,} 港元"
    window = universe.get("median_amount_days")
    history = universe.get("min_history_days")
    excluded = "、".join(str(code) for code in (universe.get("exclude") or [])) or "无"
    volume = universe.get("require_positive_volume")
    rows = [
        ("名单", str(symbols_file or "tradable 全市场")),
        ("候选数量", "—" if symbol_count is None else f"{symbol_count} 只"),
        ("排除", excluded),
        ("最短历史", "—" if history is None else f"{history} 个交易日"),
        (
            "成交额",
            "—" if window is None else f"之前 {window} 个交易日的中位数不低于 {amount_text}",
        ),
        ("当日成交量", "必须大于 0" if volume else "不要求"),
    ]
    note = (
        "训练、标签标准化和选股只使用这份名单。"
        "名单里的股票还要在当天满足上面的历史、成交额和成交量条件。"
        if symbols_file
        else "没有单独名单时，使用数据目录里的 tradable，再套用上面的时点条件。"
    )
    return (
        "<section><h2>股票池</h2>"
        f"<p class='muted'>{html.escape(note)}</p>"
        f"{_kv_table(rows)}</section>"
    )


def _parameter_block(model: dict, summary: dict) -> str:
    parameters = model.get("parameters") or {}
    kwargs = parameters.get("kwargs") or {}
    if not parameters and not kwargs:
        return ""
    rows = []
    if parameters.get("version"):
        rows.append(("配置", str(parameters["version"])))
    rows.extend(
        [
            ("模型", str(parameters.get("class") or model.get("class") or model.get("model"))),
            ("模块", str(parameters.get("module_path") or "")),
        ]
    )
    rolling = model.get("rolling") or {}
    if isinstance(rolling, dict) and rolling.get("step") is not None:
        rows.append(("滚动", f"{rolling.get('rtype', 'expanding')}，步长 {rolling.get('step')} 个交易日"))
        rows.append(("重训次数", str(rolling.get("folds"))))
    if isinstance(rolling, dict) and rolling.get("early_stop_days") is not None:
        rows.append(("早停窗口", f"预测日前 {rolling['early_stop_days']} 个交易日"))
    holdings = model.get("holdings") or {}
    if holdings.get("topk") is not None:
        rows.append(("持有只数", str(holdings["topk"])))
        rows.append(("每日替换", str(holdings["n_drop"])))
    if holdings.get("account") is not None:
        rows.append(("资金", f"{int(holdings['account']):,} 港元"))
    for key, value in kwargs.items():
        label = _PARAM_LABELS.get(key, key)
        rows.append((f"{label}（{key}）", _fmt_param(value)))
    return f"<h3>模型参数</h3>{_kv_table(rows)}"


def write_report(path: Path, summary: dict, model_dirs: dict[str, Path]) -> None:
    sections = []
    for model in summary.get("models", []):
        name = model["model"]
        rows = []
        for segment, metrics in model.get("metrics", {}).items():
            rows.append(
                "<tr>"
                f"<td>{html.escape(segment)}</td>"
                f"<td>{html.escape('—' if metrics.get('names') is None else str(metrics.get('names')))}</td>"
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
            "<table><thead><tr><th>段</th><th>出现股票</th><th>Rank IC</th><th>Rank ICIR</th>"
            "<th>区间收益</th><th>基准</th><th>超额</th><th>最大回撤</th><th>换手</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            f"{_parameter_block(model, summary)}{body}</section>"
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
table.kv td {{ text-align: left; }}
table.kv td:first-child {{ width: 240px; color: #495057; }}
.muted {{ color: #868e96; }}
.chart {{ margin-top: 8px; }}
svg {{ width: 100%; height: auto; background: #f8f9fa; border-radius: 8px; }}
svg text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.legend {{ display: flex; gap: 16px; margin-top: 8px; color: #495057; font-size: 13px; }}
.swatch {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}
</style>
</head>
<body>
<main>
<h1>港股 Alpha158 基线</h1>
<p class="muted">生成于 {html.escape(str(summary.get("generated_at", "")))}</p>
{_universe_section(summary)}
{''.join(sections)}
</main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
