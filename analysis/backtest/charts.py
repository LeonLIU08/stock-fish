"""Matplotlib charts for backtest reports."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from analysis.backtest.analytics import nav_from_price, underwater
from analysis.backtest.engine import Trade
from analysis.backtest.signals import ChartOverlay

HK_TZ = "Asia/Hong_Kong"

MONTHS = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]


def _cjk_font_path() -> str | None:
    candidates = [
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def _setup_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcParams.update({
        "figure.facecolor": "#0f1419",
        "axes.facecolor": "#151b22",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#cbd5e1",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "text.color": "#e2e8f0",
        "grid.color": "#1e293b",
        "grid.linestyle": "--",
        "font.size": 10,
        "axes.titleweight": "bold",
        "savefig.facecolor": "#0f1419",
        "savefig.bbox": "tight",
        "legend.facecolor": "#151b22",
        "legend.edgecolor": "#334155",
        "axes.unicode_minus": False,
    })
    font_path = _cjk_font_path()
    if font_path:
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = font_name
    return plt


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _normalize_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series)
    if getattr(ts.dt, "tz", None) is None:
        return ts.dt.tz_localize(HK_TZ)
    return ts.dt.tz_convert(HK_TZ)


def _to_daily_bars(df: pd.DataFrame, overlay_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if df is None or df.empty:
        cols = ["datetime", "open", "high", "low", "close", "volume"]
        for col in overlay_columns or ():
            cols.append(col)
        return pd.DataFrame(columns=cols)

    work = df.copy()
    work["datetime"] = _normalize_ts(work["datetime"])
    work["_date"] = work["datetime"].dt.normalize()
    if work.groupby("_date").size().max() <= 1:
        out = work.drop(columns=["_date"]).copy()
        out["datetime"] = out["datetime"].dt.normalize()
        return out.reset_index(drop=True)

    agg: Dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    for col in overlay_columns or ():
        if col in work.columns:
            agg[col] = "last"
    daily = work.groupby("_date", as_index=False).agg(agg)
    daily = daily.rename(columns={"_date": "datetime"})
    return daily.sort_values("datetime").reset_index(drop=True)


def _trade_points(
    trades: Sequence[Union[Trade, Dict]],
    day_index: pd.DatetimeIndex,
) -> tuple[list[float], list[float], list[float], list[float]]:
    day_lookup = {pd.Timestamp(d).normalize(): i for i, d in enumerate(day_index)}
    buy_x: list[float] = []
    buy_y: list[float] = []
    sell_x: list[float] = []
    sell_y: list[float] = []
    for trade in trades or []:
        if isinstance(trade, Trade):
            if trade.skipped or trade.shares <= 0:
                continue
            side = trade.side
            price = float(trade.price)
            ts = pd.Timestamp(trade.timestamp)
        else:
            if trade.get("skipped") or int(trade.get("shares") or 0) <= 0:
                continue
            side = str(trade.get("side") or "")
            price = float(trade.get("price") or 0)
            ts = pd.Timestamp(trade.get("timestamp"))
        if price <= 0:
            continue
        if ts.tzinfo is None:
            ts = ts.tz_localize(HK_TZ)
        else:
            ts = ts.tz_convert(HK_TZ)
        x = day_lookup.get(ts.normalize())
        if x is None:
            continue
        if side == "buy":
            buy_x.append(float(x))
            buy_y.append(price)
        elif side == "sell":
            sell_x.append(float(x))
            sell_y.append(price)
    return buy_x, buy_y, sell_x, sell_y


def plot_daily_kline_with_trades(
    bars: pd.DataFrame,
    trades: Sequence[Union[Trade, Dict]],
    out_dir: Path,
    overlays: Optional[Sequence[ChartOverlay]] = None,
    ma_fast_n: Optional[int] = None,
    ma_slow_n: Optional[int] = None,
) -> Optional[Path]:
    overlay_cols = [item.column for item in (overlays or [])]
    daily = _to_daily_bars(bars, overlay_columns=overlay_cols)
    if daily.empty:
        return None

    plt = _setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.0]})
    ax = axes[0]
    vol_ax = axes[1]

    dates = pd.DatetimeIndex(_normalize_ts(daily["datetime"]).dt.normalize())
    xs = np.arange(len(daily))
    opens = daily["open"].astype(float).to_numpy()
    highs = daily["high"].astype(float).to_numpy()
    lows = daily["low"].astype(float).to_numpy()
    closes = daily["close"].astype(float).to_numpy()
    width = 0.62

    for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
        color = "#f87171" if c >= o else "#34d399"
        ax.vlines(xs[i], l, h, color=color, linewidth=0.9, zorder=2)
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height == 0:
            body_height = max((h - l) * 0.08, h * 0.0005, 0.01)
        ax.bar(xs[i], body_height, bottom=body_bottom, width=width, color=color, edgecolor=color, linewidth=0.8, zorder=3)

    if overlays:
        for item in overlays:
            if item.column in daily.columns and daily[item.column].notna().any():
                ax.plot(xs, daily[item.column], color=item.color, lw=1.1, label=item.label, zorder=4)
    else:
        # Back-compat for callers/tests that still pass MA window hints.
        if "ma_fast" in daily.columns and daily["ma_fast"].notna().any():
            fast_n = ma_fast_n
            if fast_n is None and "ma_fast_n" in bars.columns and not bars["ma_fast_n"].empty:
                fast_n = int(bars["ma_fast_n"].iloc[0])
            ax.plot(xs, daily["ma_fast"], color="#fbbf24", lw=1.1, label=f"MA{fast_n or '快'}", zorder=4)
        if "ma_slow" in daily.columns and daily["ma_slow"].notna().any():
            slow_n = ma_slow_n
            if slow_n is None and "ma_slow_n" in bars.columns and not bars["ma_slow_n"].empty:
                slow_n = int(bars["ma_slow_n"].iloc[0])
            ax.plot(xs, daily["ma_slow"], color="#60a5fa", lw=1.1, label=f"MA{slow_n or '慢'}", zorder=4)

    buy_x, buy_y, sell_x, sell_y = _trade_points(trades, dates)
    if buy_x:
        ax.scatter(buy_x, buy_y, marker="^", s=72, color="#ef4444", edgecolors="#fecaca", linewidths=0.8, label="买入", zorder=6)
    if sell_x:
        ax.scatter(sell_x, sell_y, marker="v", s=72, color="#22c55e", edgecolors="#bbf7d0", linewidths=0.8, label="卖出", zorder=6)

    ax.set_title("日K线 + 均线 + 买卖点")
    ax.set_ylabel("价格")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", fontsize=8, ncol=3)

    if "volume" in daily.columns:
        vol_colors = ["#f87171" if c >= o else "#34d399" for o, c in zip(opens, closes)]
        vol_ax.bar(xs, daily["volume"].astype(float), width=width, color=vol_colors, alpha=0.85)
    vol_ax.set_ylabel("成交量")
    vol_ax.grid(True, alpha=0.35)

    tick_step = max(len(xs) // 10, 1)
    tick_idx = xs[::tick_step]
    tick_labels = [dates[i].strftime("%Y-%m-%d") for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    fig.tight_layout()
    return _save(fig, out_dir / "daily_kline_trades.png")


def plot_nav_and_underwater(
    strat_nav: pd.Series,
    benches: Dict[str, pd.Series],
    out_dir: Path,
) -> Path:
    plt = _setup_style()
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.2), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.2, 1.0]})
    ax = axes[0]
    if not strat_nav.empty:
        ax.plot(strat_nav.index, strat_nav.values, color="#f87171", lw=1.6, label="策略(成本后)")
    colors = ["#60a5fa", "#fbbf24", "#34d399", "#c084fc"]
    aligned_benches = {}
    for i, (name, series) in enumerate(benches.items()):
        if series is None or series.empty:
            continue
        aligned = series.reindex(strat_nav.index).ffill() if not strat_nav.empty else series
        if aligned.dropna().empty:
            continue
        aligned_benches[name] = (aligned, colors[i % len(colors)])
        ax.plot(aligned.index, aligned.values, color=colors[i % len(colors)], lw=1.2, label=name)
    ax.set_yscale("log")
    ax.set_ylabel("净值 (对数)")
    ax.set_title("净值曲线 · 策略 / 基准")
    ax.grid(True, which="both")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    ax_ex = axes[1]
    if not strat_nav.empty:
        for name, (aligned, color) in aligned_benches.items():
            excess = strat_nav / aligned.replace(0, np.nan)
            ax_ex.plot(excess.index, excess.values, color=color, lw=1.2, label=f"超额 vs {name}")
    ax_ex.axhline(1.0, color="#64748b", lw=0.6)
    ax_ex.set_yscale("log")
    ax_ex.set_ylabel("相对净值 (对数)")
    ax_ex.set_title("超额曲线 · 策略 / 基准")
    ax_ex.grid(True, which="both")
    ax_ex.legend(loc="upper left", fontsize=8)

    ax2 = axes[2]
    dd = underwater(strat_nav)
    if not dd.empty:
        ax2.fill_between(dd.index, dd.values, 0, color="#ef4444", alpha=0.55)
        ax2.plot(dd.index, dd.values, color="#fca5a5", lw=0.8)
    ax2.set_ylabel("回撤")
    ax2.set_title("回撤水下图 (underwater)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _p: f"{x:.0%}"))
    ax2.grid(True)
    fig.tight_layout()
    return _save(fig, out_dir / "nav_underwater.png")


def plot_monthly_heatmap(pivot: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    if pivot is None or pivot.empty:
        return None
    plt = _setup_style()
    fig, ax = plt.subplots(figsize=(12.5, max(2.6, 0.55 * len(pivot) + 1.4)))
    data = pivot.values.astype(float)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 0.01
    vmax = max(vmax, 0.01)
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
    labels = []
    for col in pivot.columns:
        if col == "全年":
            labels.append("全年")
        else:
            try:
                labels.append(MONTHS[int(col) - 1])
            except (TypeError, ValueError, IndexError):
                labels.append(str(col))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(y) for y in pivot.index])
    ax.set_title("逐年逐月收益热力图")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isfinite(val):
                continue
            ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=8, color="#0f172a")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, format=plt.FuncFormatter(lambda x, _p: f"{x:.0%}"))
    fig.tight_layout()
    return _save(fig, out_dir / "monthly_heatmap.png")


def plot_rolling(roll: pd.DataFrame, window: int, out_dir: Path) -> Optional[Path]:
    if roll is None or roll.empty:
        return None
    plt = _setup_style()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(roll.index, roll["roll_return"].values if "roll_return" in roll.columns else [], color="#38bdf8", lw=1.3)
    axes[0].axhline(0, color="#64748b", lw=0.6)
    axes[0].set_title(f"滚动 {window} 日收益")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _p: f"{x:.0%}"))
    axes[0].grid(True)

    axes[1].plot(roll.index, roll["roll_sharpe"].values if "roll_sharpe" in roll.columns else [], color="#38bdf8", lw=1.3)
    axes[1].axhline(0, color="#64748b", lw=0.6)
    axes[1].set_title(f"滚动 {window} 日 Sharpe")
    axes[1].grid(True)

    beta_cols = [c for c in roll.columns if c.startswith("roll_beta")]
    colors = ["#60a5fa", "#fbbf24", "#34d399"]
    if beta_cols:
        for i, col in enumerate(beta_cols):
            axes[2].plot(roll.index, roll[col].values, color=colors[i % len(colors)], lw=1.2, label=col.replace("roll_beta_", "Beta "))
        axes[2].legend(fontsize=8)
    axes[2].axhline(1, color="#64748b", lw=0.6, ls="--")
    axes[2].set_title(f"滚动 {window} 日 Beta")
    axes[2].grid(True)
    fig.tight_layout()
    return _save(fig, out_dir / "rolling.png")


def plot_position_turnover_leverage(daily: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    if daily is None or daily.empty:
        return None
    plt = _setup_style()
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    axes[0].step(daily.index, daily.get("position", 0), where="post", color="#f87171", lw=1.2)
    axes[0].set_title("持仓数量（单票 0/1）")
    axes[0].set_ylim(-0.05, 1.15)
    axes[0].grid(True)

    turn = daily["turnover_d"] if "turnover_d" in daily.columns else daily.get("turnover", 0)
    axes[1].bar(daily.index, turn, color="#fbbf24", width=1.0)
    axes[1].set_title("换手率（当日成交额 / 初始资金）")
    axes[1].grid(True)

    axes[2].plot(daily.index, daily.get("leverage", 0), color="#a78bfa", lw=1.2)
    axes[2].set_title("杠杆（市值 / 初始资金）")
    axes[2].grid(True)
    fig.tight_layout()
    return _save(fig, out_dir / "position_turnover_leverage.png")


def plot_factor_exposure(exposures: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    if exposures is None or exposures.empty:
        return None
    plt = _setup_style()
    cols = [c for c in ("trend_gap", "realized_vol_20", "position") if c in exposures.columns]
    if not cols:
        return None
    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 2.4 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    titles = {
        "trend_gap": "趋势暴露 (MA快 - MA慢) / MA慢",
        "realized_vol_20": "20日实现波动（年化）",
        "position": "仓位暴露",
    }
    palette = {"trend_gap": "#34d399", "realized_vol_20": "#f59e0b", "position": "#f87171"}
    for ax, col in zip(axes, cols):
        ax.plot(exposures.index, exposures[col].values, color=palette.get(col, "#38bdf8"), lw=1.2)
        ax.set_title(titles.get(col, col))
        ax.grid(True)
    fig.suptitle("因子暴露时序（风格漂移代理）", y=1.01)
    fig.tight_layout()
    return _save(fig, out_dir / "factor_exposure.png")
