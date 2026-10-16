"""用 baseline.yaml 在已落盘的港股 qlib 数据上跑表格模型基线。

在镜像 zhuhai123/qlib-rdagent:v1 里执行。标签、切分、费用和模型参数都读配置，
不在这里另写一套数字。默认只跑 lightgbm，把整条训练、验证、测试和回测跑通。

    python3 qlib-zh/hk/run_baseline.py --model lightgbm
    python3 qlib-zh/hk/run_baseline.py --model all
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置不是映射: {path}")
    return cfg


def provider_dir(cfg: dict) -> Path:
    uri = str(cfg["runtime"]["provider_uri"])
    path = Path(uri)
    if not path.exists():
        home = Path.home() / ".qlib" / "qlib_data" / cfg["data"]["freeze_dirname"]
        if home.exists():
            return home
    return path


def load_calendar(path: Path) -> pd.DatetimeIndex:
    days = pd.to_datetime(path.read_text(encoding="utf-8").splitlines())
    return pd.DatetimeIndex(days).sort_values().unique()


def trading_day_on_or_before(calendar: pd.DatetimeIndex, day: str) -> pd.Timestamp:
    pos = int(calendar.searchsorted(pd.Timestamp(day), side="right")) - 1
    if pos < 0:
        raise ValueError(f"{day} 早于交易日历")
    return calendar[pos]


def embargo_end(calendar: pd.DatetimeIndex, day: str, horizon: int) -> pd.Timestamp:
    """标签会看到之后 horizon 个交易日，特征日要停在这段结束之前。"""
    last = trading_day_on_or_before(calendar, day)
    pos = int(calendar.get_loc(last))
    pos -= int(horizon)
    if pos < 0:
        raise ValueError(f"{day} 往前空出 {horizon} 个交易日后越界")
    return calendar[pos]


def side_cost(costs: dict) -> float:
    total = (
        float(costs["commission_rate"])
        + float(costs["exchange_fee_rate"])
        + float(costs["settlement_rate"])
        + float(costs["slippage_rate"])
    )
    if costs.get("stamp_both_sides", True):
        total += float(costs["stamp_duty_rate"])
    return total


def universe_expression(universe: dict) -> str:
    parts = []
    history = int(universe.get("min_history_days") or 0)
    if history > 1:
        parts.append(f"Ref($close, {history - 1})>0")
    amount = universe.get("min_median_amount")
    window = int(universe.get("median_amount_days") or 0)
    if amount is not None and window > 0:
        parts.append(f"Mean($close*$volume, {window})>={float(amount)}")
    if universe.get("require_positive_volume", False):
        parts.append("$volume>0")
    expr = parts[0]
    for part in parts[1:]:
        expr = f"And({expr}, {part})"
    return expr


def segments_for(cfg: dict, calendar: pd.DatetimeIndex) -> dict[str, tuple[str, str]]:
    horizon = int(cfg["label"]["horizon_trading_days"])
    raw = cfg["segments"]
    train_end = embargo_end(calendar, str(raw["train"]["end"])[:10], horizon)
    valid_end = embargo_end(calendar, str(raw["valid"]["end"])[:10], horizon)
    test_end_raw = raw["test"].get("end")
    if test_end_raw:
        test_end = embargo_end(calendar, str(test_end_raw)[:10], horizon)
    else:
        test_end = calendar[-1]
    resolved = {
        "train": (str(raw["train"]["start"])[:10], train_end.strftime("%Y-%m-%d")),
        "valid": (str(raw["valid"]["start"])[:10], valid_end.strftime("%Y-%m-%d")),
        "test": (str(raw["test"]["start"])[:10], test_end.strftime("%Y-%m-%d")),
    }
    return resolved


def selected_models(cfg: dict, choice: str) -> list[tuple[str, dict]]:
    models = []
    for name, spec in cfg["models"].items():
        if spec.get("dataset_kind", "tabular") != "tabular":
            continue
        if choice != "all" and name != choice:
            continue
        if choice == "all" and not spec.get("enabled", False):
            continue
        if choice != "all" and name == choice and not spec.get("enabled", False):
            raise ValueError(f"模型 {name} 在配置里是 enabled: false")
        models.append((name, spec))
    if not models:
        raise ValueError(f"没有可跑的模型: {choice}")
    return models


def score_frame(pred) -> pd.Series:
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    pred = pred.astype(float)
    pred.index = pred.index.set_names(["datetime", "instrument"])
    return pred.sort_index()


def ic_summary(pred: pd.Series, label: pd.Series) -> dict:
    frame = pd.concat([pred.rename("score"), label.rename("label")], axis=1).dropna()
    if frame.empty:
        return {"days": 0, "ic": None, "rank_ic": None, "icir": None, "rank_icir": None}

    def _one(group: pd.DataFrame) -> pd.Series:
        if len(group) < 5:
            return pd.Series({"ic": np.nan, "rank_ic": np.nan})
        return pd.Series(
            {
                "ic": group["score"].corr(group["label"]),
                "rank_ic": group["score"].corr(group["label"], method="spearman"),
            }
        )

    daily = frame.groupby(level="datetime", sort=True).apply(_one)
    ic = daily["ic"].dropna()
    rank_ic = daily["rank_ic"].dropna()

    def _ir(series: pd.Series) -> float | None:
        if len(series) < 2 or float(series.std()) == 0:
            return None
        return float(series.mean() / series.std())

    return {
        "days": int(len(rank_ic)),
        "names": int(frame.index.get_level_values("instrument").nunique()),
        "ic": None if ic.empty else float(ic.mean()),
        "rank_ic": None if rank_ic.empty else float(rank_ic.mean()),
        "icir": _ir(ic),
        "rank_icir": _ir(rank_ic),
        "daily": daily,
    }


def portfolio_summary(report_df: pd.DataFrame) -> dict:
    ret = report_df["return"].astype(float)
    if "cost" in report_df.columns:
        ret = ret - report_df["cost"].astype(float)
    bench = report_df["bench"].astype(float) if "bench" in report_df.columns else pd.Series(0.0, index=ret.index)
    excess = ret - bench
    equity = (1.0 + ret).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    turnover = None
    if "turnover" in report_df.columns:
        turnover = float(report_df["turnover"].astype(float).mean())
    annualized = None
    if len(ret) > 0:
        annualized = float((1.0 + ret.mean()) ** 252 - 1.0)
    return {
        "interval_return": float(equity.iloc[-1] - 1.0),
        "benchmark_return": float((1.0 + bench).prod() - 1.0),
        "excess_return": float((1.0 + excess).prod() - 1.0),
        "max_drawdown": float(drawdown.min()),
        "turnover": turnover,
        "annualized_return": annualized,
        "days": int(len(ret)),
    }


def run_model(
    name: str,
    spec: dict,
    cfg: dict,
    resolved: dict,
    handler_span: dict,
    calendar: pd.DatetimeIndex,
    out_dir: Path,
) -> dict:
    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.utils import init_instance_by_config

    label = cfg["label"]
    universe = cfg["universe"]
    backtest = cfg["backtest"]
    features = cfg["features"]
    rule = universe_expression(universe)
    cost = side_cost(backtest["costs"])
    handler = {
        "class": features["class"],
        "module_path": features["module_path"],
        "kwargs": {
            "instruments": "tradable",
            "start_time": handler_span["start"],
            "end_time": handler_span["end"],
            "fit_start_time": resolved["train"][0],
            "fit_end_time": resolved["train"][1],
            "freq": "day",
            "label": ([label["expr"]], [label["column"]]),
            "filter_pipe": [
                {
                    "filter_type": "ExpressionDFilter",
                    "rule_expression": rule,
                    "filter_start_time": handler_span["start"],
                    "filter_end_time": handler_span["end"],
                    "keep": False,
                }
            ],
        },
    }
    dataset_cfg = {
        "class": features["dataset_class"],
        "module_path": features["dataset_module_path"],
        "kwargs": {
            "handler": handler,
            "segments": resolved,
        },
    }
    model_cfg = {
        "class": spec["class"],
        "module_path": spec["module_path"],
        "kwargs": spec.get("kwargs") or {},
    }
    print(f"[{name}] 拟合 {spec['class']}", flush=True)
    dataset = init_instance_by_config(dataset_cfg)
    model = init_instance_by_config(model_cfg)
    evals_result: dict = {}
    try:
        model.fit(dataset, evals_result=evals_result)
    except TypeError:
        model.fit(dataset)
        evals_result = {}
    evals_result = {
        str(split): {str(key): [float(value) for value in values] for key, values in metrics.items()}
        for split, metrics in evals_result.items()
    }

    result = {
        "model": name,
        "class": spec["class"],
        "universe_rule": rule,
        "side_cost": cost,
        "segments": {key: list(value) for key, value in resolved.items()},
        "train_curves": evals_result,
        "metrics": {},
    }
    pred_dir = out_dir / name
    pred_dir.mkdir(parents=True, exist_ok=True)
    for segment in cfg["report"]["segments"]:
        pred = score_frame(model.predict(dataset, segment=segment))
        label_df = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_I)
        label_s = score_frame(label_df)
        summary = ic_summary(pred, label_s)
        daily = summary.pop("daily")
        daily.to_csv(pred_dir / f"{segment}_ic.csv")
        pred.to_frame("score").to_pickle(pred_dir / f"{segment}_pred.pkl")
        start, end = resolved[segment]
        # 回测最后一步会读下一个交易日。停在日历倒数第二天，避免越界。
        if pd.Timestamp(end) >= calendar[-1]:
            end = calendar[-2].strftime("%Y-%m-%d")
        print(f"[{name}] 回测 {segment} {start} ~ {end}", flush=True)
        strategy_cls = TopkDropoutStrategy
        saved_exchange = None
        if backtest.get("respect_lot_size", False):
            import qlib.backtest as qbacktest

            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from exchange import HKLotExchange, LotTopkDropoutStrategy
            from lots import load_lot_sizes

            codes = pred.index.get_level_values("instrument").unique().tolist()
            lot_sizes, fallback = load_lot_sizes(
                codes,
                cfg["data"]["freeze_dirname"],
                default_lot=int(backtest.get("default_lot_size", 100)),
            )
            print(
                f"[{name}] 回测每手股数 {len(lot_sizes)} 只，"
                f"其中 {fallback} 只按 {int(backtest.get('default_lot_size', 100))} 股",
                flush=True,
            )
            HKLotExchange.lot_sizes = lot_sizes
            saved_exchange = qbacktest.Exchange
            qbacktest.Exchange = HKLotExchange
            strategy_cls = LotTopkDropoutStrategy
        try:
            report_df, _positions = backtest_daily(
                start_time=start,
                end_time=end,
                strategy=strategy_cls(
                    signal=pred,
                    topk=int(backtest["topk"]),
                    n_drop=int(backtest["n_drop"]),
                ),
                account=float(backtest["account"]),
                benchmark=str(backtest["benchmark"]),
                exchange_kwargs={
                    "limit_threshold": backtest.get("limit_threshold"),
                    "deal_price": backtest["deal_price"],
                    "open_cost": cost,
                    "close_cost": cost,
                    "min_cost": float(backtest["costs"]["min_commission"]),
                },
            )
        finally:
            if saved_exchange is not None:
                import qlib.backtest as qbacktest

                qbacktest.Exchange = saved_exchange
        report_df.to_csv(pred_dir / f"{segment}_backtest.csv")
        portfolio = portfolio_summary(report_df)
        result["metrics"][segment] = {**summary, **portfolio}
        print(
            f"[{name}] {segment} rank_ic={summary['rank_ic']} "
            f"excess={portfolio['excess_return']}",
            flush=True,
        )
    (pred_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="港股 Alpha158 基线")
    parser.add_argument("--config", type=Path, default=Path("qlib-zh/hk/baseline.yaml"))
    parser.add_argument("--model", default="lightgbm", help="模型名，或 all")
    args = parser.parse_args()

    import qlib

    cfg = load_config(args.config)
    data_dir = provider_dir(cfg)
    calendar = load_calendar(data_dir / "calendars" / "day.txt")
    resolved = segments_for(cfg, calendar)
    handler_span = {
        "start": str(cfg["segments"]["warmup"]["start"])[:10],
        "end": calendar[-1].strftime("%Y-%m-%d"),
    }
    repo_root = Path(__file__).resolve().parents[2]
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = repo_root / "qlib-zh" / "hk" / "output" / cfg["data"]["freeze_dirname"] / run_stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, out_dir / "baseline.yaml")

    print(f"数据 {data_dir}", flush=True)
    print(f"切分 {resolved}", flush=True)
    qlib.init(provider_uri=str(data_dir), region="us", kernels=4)
    reports = []
    for name, spec in selected_models(cfg, args.model):
        reports.append(run_model(name, spec, cfg, resolved, handler_span, calendar, out_dir))
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provider_uri": str(data_dir),
        "partial_download": True,
        "models": reports,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from report_html import write_report

    report_path = out_dir / "report.html"
    write_report(
        report_path,
        summary,
        {item["model"]: out_dir / item["model"] for item in summary["models"]},
    )
    print(f"报告 {report_path}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
