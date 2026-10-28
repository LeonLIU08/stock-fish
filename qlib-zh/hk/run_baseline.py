"""用 baseline.yaml 在已落盘的港股 qlib 数据上跑表格模型基线。

在镜像 zhuhai123/qlib-rdagent:v1 里执行。标签、切分、费用和模型参数都读配置，
不在这里另写一套数字。默认只跑 lightgbm，把整条训练、验证、测试和回测跑通。

    python3 qlib-zh/hk/run_baseline.py --model lightgbm
    python3 qlib-zh/hk/run_baseline.py --model all
"""

from __future__ import annotations

import argparse
import json
import os
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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def runtime_path(cfg: dict, key: str, default: Path) -> Path:
    raw = (cfg.get("runtime") or {}).get(key)
    if not raw:
        return default
    path = Path(str(raw))
    return path if path.is_absolute() else repo_root() / path


def cache_dir(freeze_dirname: str, cfg: dict | None = None) -> Path:
    """训练复用的缓存。放在工程内，不写到用户主目录或容器 /root。"""
    root = runtime_path(cfg or {}, "cache_dir", repo_root() / "qlib-zh" / "hk" / "cache")
    return root / freeze_dirname


def run_output_dir(cfg: dict, config_path: Path, *parts: str) -> Path:
    root = runtime_path(cfg, "output_dir", config_path.resolve().parent / "output")
    return root.joinpath(cfg["data"]["freeze_dirname"], *parts)


def provider_dir(cfg: dict) -> Path:
    """容器里的 /work 对应仓库根目录。"""
    uri = str(cfg["runtime"]["provider_uri"])
    if uri.startswith("/work/"):
        return repo_root() / uri[len("/work/") :]
    path = Path(uri)
    if not path.is_absolute():
        return repo_root() / path
    if path.exists():
        return path
    return path


def require_calendar(cfg: dict) -> tuple[Path, pd.DatetimeIndex]:
    data_dir = provider_dir(cfg)
    calendar_path = data_dir / "calendars" / "day.txt"
    if calendar_path.exists():
        return data_dir, load_calendar(calendar_path)
    raise FileNotFoundError(
        f"找不到行情日历 {calendar_path}。配置里的 provider_uri 是 {cfg['runtime']['provider_uri']}"
    )


def init_qlib(
    data_dir: Path,
    run_dir: Path,
    kernels: int = 1,
    experiment: str = "hk",
    region: str = "us",
) -> None:
    """MLflow 和 qlib 磁盘缓存都放进本次输出目录。不启用表达式缓存。"""
    import qlib

    run_dir.mkdir(parents=True, exist_ok=True)
    mlruns = run_dir / "mlruns"
    mlruns.mkdir(parents=True, exist_ok=True)
    qlib.init(
        provider_uri=str(data_dir),
        region=region,
        kernels=kernels,
        expression_cache=None,
        dataset_cache=None,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": mlruns.resolve().as_uri(),
                "default_exp_name": experiment,
            },
        },
    )


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def attach_log(path: Path) -> None:
    """把标准输出和 qlib 日志追加到工程内的 run.log。"""
    import logging

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logging.getLogger("qlib").addHandler(file_handler)


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


def _fee_base(costs: dict) -> float:
    return (
        float(costs["commission_rate"])
        + float(costs["exchange_fee_rate"])
        + float(costs["settlement_rate"])
        + float(costs["slippage_rate"])
    )


def trade_costs(costs: dict) -> tuple[float, float]:
    """买入成本和卖出成本。印花税默认买卖都收；A 股只在卖出时收。"""
    base = _fee_base(costs)
    stamp = float(costs["stamp_duty_rate"])
    if costs.get("stamp_both_sides", True):
        return base + stamp, base + stamp
    return base, base + stamp


def side_cost(costs: dict) -> float:
    buy, _sell = trade_costs(costs)
    return buy


def universe_symbols(cfg: dict) -> set[str] | None:
    """名单是股票范围的硬边界。没有配置名单时不额外限制。"""
    universe = cfg.get("universe") or {}
    raw = universe.get("symbols_file")
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = repo_root() / path
    if not path.exists():
        raise FileNotFoundError(f"股票名单不存在: {path}")
    width = int(cfg["data"]["code_width"])
    excluded = set()
    for code in universe.get("exclude") or []:
        digits = "".join(ch for ch in str(code) if ch.isdigit())
        if digits:
            excluded.add(digits.zfill(width))
    found: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        token = text.split()[0]
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits or len(digits) > width:
            raise ValueError(f"无法解析股票代码: {token}")
        code = digits.zfill(width)
        if code not in excluded:
            found.add(code)
    if not found:
        raise ValueError(f"股票名单是空的: {path}")
    return found


def qlib_instruments(cfg: dict) -> str | list[str]:
    """交给 qlib 的代码。A股数据目录用 SH/SZ 前缀，名单里是六位数字。"""
    allowed = universe_symbols(cfg)
    if allowed is None:
        return "tradable"
    if cfg.get("backtest", {}).get("lot_rule") != "cn":
        return sorted(allowed)

    def prefixed(code: str) -> str:
        if code.startswith(("SH", "SZ", "BJ")):
            return code
        if code.startswith(("5", "6", "9")):
            return f"SH{code}"
        if code.startswith(("4", "8")):
            return f"BJ{code}"
        return f"SZ{code}"

    return sorted(prefixed(code) for code in allowed)


def _instrument_codes(index: pd.Index, width: int) -> pd.Index:
    names = index.get_level_values("instrument").astype(str)
    return names.str.replace(r"\D", "", regex=True).str.zfill(width)


def limit_instruments(data: pd.DataFrame | pd.Series, cfg: dict):
    """丢掉名单以外的行。截面标准化和选股都发生在这之后。"""
    allowed = universe_symbols(cfg)
    if not allowed or len(data) == 0:
        return data
    codes = _instrument_codes(data.index, int(cfg["data"]["code_width"]))
    mask = np.asarray(codes.isin(allowed), dtype=bool)
    limited = data.loc[mask]
    if len(limited) == 0:
        raise RuntimeError(f"股票名单和当前数据没有交集: {cfg['universe'].get('symbols_file')}")
    return limited


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
    wanted = {item.strip() for item in str(choice).split(",") if item.strip()}
    models = []
    for name, spec in cfg["models"].items():
        if spec.get("dataset_kind", "tabular") != "tabular":
            continue
        if choice != "all" and name not in wanted:
            continue
        if choice == "all" and not spec.get("enabled", False):
            continue
        if choice != "all" and name in wanted and not spec.get("enabled", False):
            raise ValueError(f"模型 {name} 在配置里是 enabled: false")
        models.append((name, spec))
    if choice != "all":
        missing = wanted - {name for name, _spec in models}
        if missing:
            raise ValueError(f"没有可跑的模型: {', '.join(sorted(missing))}")
    if not models:
        raise ValueError(f"没有可跑的模型: {choice}")
    return models


def model_config_and_fit_options(model_cfg: dict) -> tuple[dict, dict]:
    """verbose_eval 只控制终端打印，不能放进 LightGBM 的参数。

    XGBModel.fit 用 num_boost_round 控制轮数。n_estimators 放进 xgb.train 的 params 会被忽略。
    """
    kwargs = dict(model_cfg.get("kwargs") or {})
    fit_options = {}
    if "verbose_eval" in kwargs:
        fit_options["verbose_eval"] = int(kwargs.pop("verbose_eval"))
    if model_cfg.get("class") == "XGBModel" and "n_estimators" in kwargs:
        fit_options["num_boost_round"] = int(kwargs.pop("n_estimators"))
    cleaned = dict(model_cfg)
    cleaned["kwargs"] = kwargs
    return cleaned, fit_options


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


def _plain_curves(evals_result: dict, spec: dict) -> dict:
    """LightGBM 留下 {split: {metric: [...]}}。XGBoost 在拟合后把每个 split 收成一条列表。"""
    metric_name = str((spec.get("kwargs") or {}).get("eval_metric") or "loss")
    curves = {}
    for split, metrics in evals_result.items():
        if isinstance(metrics, dict):
            curves[str(split)] = {
                str(key): [float(value) for value in values] for key, values in metrics.items()
            }
        else:
            curves[str(split)] = {metric_name: [float(value) for value in metrics]}
    return curves


def evaluate_segment(
    name: str,
    pred: pd.Series,
    label_s: pd.Series,
    segment: str,
    start: str,
    end: str,
    calendar: pd.DatetimeIndex,
    cfg: dict,
    pred_dir: Path,
    gate_df: pd.DataFrame | None = None,
) -> dict:
    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.strategy import TopkDropoutStrategy

    backtest = cfg["backtest"]
    buy_cost, sell_cost = trade_costs(backtest["costs"])
    pred = limit_instruments(pred, cfg)
    label_s = limit_instruments(label_s, cfg)
    summary = ic_summary(pred, label_s)
    daily = summary.pop("daily")
    pred_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(pred_dir / f"{segment}_ic.csv")
    pred.to_frame("score").to_pickle(pred_dir / f"{segment}_pred.pkl")
    if pd.Timestamp(end) >= calendar[-1]:
        end = calendar[-2].strftime("%Y-%m-%d")
    print(f"[{name}] 回测 {segment} {start} ~ {end}", flush=True)
    strategy_cls = TopkDropoutStrategy
    saved_exchange = None
    limit_threshold = backtest.get("limit_threshold")
    trade_unit = None
    if backtest.get("respect_lot_size", False):
        import qlib.backtest as qbacktest

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from exchange import CNLotExchange, HKLotExchange, LotTopkDropoutStrategy

        if backtest.get("lot_rule") == "cn":
            print(f"[{name}] A股下单：主板/创业板 100 股，科创板最少 200 股", flush=True)
            saved_exchange = qbacktest.Exchange
            qbacktest.Exchange = CNLotExchange
            limit_threshold = 0.095
            trade_unit = 1
        else:
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
            HKLotExchange.default_lot = int(backtest.get("default_lot_size", 100))
            saved_exchange = qbacktest.Exchange
            qbacktest.Exchange = HKLotExchange
        strategy_cls = LotTopkDropoutStrategy
    try:
        strategy = strategy_cls(
            signal=pred,
            topk=int(backtest["topk"]),
            n_drop=int(backtest["n_drop"]),
        )
        policy = dict(cfg.get("portfolio_policy") or {})
        use_policy = bool(policy.get("enabled")) or os.environ.get("PORTFOLIO_POLICY_ENABLED") == "1"
        if use_policy:
            practice = Path(__file__).resolve().parents[1] / "scripts" / "practice"
            if str(practice) not in sys.path:
                sys.path.insert(0, str(practice))
            from portfolio_backtest import load_policy, strategy_from_score_frame, write_runtime_fingerprint

            policy_config = load_policy(policy)
            policy_config.enabled = True
            signal = pred.to_frame("score")
            if gate_df is not None and not gate_df.empty:
                signal = signal.join(gate_df, how="left")
            strategy = strategy_from_score_frame(signal, policy_config, calendar)
            write_runtime_fingerprint(cfg, pred_dir)
            print(
                f"[{name}] 使用可变持仓策略，最大槽位 {policy_config.max_holdings}，"
                f"score_gate={policy_config.score_gate.mode}",
                flush=True,
            )
        report_df, _positions = backtest_daily(
            start_time=start,
            end_time=end,
            strategy=strategy,
            account=float(backtest["account"]),
            benchmark=str(backtest["benchmark"]),
            exchange_kwargs={
                "limit_threshold": limit_threshold,
                "deal_price": backtest["deal_price"],
                "open_cost": buy_cost,
                "close_cost": sell_cost,
                "min_cost": float(backtest["costs"]["min_commission"]),
                **({} if trade_unit is None else {"trade_unit": trade_unit}),
            },
        )
    finally:
        if saved_exchange is not None:
            import qlib.backtest as qbacktest

            qbacktest.Exchange = saved_exchange
    report_df.to_csv(pred_dir / f"{segment}_backtest.csv")
    if (cfg.get("portfolio_policy") or {}).get("enabled") or os.environ.get("PORTFOLIO_POLICY_ENABLED") == "1":
        import pickle

        with open(pred_dir / f"{segment}_positions.pkl", "wb") as handle:
            pickle.dump(_positions, handle)
    portfolio = portfolio_summary(report_df)
    print(
        f"[{name}] {segment} rank_ic={summary['rank_ic']} excess={portfolio['excess_return']}",
        flush=True,
    )
    return {**summary, **portfolio}


def run_model(
    name: str,
    spec: dict,
    cfg: dict,
    resolved: dict,
    handler_span: dict,
    calendar: pd.DatetimeIndex,
    out_dir: Path,
    config_path: Path | None = None,
) -> dict:
    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.utils import init_instance_by_config

    label = cfg["label"]
    from score_gate_training import policy_requests_calibrator

    if policy_requests_calibrator(cfg):
        if config_path is None:
            raise RuntimeError("打开校准门槛时需要配置文件路径，以便按月缓存样本外预测")
        from run_rolling import run_rolling

        print(f"[{name}] 校准门槛：按月样本外预测，每个预测日冻结校准器后再重训并只预测该折", flush=True)
        return run_rolling(name, spec, cfg, calendar, out_dir, False, config_path)
    universe = cfg["universe"]
    backtest = cfg["backtest"]
    features = cfg["features"]
    rule = universe_expression(universe)
    buy_cost, sell_cost = trade_costs(backtest["costs"])
    instruments = qlib_instruments(cfg)
    handler = {
        "class": features["class"],
        "module_path": features["module_path"],
        "kwargs": {
            "instruments": instruments,
            "start_time": handler_span["start"],
            "end_time": handler_span["end"],
            "fit_start_time": resolved["train"][0],
            "fit_end_time": resolved["train"][1],
            "freq": "day",
            "label": ([label["expr"]], [label["column"]]),
            # 先丢掉没有未来收益的标签，再做截面标准化。
            # 标准化在当天只有一只股票或收益完全相同时会产生 NaN / inf，最后再滤掉。
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
                {"class": "DropInvalidLabel", "module_path": "processors"},
            ],
            "infer_processors": [
                {"class": "SanitizeFeature", "module_path": "processors"},
            ],
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
    model_cfg, fit_options = model_config_and_fit_options(
        {
            "class": spec["class"],
            "module_path": spec["module_path"],
            "kwargs": spec.get("kwargs") or {},
        }
    )
    print(f"[{name}] 拟合 {spec['class']}", flush=True)
    dataset = init_instance_by_config(dataset_cfg)
    model = init_instance_by_config(model_cfg)
    evals_result: dict = {}
    try:
        model.fit(dataset, evals_result=evals_result, **fit_options)
    except TypeError:
        model.fit(dataset)
        evals_result = {}
    evals_result = _plain_curves(evals_result, spec)

    result = {
        "model": name,
        "class": spec["class"],
        "universe_rule": rule,
        "open_cost": buy_cost,
        "close_cost": sell_cost,
        "segments": {key: list(value) for key, value in resolved.items()},
        "train_curves": evals_result,
        "metrics": {},
        "parameters": {
            "version": cfg.get("version"),
            "class": spec["class"],
            "module_path": spec["module_path"],
            "kwargs": {
                key: value
                for key, value in (spec.get("kwargs") or {}).items()
                if key != "verbose_eval"
            },
        },
    }
    pred_dir = out_dir / name
    pred_dir.mkdir(parents=True, exist_ok=True)
    for segment in cfg["report"]["segments"]:
        pred = score_frame(model.predict(dataset, segment=segment))
        label_df = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_I)
        label_s = score_frame(label_df)
        start, end = resolved[segment]
        result["metrics"][segment] = evaluate_segment(
            name, pred, label_s, segment, start, end, calendar, cfg, pred_dir
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
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    cfg = load_config(args.config)
    data_dir, calendar = require_calendar(cfg)
    resolved = segments_for(cfg, calendar)
    handler_span = {
        "start": str(cfg["segments"]["warmup"]["start"])[:10],
        "end": calendar[-1].strftime("%Y-%m-%d"),
    }
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = run_output_dir(cfg, args.config, run_stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, out_dir / "baseline.yaml")

    attach_log(out_dir / "run.log")
    print(f"数据 {data_dir}", flush=True)
    print(f"切分 {resolved}", flush=True)
    init_qlib(
        data_dir,
        out_dir,
        kernels=4,
        experiment="baseline",
        region=str((cfg.get("runtime") or {}).get("region") or "us"),
    )
    reports = []
    for name, spec in selected_models(cfg, args.model):
        reports.append(run_model(name, spec, cfg, resolved, handler_span, calendar, out_dir, args.config))
    symbols = universe_symbols(cfg)
    universe = cfg.get("universe") or {}
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provider_uri": str(data_dir),
        "partial_download": True,
        "symbols_file": None if symbols is None else str(universe.get("symbols_file")),
        "symbol_count": None if symbols is None else len(symbols),
        "universe": {
            "symbols_file": universe.get("symbols_file"),
            "symbol_count": None if symbols is None else len(symbols),
            "min_history_days": universe.get("min_history_days"),
            "median_amount_days": universe.get("median_amount_days"),
            "min_median_amount": universe.get("min_median_amount"),
            "require_positive_volume": bool(universe.get("require_positive_volume")),
            "exclude": universe.get("exclude") or [],
        },
        "models": reports,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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
