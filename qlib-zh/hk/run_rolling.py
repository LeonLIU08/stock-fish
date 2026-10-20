"""用 qlib RollingGen 做 Alpha158 扩展窗口滚动测试。

每 step 个交易日重新训练，只预测接下来的 step 天。
验证段和测试段连成一段滚动；训练起点固定，终点随预测日推进。
标签泄漏由 RollingGen 的 trunc_days 截断。

    python3 qlib-zh/hk/run_rolling.py --model lightgbm
    python3 qlib-zh/hk/run_rolling.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from run_baseline import (
    _plain_curves,
    attach_log,
    evaluate_segment,
    init_qlib,
    load_calendar,
    load_config,
    model_config_and_fit_options,
    provider_dir,
    score_frame,
    selected_models,
    universe_expression,
    universe_symbols,
)


def _stamp(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _extra_mod(task: dict, rolling_gen) -> None:
    from qlib.workflow.task.gen import handler_mod

    handler_mod(task, rolling_gen)
    segments = task["dataset"]["kwargs"]["segments"]
    handler = task["dataset"]["kwargs"]["handler"]["kwargs"]
    handler["fit_end_time"] = segments["train"][1]
    handler["end_time"] = segments["test"][1]


def _handler(cfg: dict, start: str, end: str) -> dict:
    features = cfg["features"]
    label = cfg["label"]
    return {
        "class": features["class"],
        "module_path": features["module_path"],
        "kwargs": {
            "instruments": "tradable",
            "start_time": start,
            "end_time": end,
            "fit_start_time": start,
            "fit_end_time": end,
            "freq": "day",
            "label": ([label["expr"]], [label["column"]]),
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
                    "rule_expression": universe_expression(cfg["universe"]),
                    "filter_start_time": start,
                    "filter_end_time": end,
                    "keep": False,
                }
            ],
        },
    }


def _early_stop_segment(calendar: pd.DatetimeIndex, eval_start: str, days: int) -> tuple[str, str]:
    pos = int(calendar.searchsorted(pd.Timestamp(eval_start), side="left"))
    end_idx = pos - 1
    start_idx = end_idx - int(days) + 1
    if start_idx < 0:
        raise ValueError("早停窗口超出交易日历")
    return _stamp(calendar[start_idx]), _stamp(calendar[end_idx])


def build_tasks(
    cfg: dict,
    spec: dict,
    calendar: pd.DatetimeIndex,
    eval_start: str | None = None,
    eval_end: str | None = None,
) -> list[dict]:
    from qlib.workflow.task.gen import RollingGen, task_generator

    rolling = cfg["rolling"]
    segments = cfg["segments"]
    eval_start = eval_start or str(segments["valid"]["start"])[:10]
    eval_end = eval_end or str(segments["test"]["end"])[:10]
    horizon = int(cfg["label"]["horizon_trading_days"])
    trunc_days = int(rolling.get("trunc_days") or horizon + 1)
    rtype = RollingGen.ROLL_SD if rolling.get("rtype") == "sliding" else RollingGen.ROLL_EX
    task = {
        "model": {
            "class": spec["class"],
            "module_path": spec["module_path"],
            "kwargs": spec.get("kwargs") or {},
        },
        "dataset": {
            "class": cfg["features"]["dataset_class"],
            "module_path": cfg["features"]["dataset_module_path"],
            "kwargs": {
                "handler": _handler(cfg, str(segments["warmup"]["start"])[:10], eval_end),
                "segments": {
                    "train": (str(segments["train"]["start"])[:10], eval_start),
                    "valid": _early_stop_segment(
                        calendar, eval_start, int(rolling["early_stop_days"])
                    ),
                    "test": (eval_start, eval_end),
                },
            },
        },
    }
    return task_generator(
        task,
        RollingGen(
            step=int(rolling["step"]),
            rtype=rtype,
            trunc_days=trunc_days,
            ds_extra_mod_func=_extra_mod,
        ),
    )


def _segment_dates(task: dict) -> dict[str, tuple[str, str]]:
    raw = task["dataset"]["kwargs"]["segments"]
    return {key: (_stamp(value[0]), _stamp(value[1])) for key, value in raw.items()}


def _init_qlib(cfg: dict, data_dir: Path, out_dir: Path) -> None:
    init_qlib(data_dir, out_dir, kernels=1, experiment="rolling")


def _fit_and_predict(task: dict, spec: dict, cfg: dict):
    from qlib.utils import init_instance_by_config

    from panel import PanelDataset, load_panel

    segments = task["dataset"]["kwargs"]["segments"]
    start = min(_stamp(segments[key][0]) for key in ("train", "valid", "test"))
    end = _stamp(segments["test"][1])
    frame = load_panel(cfg, start, end)
    dataset = PanelDataset(frame, segments)
    del frame
    model_cfg, fit_options = model_config_and_fit_options(task["model"])
    model = init_instance_by_config(model_cfg)
    evals_result: dict = {}
    try:
        model.fit(dataset, evals_result=evals_result, **fit_options)
    except TypeError:
        model.fit(dataset)
        evals_result = {}
    pred = score_frame(model.predict(dataset, segment="test"))
    curves = _plain_curves(evals_result, spec)
    del model, dataset
    gc.collect()
    return pred, curves


def _run_fold_worker(task_path: Path, fold_dir: Path, config_path: Path) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--worker-task",
        str(task_path),
        "--worker-out",
        str(fold_dir),
    ]
    subprocess.run(cmd, check=True)


def build_panel(cfg: dict, run_dir: Path) -> None:
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.utils import init_instance_by_config

    from panel import append_panel, extendable_end, panel_dir, panel_ready, write_panel

    if panel_ready(cfg):
        print(f"复用已落盘的 Alpha158: {panel_dir(cfg['data']['freeze_dirname'])}", flush=True)
        return
    cached_end = extendable_end(cfg)
    if cached_end:
        print(f"补齐 Alpha158 {cached_end} 之后至 {cfg['segments']['test']['end']}", flush=True)
    else:
        print("计算 Alpha158 并按月写入磁盘", flush=True)
    data_dir = provider_dir(cfg)
    _init_qlib(cfg, data_dir, run_dir)
    segments = cfg["segments"]
    end = str(segments["test"]["end"])[:10]
    if cached_end:
        # Alpha158 最长回看约 60 个交易日，往前多取半年，只保留缓存结束日之后的行。
        start = (pd.Timestamp(cached_end) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    else:
        start = str(segments["warmup"]["start"])[:10]
    task = {
        "class": cfg["features"]["dataset_class"],
        "module_path": cfg["features"]["dataset_module_path"],
        "kwargs": {
            "handler": _handler(cfg, start, end),
            "segments": {"all": (start, end)},
        },
    }
    dataset = init_instance_by_config(task)
    frame = dataset.prepare("all", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    del dataset
    gc.collect()
    path = append_panel(frame, cfg, cached_end) if cached_end else write_panel(frame, cfg)
    del frame
    gc.collect()
    print(f"Alpha158 已写入 {path}", flush=True)


def ensure_panel(cfg: dict, run_dir: Path, config_path: Path) -> None:
    from panel import panel_ready

    if panel_ready(cfg):
        print(f"复用已落盘的 Alpha158", flush=True)
        return
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(config_path),
            "--build-panel",
            str(run_dir),
        ],
        check=True,
    )


def worker_main(args) -> int:
    cfg = load_config(args.config)
    data_dir = provider_dir(cfg)
    fold_dir = Path(args.worker_out)
    run_dir = fold_dir.parents[2]
    _init_qlib(cfg, data_dir, run_dir)
    task = pickle.loads(Path(args.worker_task).read_bytes())
    spec = {
        "class": task["model"]["class"],
        "module_path": task["model"]["module_path"],
        "kwargs": task["model"].get("kwargs") or {},
    }
    pred, curves = _fit_and_predict(task, spec, cfg)
    fold_dir.mkdir(parents=True, exist_ok=True)
    pred.to_frame("score").to_pickle(fold_dir / "pred.pkl")
    (fold_dir / "train_curves.json").write_text(
        json.dumps(curves, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"已写入 {fold_dir / 'pred.pkl'}", flush=True)
    return 0


def _between(pred: pd.Series, start: str, end: str) -> pd.Series:
    dates = pd.to_datetime(pred.index.get_level_values("datetime"))
    chosen = pred.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    return chosen[~chosen.index.duplicated(keep="last")].sort_index()


def _labels_for(pred: pd.Series, cfg: dict) -> pd.Series:
    from qlib.data import D

    dates = pd.to_datetime(pred.index.get_level_values("datetime"))
    instruments = pred.index.get_level_values("instrument").unique().tolist()
    expr = cfg["label"]["expr"]
    label = D.features(instruments, [expr], start_time=dates.min(), end_time=dates.max())
    label.columns = ["label"]
    series = label["label"]
    series.index = series.index.set_names(["instrument", "datetime"])
    return series.swaplevel().sort_index()


def run_rolling(
    name: str,
    spec: dict,
    cfg: dict,
    calendar: pd.DatetimeIndex,
    out_dir: Path,
    dry_run: bool,
    config_path: Path,
    eval_start: str | None = None,
    eval_end: str | None = None,
) -> dict:
    tasks = build_tasks(cfg, spec, calendar, eval_start, eval_end)
    folds = [_segment_dates(task) for task in tasks]
    print(f"[{name}] RollingGen 任务 {len(tasks)} 个", flush=True)
    for index, fold in enumerate(folds, start=1):
        print(
            f"[{name}] {index}/{len(tasks)} train {fold['train'][1]} "
            f"valid {fold['valid'][0]}~{fold['valid'][1]} "
            f"test {fold['test'][0]}~{fold['test'][1]}",
            flush=True,
        )
    if dry_run:
        return {"model": name, "folds": [{key: list(value) for key, value in fold.items()} for fold in folds]}

    ensure_panel(cfg, out_dir, config_path)
    parts = []
    curves = {}
    end_limit = pd.Timestamp(eval_end or cfg["segments"]["test"]["end"])
    for index, task in enumerate(tasks, start=1):
        fold = folds[index - 1]
        fold_dir = out_dir / name / "folds" / f"{index:02d}"
        pred_path = fold_dir / "pred.pkl"
        if pred_path.exists():
            print(f"[{name}] 跳过已完成 {index}/{len(tasks)} {fold['test'][0]} ~ {fold['test'][1]}", flush=True)
            pred = pd.read_pickle(pred_path)["score"]
            curve_path = fold_dir / "train_curves.json"
            if curve_path.exists():
                curves = json.loads(curve_path.read_text(encoding="utf-8"))
        else:
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / "fold.json").write_text(
                json.dumps({key: list(value) for key, value in fold.items()}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            task_path = fold_dir / "task.pkl"
            task_path.write_bytes(pickle.dumps(task))
            print(f"[{name}] 训练 {index}/{len(tasks)} 预测 {fold['test'][0]} ~ {fold['test'][1]}", flush=True)
            _run_fold_worker(task_path, fold_dir, config_path)
            pred = pd.read_pickle(pred_path)["score"]
            curves = json.loads((fold_dir / "train_curves.json").read_text(encoding="utf-8"))
        dates = pd.to_datetime(pred.index.get_level_values("datetime"))
        parts.append(pred.loc[dates <= end_limit])
    stitched = pd.concat(parts)
    stitched = stitched[~stitched.index.duplicated(keep="last")].sort_index()
    labels = _labels_for(stitched, cfg)
    pred_dir = out_dir / name
    metrics = {}
    segments = cfg["segments"]
    if eval_start and eval_end:
        report_windows = (("test", eval_start, eval_end),)
    else:
        report_windows = tuple(
            (segment, str(segments[segment]["start"])[:10], str(segments[segment]["end"])[:10])
            for segment in ("valid", "test")
        )
    for segment, start, end in report_windows:
        piece = _between(stitched, start, end)
        label_piece = _between(labels, start, end)
        metrics[segment] = evaluate_segment(
            name, piece, label_piece, segment, start, end, calendar, cfg, pred_dir
        )
    result = {
        "model": name,
        "class": spec["class"],
        "rolling": {
            "step": int(cfg["rolling"]["step"]),
            "rtype": cfg["rolling"].get("rtype", "expanding"),
            "folds": len(tasks),
        },
        "segments": {
            segment: [start, end] for segment, start, end in report_windows
        },
        "train_curves": curves,
        "metrics": metrics,
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
    result["rolling"]["early_stop_days"] = int(cfg["rolling"]["early_stop_days"])
    (pred_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (pred_dir / "folds.json").write_text(
        json.dumps(
            [{key: list(value) for key, value in fold.items()} for fold in folds],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _load_score(path: Path) -> pd.Series:
    pred = pd.read_pickle(path)
    if isinstance(pred, pd.DataFrame):
        pred = pred["score"]
    return score_frame(pred)


def rebacktest(
    cfg: dict,
    source: Path,
    out_dir: Path,
    calendar: pd.DatetimeIndex,
    topk: int,
    n_drop: int,
) -> dict:
    """同一份预测分数，只改持有只数和每日替换只数。"""
    cfg["backtest"]["topk"] = int(topk)
    cfg["backtest"]["n_drop"] = int(n_drop)
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    source_model = source_summary["models"][0]
    name = source_model["model"]
    pred_dir = out_dir / name
    metrics = {}
    for segment, bounds in source_model["segments"].items():
        start, end = bounds
        pred = _load_score(source / name / f"{segment}_pred.pkl")
        labels = _labels_for(pred, cfg)
        metrics[segment] = evaluate_segment(
            name, pred, labels, segment, start, end, calendar, cfg, pred_dir
        )
    return {
        "model": name,
        "class": source_model.get("class"),
        "rolling": source_model.get("rolling"),
        "segments": source_model.get("segments"),
        "train_curves": source_model.get("train_curves"),
        "parameters": source_model.get("parameters"),
        "holdings": {
            "topk": int(topk),
            "n_drop": int(n_drop),
            "account": float(cfg["backtest"]["account"]),
        },
        "scores_from": str(source),
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="港股 Alpha158 滚动测试")
    parser.add_argument("--config", type=Path, default=Path("qlib-zh/hk/baseline.yaml"))
    parser.add_argument("--model", default="lightgbm", help="模型名，或 all")
    parser.add_argument("--dry-run", action="store_true", help="只打印 RollingGen 任务，不训练")
    parser.add_argument("--resume", type=Path, default=None, help="接着某个已有的滚动输出目录跑")
    parser.add_argument("--worker-task", type=Path, default=None)
    parser.add_argument("--worker-out", type=Path, default=None)
    parser.add_argument("--build-panel", type=Path, default=None)
    parser.add_argument("--eval-start", default=None, help="滚动预测起点，覆盖配置里的验证段起点")
    parser.add_argument("--eval-end", default=None, help="滚动预测终点，覆盖配置里的测试段终点")
    parser.add_argument("--scores-from", type=Path, default=None, help="复用这份滚动结果的预测分数，只重跑回测")
    parser.add_argument("--topk", type=int, default=None, help="覆盖持有只数")
    parser.add_argument("--n-drop", type=int, default=None, help="覆盖每个交易日替换只数")
    parser.add_argument("--account", type=float, default=None, help="覆盖回测初始资金")
    args = parser.parse_args()
    if args.build_panel:
        build_panel(load_config(args.config), args.build_panel)
        return 0
    if args.worker_task:
        return worker_main(args)

    cfg = load_config(args.config)
    data_dir = provider_dir(cfg)
    calendar = load_calendar(data_dir / "calendars" / "day.txt")
    repo_root = Path(__file__).resolve().parents[2]
    if args.scores_from:
        if args.topk is None or args.n_drop is None:
            raise ValueError("复用预测分数时需要同时给出 --topk 和 --n-drop")
        cfg["backtest"]["topk"] = int(args.topk)
        cfg["backtest"]["n_drop"] = int(args.n_drop)
        if args.account is not None:
            cfg["backtest"]["account"] = float(args.account)
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M")
    if args.resume:
        out_dir = args.resume
    elif args.scores_from:
        account_tag = f"-acct{int(args.account)}" if args.account is not None else ""
        out_dir = (
            repo_root
            / "qlib-zh"
            / "hk"
            / "output"
            / cfg["data"]["freeze_dirname"]
            / f"{run_stamp}-topk{args.topk}-drop{args.n_drop}{account_tag}"
        )
    else:
        suffix = "train-tail" if args.eval_start and args.eval_end else "rolling"
        out_dir = repo_root / "qlib-zh" / "hk" / "output" / cfg["data"]["freeze_dirname"] / f"{run_stamp}-{suffix}"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.scores_from:
            text = args.config.read_text(encoding="utf-8")
            text = text.replace(f"topk: {load_config(args.config)['backtest']['topk']}", f"topk: {args.topk}", 1)
            text = text.replace(f"n_drop: {load_config(args.config)['backtest']['n_drop']}", f"n_drop: {args.n_drop}", 1)
            if args.account is not None:
                text = text.replace(
                    f"account: {load_config(args.config)['backtest']['account']}",
                    f"account: {int(args.account) if float(args.account).is_integer() else args.account}",
                    1,
                )
            (out_dir / "baseline.yaml").write_text(text, encoding="utf-8")
        else:
            shutil.copyfile(args.config, out_dir / "baseline.yaml")
        attach_log(out_dir / "run.log")
    print(f"数据 {data_dir}", flush=True)
    print(f"输出 {out_dir}", flush=True)
    symbols = universe_symbols(cfg)
    if symbols is not None:
        print(
            f"股票范围 {cfg['universe']['symbols_file']}，{len(symbols)} 只。"
            "名单外不进入训练、标签标准化和选股。",
            flush=True,
        )
    _init_qlib(cfg, data_dir, out_dir)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if args.scores_from:
        print(
            f"复用预测 {args.scores_from}，持有 {args.topk} 只，每日替换 {args.n_drop} 只，资金 {cfg['backtest']['account']:.0f}",
            flush=True,
        )
        reports = [
            rebacktest(cfg, args.scores_from, out_dir, calendar, args.topk, args.n_drop)
        ]
    else:
        reports = []
        for name, spec in selected_models(cfg, args.model):
            kwargs = spec.get("kwargs") or {}
            print(
                f"[{name}] 配置 {cfg.get('version')} "
                f"learning_rate={kwargs.get('learning_rate')} "
                f"num_boost_round={kwargs.get('num_boost_round')}",
                flush=True,
            )
            reports.append(
                run_rolling(
                    name,
                    spec,
                    cfg,
                    calendar,
                    out_dir,
                    args.dry_run,
                    args.config,
                    args.eval_start,
                    args.eval_end,
                )
            )
    if args.dry_run:
        print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)
        return 0
    universe = cfg.get("universe") or {}
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provider_uri": str(data_dir),
        "rolling": True,
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
    write_report(report_path, summary, {item["model"]: out_dir / item["model"] for item in reports})
    print(f"报告 {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
