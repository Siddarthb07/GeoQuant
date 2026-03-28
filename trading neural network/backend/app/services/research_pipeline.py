from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from app.core.config import settings
from app.services.backtest_engine import BacktestConfig
from app.services.feature_engineering import INTRADAY_FEATURE_COLUMNS
from app.services.neural_model import train_binary_classifier
from app.services.research_data import ResearchDataset, load_research_dataset
from app.services.visualization_service import create_visualizations
from app.services.walk_forward_validation import WalkForwardConfig, run_walk_forward_validation


@dataclass
class ResearchRunConfig:
    data_csv_path: Optional[str] = None
    symbols: List[str] = field(default_factory=list)
    interval: str = "5m"
    train_start: str = "2000-01-01"
    test_start: str = "2020-01-01"
    test_end: str = "2025-12-31"
    step_months: int = 3
    buy_threshold: float = 0.55
    sell_threshold: float = 0.45
    initial_capital: float = 100000.0
    position_fraction: float = 0.10
    brokerage_fee_bps: float = 10.0
    slippage_bps: float = 5.0
    random_seed: int = 42
    output_dir: str = "artifacts"
    persist_model: bool = True
    epochs: int = 35
    batch_size: int = 512
    learning_rate: float = 1e-3
    min_train_rows: int = 800
    min_test_rows: int = 100
    periods_per_year: int = 19656


def _resolve_validation_dates(
    feature_frame: pd.DataFrame,
    train_start: str,
    test_start: str,
    test_end: str,
) -> tuple[str, str, str, List[str]]:
    warnings: List[str] = []
    if feature_frame.empty:
        return train_start, test_start, test_end, warnings

    ts_min = pd.to_datetime(feature_frame["timestamp"], utc=True, errors="coerce").min()
    ts_max = pd.to_datetime(feature_frame["timestamp"], utc=True, errors="coerce").max()
    if pd.isna(ts_min) or pd.isna(ts_max):
        return train_start, test_start, test_end, warnings

    req_train = pd.Timestamp(train_start, tz="UTC")
    req_test_start = pd.Timestamp(test_start, tz="UTC")
    req_test_end = pd.Timestamp(test_end, tz="UTC")

    effective_train = max(req_train, ts_min)
    effective_test_start = req_test_start
    effective_test_end = min(req_test_end, ts_max)

    in_requested_test = feature_frame[
        (feature_frame["timestamp"] >= req_test_start) & (feature_frame["timestamp"] <= req_test_end)
    ]
    if in_requested_test.empty:
        total = len(feature_frame)
        split_idx = int(total * 0.7)
        split_idx = min(max(split_idx, 1), total - 1)
        effective_train = pd.to_datetime(feature_frame["timestamp"].iloc[0], utc=True)
        effective_test_start = pd.to_datetime(feature_frame["timestamp"].iloc[split_idx], utc=True)
        effective_test_end = pd.to_datetime(feature_frame["timestamp"].iloc[-1], utc=True)
        warnings.append(
            (
                "Requested test window has no rows in available intraday feed. "
                f"Auto-adjusted walk-forward window to available coverage "
                f"{effective_train.date()} -> {effective_test_end.date()}."
            )
        )
    elif effective_test_end <= effective_test_start:
        effective_test_end = ts_max
        warnings.append(
            "Adjusted test_end to latest available timestamp because requested test_end precedes coverage."
        )

    if effective_train >= effective_test_start:
        effective_train = ts_min
        warnings.append(
            "Adjusted train_start to earliest available timestamp because training window was empty."
        )

    return (
        effective_train.strftime("%Y-%m-%d"),
        effective_test_start.strftime("%Y-%m-%d"),
        effective_test_end.strftime("%Y-%m-%d"),
        warnings,
    )


def _read_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a top-level mapping/dictionary.")
    return data


def load_pipeline_config(config_path: str, overrides: Optional[Dict] = None) -> ResearchRunConfig:
    raw = _read_yaml(Path(config_path))
    pipeline = raw.get("pipeline", {}) if isinstance(raw.get("pipeline", {}), dict) else {}
    costs = raw.get("costs", {}) if isinstance(raw.get("costs", {}), dict) else {}
    model = raw.get("model", {}) if isinstance(raw.get("model", {}), dict) else {}
    data_cfg = raw.get("data", {}) if isinstance(raw.get("data", {}), dict) else {}
    validation = raw.get("validation", {}) if isinstance(raw.get("validation", {}), dict) else {}
    outputs = raw.get("outputs", {}) if isinstance(raw.get("outputs", {}), dict) else {}

    symbols = data_cfg.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        symbols = settings.default_us_universe + settings.default_india_universe

    cfg = ResearchRunConfig(
        data_csv_path=data_cfg.get("csv_path"),
        symbols=[str(x) for x in symbols],
        interval=str(data_cfg.get("interval", "5m")),
        train_start=str(validation.get("train_start", "2000-01-01")),
        test_start=str(validation.get("test_start", "2020-01-01")),
        test_end=str(validation.get("test_end", "2025-12-31")),
        step_months=int(validation.get("step_months", 3)),
        buy_threshold=float(model.get("buy_threshold", 0.55)),
        sell_threshold=float(model.get("sell_threshold", 0.45)),
        initial_capital=float(costs.get("initial_capital", 100000.0)),
        position_fraction=float(costs.get("position_fraction", 0.10)),
        brokerage_fee_bps=float(costs.get("brokerage_fee_bps", 10.0)),
        slippage_bps=float(costs.get("slippage_bps", 5.0)),
        random_seed=int(pipeline.get("random_seed", 42)),
        output_dir=str(outputs.get("output_dir", "artifacts")),
        persist_model=bool(outputs.get("persist_model", True)),
        epochs=int(model.get("epochs", 35)),
        batch_size=int(model.get("batch_size", 512)),
        learning_rate=float(model.get("learning_rate", 1e-3)),
        min_train_rows=int(validation.get("min_train_rows", 800)),
        min_test_rows=int(validation.get("min_test_rows", 100)),
        periods_per_year=int(validation.get("periods_per_year", 19656)),
    )

    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    return cfg


def _persist_intraday_model(
    dataset: ResearchDataset,
    run_cfg: ResearchRunConfig,
    feature_columns: List[str],
    effective_train_start: str,
    effective_test_start: str,
) -> Dict:
    train_mask = (
        (dataset.features["timestamp"] >= pd.Timestamp(effective_train_start, tz="UTC"))
        & (dataset.features["timestamp"] < pd.Timestamp(effective_test_start, tz="UTC"))
    )
    train_frame = dataset.features.loc[train_mask].copy()
    if len(train_frame) < max(500, run_cfg.min_train_rows):
        return {"saved": False, "reason": "not_enough_train_rows"}

    split_idx = int(len(train_frame) * 0.85)
    split_idx = min(max(split_idx, 1), len(train_frame) - 1)
    head = train_frame.iloc[:split_idx]
    tail = train_frame.iloc[split_idx:]

    x_train = head[feature_columns].astype(float).to_numpy()
    y_train = head["target_up"].astype(int).to_numpy()
    x_val = tail[feature_columns].astype(float).to_numpy()
    y_val = tail["target_up"].astype(int).to_numpy()

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    train_result = train_binary_classifier(
        x_train=x_train_scaled,
        y_train=y_train,
        x_val=x_val_scaled,
        y_val=y_val,
        epochs=run_cfg.epochs,
        batch_size=run_cfg.batch_size,
        lr=run_cfg.learning_rate,
    )

    model_path = settings.model_dir / "intraday_model.pkl"
    scaler_path = settings.model_dir / "intraday_scaler.pkl"
    meta_path = settings.model_dir / "intraday_meta.json"

    joblib.dump(train_result.model, model_path)
    joblib.dump(scaler, scaler_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": feature_columns,
                "symbol_to_code": dataset.symbol_to_code,
                "input_dim": len(feature_columns),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "metrics": {
                    "accuracy": round(float(train_result.accuracy), 6),
                    "auc": round(float(train_result.auc), 6),
                    "train_rows": int(len(x_train)),
                    "val_rows": int(len(x_val)),
                    "source": "reproducible_research_pipeline",
                },
            },
            f,
            indent=2,
        )

    return {
        "saved": True,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "meta_path": str(meta_path),
    }


def _serialize_report_path(run_dir: Path, latest_dir: Path, filename: str) -> Dict:
    return {
        "run_path": str(run_dir / filename),
        "latest_path": str(latest_dir / filename),
    }


def _write_csv_dual(frame: pd.DataFrame, run_dir: Path, latest_dir: Path, name: str) -> Dict:
    run_path = run_dir / name
    latest_path = latest_dir / name
    frame.to_csv(run_path, index=False)
    frame.to_csv(latest_path, index=False)
    return {"run_path": str(run_path), "latest_path": str(latest_path)}


def run_research_pipeline(config: ResearchRunConfig) -> Dict:
    np.random.seed(int(config.random_seed))
    dataset = load_research_dataset(
        csv_path=config.data_csv_path,
        symbols=config.symbols,
        interval=config.interval,
        start_date=config.train_start,
        end_date=config.test_end,
    )

    effective_train_start, effective_test_start, effective_test_end, date_warnings = _resolve_validation_dates(
        feature_frame=dataset.features,
        train_start=config.train_start,
        test_start=config.test_start,
        test_end=config.test_end,
    )

    walk_cfg = WalkForwardConfig(
        train_start=effective_train_start,
        test_start=effective_test_start,
        test_end=effective_test_end,
        step_months=config.step_months,
        min_train_rows=config.min_train_rows,
        min_test_rows=config.min_test_rows,
        buy_threshold=config.buy_threshold,
        sell_threshold=config.sell_threshold,
        periods_per_year=config.periods_per_year,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
    )
    bt_cfg = BacktestConfig(
        initial_capital=config.initial_capital,
        position_fraction=config.position_fraction,
        brokerage_fee_bps=config.brokerage_fee_bps,
        slippage_bps=config.slippage_bps,
        allow_short=True,
        timestamp_col="timestamp",
        symbol_col="symbol",
        price_col="close",
        signal_col="signal",
    )

    feature_columns = list(INTRADAY_FEATURE_COLUMNS)
    result = run_walk_forward_validation(
        feature_frame=dataset.features,
        feature_columns=feature_columns,
        walk_cfg=walk_cfg,
        backtest_cfg=bt_cfg,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_output = Path(config.output_dir)
    run_dir = base_output / timestamp
    latest_dir = base_output / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    predictions = result["predictions"]
    backtest = result["backtest"]
    equity_curve = backtest["equity_curve"]
    trade_log = backtest["trade_log"]
    benchmark_curve = backtest["benchmark_curve"]

    files = {
        "predictions_csv": _write_csv_dual(predictions, run_dir, latest_dir, "predictions.csv"),
        "trade_log_csv": _write_csv_dual(trade_log, run_dir, latest_dir, "trade_log.csv"),
        "equity_curve_csv": _write_csv_dual(equity_curve, run_dir, latest_dir, "equity_curve.csv"),
        "benchmark_curve_csv": _write_csv_dual(benchmark_curve, run_dir, latest_dir, "benchmark_curve.csv"),
    }

    plots_run = create_visualizations(
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        prediction_prices=predictions[["timestamp", "symbol", "close"]].copy(),
        trade_log=trade_log,
        output_dir=run_dir / "plots",
    )
    plots_latest = create_visualizations(
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        prediction_prices=predictions[["timestamp", "symbol", "close"]].copy(),
        trade_log=trade_log,
        output_dir=latest_dir / "plots",
    )

    model_artifacts = {"saved": False}
    if config.persist_model:
        model_artifacts = _persist_intraday_model(
            dataset=dataset,
            run_cfg=config,
            feature_columns=feature_columns,
            effective_train_start=effective_train_start,
            effective_test_start=effective_test_start,
        )

    report = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "train_start": config.train_start,
            "test_start": config.test_start,
            "test_end": config.test_end,
            "effective_train_start": effective_train_start,
            "effective_test_start": effective_test_start,
            "effective_test_end": effective_test_end,
            "interval": config.interval,
            "step_months": config.step_months,
            "buy_threshold": config.buy_threshold,
            "sell_threshold": config.sell_threshold,
            "initial_capital": config.initial_capital,
            "position_fraction": config.position_fraction,
            "brokerage_fee_bps": config.brokerage_fee_bps,
            "slippage_bps": config.slippage_bps,
            "symbols": config.symbols,
            "data_csv_path": config.data_csv_path,
        },
        "data_warnings": dataset.warnings + date_warnings,
        "classification_metrics": result["classification_metrics"],
        "directional_trade_metrics": result["directional_trade_metrics"],
        "performance_metrics": result["performance"],
        "benchmark_comparison": result["benchmark"],
        "walk_forward_folds": result["fold_metrics"],
        "artifacts": {
            "run_dir": str(run_dir),
            "latest_dir": str(latest_dir),
            "files": files,
            "plots": {
                "run": plots_run,
                "latest": plots_latest,
            },
            "model": model_artifacts,
        },
    }

    run_report = run_dir / "report.json"
    latest_report = latest_dir / "report.json"
    with run_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with latest_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report["artifacts"]["report"] = _serialize_report_path(run_dir, latest_dir, "report.json")
    return report


def load_latest_report(output_dir: str = "artifacts") -> Dict:
    latest = Path(output_dir) / "latest" / "report.json"
    if not latest.exists():
        raise FileNotFoundError(f"Latest report not found at: {latest}")
    with latest.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Latest report JSON is invalid.")
    return payload
