from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

from app.services.research_pipeline import load_pipeline_config, run_research_pipeline


def _optional_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GeoQuant reproducible research runner. "
            "Executes walk-forward training, backtest, metrics, benchmark comparison, and artifacts."
        )
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--csv-path", default=None, help="Optional market data CSV path.")
    parser.add_argument("--initial-capital", type=float, default=None, help="Initial capital for backtest.")
    parser.add_argument("--position-fraction", type=float, default=None, help="Position sizing fraction (0-1).")
    parser.add_argument("--brokerage-fee-bps", type=float, default=None, help="Brokerage fee in bps per side.")
    parser.add_argument("--slippage-bps", type=float, default=None, help="Slippage in bps per side.")
    parser.add_argument("--buy-threshold", type=float, default=None, help="Probability threshold for long signal.")
    parser.add_argument("--sell-threshold", type=float, default=None, help="Probability threshold for short signal.")
    parser.add_argument("--step-months", type=int, default=None, help="Walk-forward step in months.")
    parser.add_argument("--epochs", type=int, default=None, help="Model training epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=None, help="Model training batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Model learning rate.")
    parser.add_argument("--output-dir", default=None, help="Artifacts output directory.")
    parser.add_argument("--no-persist-model", action="store_true", help="Do not save trained model artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides: Dict = {
        "data_csv_path": args.csv_path,
        "initial_capital": _optional_float(args.initial_capital),
        "position_fraction": _optional_float(args.position_fraction),
        "brokerage_fee_bps": _optional_float(args.brokerage_fee_bps),
        "slippage_bps": _optional_float(args.slippage_bps),
        "buy_threshold": _optional_float(args.buy_threshold),
        "sell_threshold": _optional_float(args.sell_threshold),
        "step_months": _optional_int(args.step_months),
        "epochs": _optional_int(args.epochs),
        "batch_size": _optional_int(args.batch_size),
        "learning_rate": _optional_float(args.learning_rate),
        "output_dir": args.output_dir,
    }
    if args.no_persist_model:
        overrides["persist_model"] = False

    config = load_pipeline_config(config_path=args.config, overrides=overrides)
    report = run_research_pipeline(config)

    summary = {
        "classification_metrics": report.get("classification_metrics", {}),
        "performance_metrics": report.get("performance_metrics", {}),
        "benchmark_comparison": report.get("benchmark_comparison", {}),
        "report_path": report.get("artifacts", {}).get("report", {}).get("latest_path"),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
