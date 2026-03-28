from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.graph_objects as go


def _save_figure(fig: go.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return str(path)


def build_equity_curve_plot(
    equity_curve: pd.DataFrame,
    benchmark_curve: pd.DataFrame,
    output_path: Path,
) -> str:
    fig = go.Figure()
    if not equity_curve.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(equity_curve["timestamp"], utc=True),
                y=equity_curve["equity"],
                mode="lines",
                name="Strategy",
                line={"width": 2},
            )
        )
    if not benchmark_curve.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(benchmark_curve["timestamp"], utc=True),
                y=benchmark_curve["benchmark_equity"],
                mode="lines",
                name="Buy & Hold",
                line={"width": 2, "dash": "dash"},
            )
        )
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Time",
        yaxis_title="Equity",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.1},
    )
    return _save_figure(fig, output_path)


def build_drawdown_plot(equity_curve: pd.DataFrame, output_path: Path) -> str:
    frame = equity_curve.copy()
    if not frame.empty and "drawdown_pct" not in frame.columns:
        roll_max = frame["equity"].cummax().replace(0, pd.NA)
        frame["drawdown_pct"] = (frame["equity"] / roll_max - 1.0) * 100.0

    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(frame["timestamp"], utc=True),
                y=frame["drawdown_pct"],
                mode="lines",
                name="Drawdown %",
                line={"color": "#d62728", "width": 2},
                fill="tozeroy",
            )
        )
    fig.update_layout(
        title="Drawdown Curve",
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
    )
    return _save_figure(fig, output_path)


def build_trade_marker_plot(
    price_frame: pd.DataFrame,
    trade_log: pd.DataFrame,
    output_path: Path,
) -> str:
    if price_frame.empty:
        fig = go.Figure()
        fig.update_layout(title="No price data available for marker chart", template="plotly_white")
        return _save_figure(fig, output_path)

    plot_symbol = str(price_frame["symbol"].iloc[0])
    if not trade_log.empty and "symbol" in trade_log.columns:
        counts = trade_log["symbol"].value_counts()
        if not counts.empty:
            plot_symbol = str(counts.index[0])

    px = price_frame[price_frame["symbol"] == plot_symbol].copy()
    px["timestamp"] = pd.to_datetime(px["timestamp"], utc=True, errors="coerce")
    px = px.dropna(subset=["timestamp"]).sort_values("timestamp")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=px["timestamp"],
            y=px["close"],
            mode="lines",
            name=f"{plot_symbol} Close",
            line={"width": 1.8},
        )
    )

    if not trade_log.empty:
        logs = trade_log[trade_log["symbol"] == plot_symbol].copy()
        if not logs.empty:
            logs["entry_time"] = pd.to_datetime(logs["entry_time"], utc=True, errors="coerce")
            logs["exit_time"] = pd.to_datetime(logs["exit_time"], utc=True, errors="coerce")
            long_entries = logs[logs["side"] == "LONG"]
            short_entries = logs[logs["side"] == "SHORT"]

            if not long_entries.empty:
                fig.add_trace(
                    go.Scatter(
                        x=long_entries["entry_time"],
                        y=long_entries["entry_price"],
                        mode="markers",
                        name="Buy Entry",
                        marker={"symbol": "triangle-up", "size": 9, "color": "#2ca02c"},
                    )
                )
            if not short_entries.empty:
                fig.add_trace(
                    go.Scatter(
                        x=short_entries["entry_time"],
                        y=short_entries["entry_price"],
                        mode="markers",
                        name="Short Entry",
                        marker={"symbol": "triangle-down", "size": 9, "color": "#d62728"},
                    )
                )
            exits = logs.dropna(subset=["exit_time", "exit_price"])
            if not exits.empty:
                fig.add_trace(
                    go.Scatter(
                        x=exits["exit_time"],
                        y=exits["exit_price"],
                        mode="markers",
                        name="Exit",
                        marker={"symbol": "x", "size": 8, "color": "#1f77b4"},
                    )
                )

    fig.update_layout(
        title=f"Trade Markers on Price ({plot_symbol})",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.1},
    )
    return _save_figure(fig, output_path)


def create_visualizations(
    equity_curve: pd.DataFrame,
    benchmark_curve: pd.DataFrame,
    prediction_prices: pd.DataFrame,
    trade_log: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    equity_path = build_equity_curve_plot(equity_curve, benchmark_curve, output_dir / "equity_curve.html")
    drawdown_path = build_drawdown_plot(equity_curve, output_dir / "drawdown_curve.html")
    marker_path = build_trade_marker_plot(prediction_prices, trade_log, output_dir / "trade_markers.html")
    return {
        "equity_curve_plot": equity_path,
        "drawdown_plot": drawdown_path,
        "trade_marker_plot": marker_path,
    }
