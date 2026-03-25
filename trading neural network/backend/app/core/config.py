from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "GeoQuant Neural Trader"
    app_host: str = "0.0.0.0"
    app_port: int = 8020

    data_start_date: str = "2000-01-01"
    data_end_date: str = "2025-12-31"
    intraday_period: str = "60d"
    intraday_interval: str = "30m"
    enable_self_learning: bool = True
    self_learning_refresh_minutes: int = 20
    self_learning_retrain_hours: int = 24

    model_dir: Path = Path(__file__).resolve().parents[1] / "models_store"
    sqlite_path: Path = Path(__file__).resolve().parents[1] / "trading.db"

    default_us_universe: List[str] = Field(
        default_factory=lambda: [
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOGL",
            "META",
            "TSLA",
            "JPM",
            "XOM",
            "UNH",
            "V",
            "MA",
            "AVGO",
            "AMD",
            "BA",
            "DIS",
            "NFLX",
            "NKE",
            "INTC",
            "PLTR",
        ]
    )
    default_india_universe: List[str] = Field(
        default_factory=lambda: [
            "RELIANCE.NS",
            "TCS.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS",
            "INFY.NS",
            "SBIN.NS",
            "ITC.NS",
            "LT.NS",
            "BHARTIARTL.NS",
            "KOTAKBANK.NS",
            "ASIANPAINT.NS",
            "BAJFINANCE.NS",
            "HCLTECH.NS",
            "AXISBANK.NS",
            "SUNPHARMA.NS",
            "MARUTI.NS",
            "TITAN.NS",
            "ULTRACEMCO.NS",
            "WIPRO.NS",
            "TATAMOTORS.NS",
        ]
    )

    alpaca_key_id: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"


settings = Settings()
settings.model_dir.mkdir(parents=True, exist_ok=True)
