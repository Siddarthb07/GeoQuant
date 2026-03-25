from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    force: bool = False


class TrainStatus(BaseModel):
    state: Literal["idle", "running", "failed", "completed"]
    message: str
    updated_at: datetime
    metrics: dict = Field(default_factory=dict)


class NewsItem(BaseModel):
    title: str
    summary: str
    link: str
    source: str
    published_at: datetime
    sentiment: float
    relevance: float
    tags: List[str] = Field(default_factory=list)


class Candidate(BaseModel):
    symbol: str
    market: Literal["US", "INDIA"]
    direction: Literal["LONG", "SHORT"]
    next_day_up_prob: float
    intraday_up_prob: float
    combined_confidence: float
    expected_profit_pct: float
    risk_score: float
    entry_price: float
    stop_loss: float
    target_price: float
    risk_reward: float
    max_loss_pct: float
    take_profit_pct: float
    expected_accuracy: float
    accuracy_reasoning: str
    rationale: str


class ChartPoint(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    sma20: Optional[float] = None
    sma50: Optional[float] = None
    vwap: Optional[float] = None


class TradeOrderRequest(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    qty: float = Field(gt=0)
    market: Literal["US", "INDIA"]
    live: bool = False


class TradeOrderResponse(BaseModel):
    id: int
    symbol: str
    status: str
    mode: Literal["paper", "live"]
    fill_price: Optional[float] = None
    note: str


class ModelMetrics(BaseModel):
    daily_accuracy: float
    daily_auc: float
    intraday_accuracy: float
    intraday_auc: float
    expected_accuracy_band: str
    reasoning: str


class CandidateSplitResponse(BaseModel):
    longs: List[Candidate] = Field(default_factory=list)
    shorts: List[Candidate] = Field(default_factory=list)


class NewsImpactItem(BaseModel):
    symbol: str
    market: Literal["US", "INDIA"]
    impact_score: float
    expected_move: Literal["UP", "DOWN"]
    confidence: float
    themes: List[str] = Field(default_factory=list)
    drivers: List[str] = Field(default_factory=list)


class SelfLearningStatus(BaseModel):
    enabled: bool
    cycle_minutes: int
    retrain_hours: int
    last_cycle_at: Optional[datetime] = None
    last_retrain_at: Optional[datetime] = None
    resolved_signals: int
    rolling_hit_rate: float
    sample_size: int
    state: str
    message: str


class PortfolioItem(BaseModel):
    symbol: str
    market: Literal["US", "INDIA"]
    executed_orders: int
    total_bought_qty: float
    total_sold_qty: float
    net_qty: float
    open_side: Literal["LONG", "SHORT", "FLAT"]
    avg_entry_price: Optional[float] = None
    current_price: Optional[float] = None
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    realized_return_pct: float
    total_return_pct: float


class PortfolioSummary(BaseModel):
    symbols_traded: int
    open_positions: int
    total_realized_pnl: float
    total_unrealized_pnl: float
    total_pnl: float


class PortfolioResponse(BaseModel):
    summary: PortfolioSummary
    items: List[PortfolioItem] = Field(default_factory=list)
