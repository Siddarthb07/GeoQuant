from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.database import list_orders
from app.core.schemas import (
    Candidate,
    CandidateSplitResponse,
    ModelMetrics,
    NewsImpactItem,
    NewsItem,
    PortfolioResponse,
    SelfLearningStatus,
    TradeOrderRequest,
    TradeOrderResponse,
    TrainRequest,
    TrainStatus,
)
from app.data.news_data import fetch_global_news
from app.services.broker_service import place_trade
from app.services.portfolio_service import get_portfolio_performance
from app.services.self_learning import get_self_learning_status
from app.services.signal_service import (
    get_chart_payload,
    get_model_metrics,
    get_news_impact,
    get_ranked_candidates,
    get_ranked_candidates_split,
    refresh_bundles,
)
from app.services.train_service import get_status, launch_training

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/health")
def health() -> dict:
    return {"ok": True}


@router.get("/news", response_model=list[NewsItem])
def news(limit: int = Query(default=40, ge=1, le=200)) -> list[NewsItem]:
    try:
        return [NewsItem(**item) for item in fetch_global_news(limit=limit)]
    except Exception:  # noqa: BLE001
        return []


@router.get("/candidates", response_model=list[Candidate])
def candidates(top_n: int = Query(default=15, ge=1, le=50)) -> list[Candidate]:
    try:
        return [Candidate(**item) for item in get_ranked_candidates(top_n=top_n)]
    except Exception:  # noqa: BLE001
        return []


@router.get("/candidates/split", response_model=CandidateSplitResponse)
def candidates_split(per_side: int = Query(default=15, ge=1, le=30)) -> CandidateSplitResponse:
    try:
        payload = get_ranked_candidates_split(per_side=per_side)
        return CandidateSplitResponse(
            longs=[Candidate(**item) for item in payload.get("longs", [])],
            shorts=[Candidate(**item) for item in payload.get("shorts", [])],
        )
    except Exception:  # noqa: BLE001
        return CandidateSplitResponse(longs=[], shorts=[])


@router.get("/chart/{symbol}")
def chart(
    symbol: str,
    bars: int = Query(default=250, ge=60, le=1000),
    timeframe: str = Query(default="5m"),
) -> dict:
    return get_chart_payload(symbol, bars=bars, timeframe=timeframe)


@router.post("/train")
def train(request: TrainRequest) -> dict:
    started = launch_training(force=request.force)
    if started:
        return {"started": True, "message": "Training started in background."}
    return {"started": False, "message": "Training already running or model already present."}


@router.get("/train/status", response_model=TrainStatus)
def train_status() -> TrainStatus:
    return TrainStatus(**get_status())


@router.post("/order", response_model=TradeOrderResponse)
def order(request: TradeOrderRequest) -> TradeOrderResponse:
    payload = place_trade(
        symbol=request.symbol,
        side=request.side,
        qty=request.qty,
        market=request.market,
        live=request.live,
    )
    return TradeOrderResponse(**payload)


@router.get("/orders")
def orders(limit: int = Query(default=100, ge=1, le=500)) -> dict:
    return {"items": list_orders(limit=limit)}


@router.get("/portfolio", response_model=PortfolioResponse)
def portfolio(limit: int = Query(default=2000, ge=50, le=10000)) -> PortfolioResponse:
    return PortfolioResponse(**get_portfolio_performance(limit=limit))


@router.get("/model-metrics", response_model=ModelMetrics)
def model_metrics() -> ModelMetrics:
    return ModelMetrics(**get_model_metrics())


@router.get("/news-impact", response_model=list[NewsImpactItem])
def news_impact(limit: int = Query(default=15, ge=1, le=40)) -> list[NewsImpactItem]:
    return [NewsImpactItem(**item) for item in get_news_impact(limit=limit)]


@router.get("/self-learning/status", response_model=SelfLearningStatus)
def self_learning_status() -> SelfLearningStatus:
    return SelfLearningStatus(**get_self_learning_status())


@router.post("/reload-models")
def reload_models() -> dict:
    refresh_bundles()
    return {"ok": True, "message": "Model bundles refreshed from disk."}
