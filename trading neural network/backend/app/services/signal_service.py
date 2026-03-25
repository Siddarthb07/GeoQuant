from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.database import prediction_accuracy_stats, save_prediction_rows
from app.data.market_data import download_daily_history, download_intraday_history, market_from_symbol
from app.data.news_data import daily_news_sentiment_series, fetch_global_news
from app.services.feature_engineering import (
    FEATURE_COLUMNS,
    INTRADAY_FEATURE_COLUMNS,
    add_technical_features,
)
from app.services.neural_model import predict_proba
from app.services.train_service import model_paths


@dataclass
class Bundle:
    model: object
    scaler: object
    feature_columns: list[str]
    symbol_to_code: dict[str, int]
    metrics: Dict


def _load_bundle(prefix: str) -> Optional[Bundle]:
    paths = model_paths(prefix)
    if not all(path.exists() for path in paths.values()):
        return None
    try:
        with paths["meta"].open("r", encoding="utf-8") as f:
            meta = json.load(f)
        model = joblib.load(paths["model"])
        scaler = joblib.load(paths["scaler"])
        return Bundle(
            model=model,
            scaler=scaler,
            feature_columns=meta["feature_columns"],
            symbol_to_code=meta["symbol_to_code"],
            metrics=meta.get("metrics", {}),
        )
    except Exception:  # noqa: BLE001
        return None


DAILY_BUNDLE: Optional[Bundle] = None
INTRADAY_BUNDLE: Optional[Bundle] = None
CALIBRATION_CACHE: Dict = {"updated_at": None, "hit_rate": 0.5, "sample_size": 0}
CANDIDATE_CACHE: Dict = {"updated_at": None, "items": []}

NEWS_THEME_RULES = [
    {
        "theme": "Energy & Oil",
        "keywords": ["oil", "crude", "opec", "brent", "refinery", "middle east", "shipping lane"],
        "symbols": ["XOM", "RELIANCE.NS", "TATAMOTORS.NS"],
        "beta": 1.35,
    },
    {
        "theme": "Rates & Banking",
        "keywords": ["federal reserve", "rates", "bond yields", "rbi", "liquidity", "banking"],
        "symbols": ["JPM", "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"],
        "beta": 1.25,
    },
    {
        "theme": "Technology & AI",
        "keywords": ["ai", "chip", "semiconductor", "cloud", "data center", "gpu", "software"],
        "symbols": ["NVDA", "AMD", "INTC", "MSFT", "GOOGL", "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS"],
        "beta": 1.2,
    },
    {
        "theme": "Defense & Conflict",
        "keywords": ["war", "missile", "conflict", "ceasefire", "attack", "defense", "drone"],
        "symbols": ["BA", "XOM", "RELIANCE.NS", "LT.NS"],
        "beta": 1.3,
    },
    {
        "theme": "Consumer & Media",
        "keywords": ["consumer demand", "retail", "streaming", "media", "advertising", "entertainment"],
        "symbols": ["AMZN", "DIS", "NFLX", "NKE", "ITC.NS", "ASIANPAINT.NS", "TITAN.NS"],
        "beta": 1.05,
    },
]


def _ensure_bundles() -> Tuple[Optional[Bundle], Optional[Bundle]]:
    global DAILY_BUNDLE, INTRADAY_BUNDLE
    if DAILY_BUNDLE is None:
        DAILY_BUNDLE = _load_bundle("daily")
    if INTRADAY_BUNDLE is None:
        INTRADAY_BUNDLE = _load_bundle("intraday")
    return DAILY_BUNDLE, INTRADAY_BUNDLE


def refresh_bundles() -> None:
    global DAILY_BUNDLE, INTRADAY_BUNDLE
    DAILY_BUNDLE = None
    INTRADAY_BUNDLE = None
    _ensure_bundles()


def _stable_unit_interval(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16)
    return value / float(0xFFFFFFFFFFFF)


def _synthetic_price(symbol: str) -> float:
    return round(45.0 + _stable_unit_interval(f"{symbol}:price") * 460.0, 2)


def _set_candidate_cache(items: List[Dict]) -> None:
    CANDIDATE_CACHE["updated_at"] = datetime.now(timezone.utc)
    CANDIDATE_CACHE["items"] = [dict(item) for item in items]


def _get_candidate_cache(top_n: int) -> List[Dict]:
    updated = CANDIDATE_CACHE.get("updated_at")
    if updated is None:
        return []
    if (datetime.now(timezone.utc) - updated).total_seconds() > 120:
        return []
    cached = CANDIDATE_CACHE.get("items") or []
    sliced = [dict(item) for item in cached[:top_n]]
    if len(sliced) < min(top_n, 15):
        return []
    return sliced


def _latest_feature_row_daily(symbol: str, symbol_code: int, news_daily: pd.Series) -> Optional[pd.DataFrame]:
    df = download_daily_history(symbol, start="2021-01-01", end="2030-01-01")
    if len(df) < 80:
        return None
    feats = add_technical_features(df)
    feats["date"] = feats.index.date
    feats["news_sentiment"] = feats["date"].map(news_daily).ffill().fillna(0.0)
    feats["symbol_code"] = symbol_code
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)
    if feats.empty:
        return None
    return feats.iloc[[-1]]


def _latest_feature_row_intraday(symbol: str, symbol_code: int) -> Optional[pd.DataFrame]:
    df = download_intraday_history(symbol, period=settings.intraday_period, interval=settings.intraday_interval)
    if len(df) < 80:
        return None
    feats = add_technical_features(df)
    feats["symbol_code"] = symbol_code
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna(subset=INTRADAY_FEATURE_COLUMNS)
    if feats.empty:
        return None
    return feats.iloc[[-1]]


def _trade_levels(entry: float, atr_pct: float, direction: str) -> tuple[float, float]:
    move = entry * max(0.4, atr_pct) / 100
    risk = 1.2 * move
    reward = 2.2 * move
    if direction == "LONG":
        stop = entry - risk
        target = entry + reward
    else:
        stop = entry + risk
        target = entry - reward
    return round(stop, 2), round(target, 2)


def _risk_profile(entry: float, stop: float, target: float) -> tuple[float, float, float]:
    max_loss_pct = abs(entry - stop) / entry * 100
    take_profit_pct = abs(target - entry) / entry * 100
    risk_reward = take_profit_pct / max(0.001, max_loss_pct)
    return round(risk_reward, 2), round(max_loss_pct, 2), round(take_profit_pct, 2)


def _metric_or_default(metrics: Dict, key: str, default: float) -> float:
    try:
        value = float(metrics.get(key, default))
        if np.isnan(value):
            return default
        return value
    except Exception:  # noqa: BLE001
        return default


def _get_live_hit_rate() -> tuple[float, int]:
    now = datetime.now(timezone.utc)
    updated = CALIBRATION_CACHE.get("updated_at")
    if updated is None or (now - updated).total_seconds() > 300:
        try:
            stats = prediction_accuracy_stats(window=350)
            CALIBRATION_CACHE["hit_rate"] = float(stats["hit_rate"])
            CALIBRATION_CACHE["sample_size"] = int(stats["sample_size"])
            CALIBRATION_CACHE["updated_at"] = now
        except Exception:  # noqa: BLE001
            pass
    return float(CALIBRATION_CACHE.get("hit_rate", 0.5)), int(CALIBRATION_CACHE.get("sample_size", 0))


def _persist_prediction_snapshot(candidates: List[Dict]) -> None:
    if not candidates:
        return
    now = datetime.now(timezone.utc)
    rows: list[dict] = []
    for item in candidates:
        try:
            rows.append(
                {
                    "symbol": item["symbol"],
                    "market": item["market"],
                    "direction": item["direction"],
                    "prediction_date": now.date().isoformat(),
                    "predicted_at": now.isoformat(),
                    "reference_price": float(item["entry_price"]),
                    "predicted_up_prob": float(item.get("combined_up_prob", 0.5)),
                    "confidence": float(item.get("combined_confidence", 0.0)),
                    "expected_accuracy": float(item.get("expected_accuracy", 0.5)),
                    "horizon": "1d",
                }
            )
        except Exception:  # noqa: BLE001
            continue
    save_prediction_rows(rows)


def _expected_accuracy_value(confidence: float, daily_acc: float, intraday_acc: float) -> float:
    # Confidence lifts baseline holdout accuracy but stays bounded.
    live_hit, sample_size = _get_live_hit_rate()
    if sample_size >= 20:
        baseline = (daily_acc + intraday_acc + live_hit) / 3.0
    else:
        baseline = (daily_acc + intraday_acc) / 2.0
    adjusted = baseline + (confidence * 0.12) - 0.03
    return float(np.clip(adjusted, 0.40, 0.88))


def _expected_accuracy_reasoning(
    expected_acc: float,
    daily_acc: float,
    intraday_acc: float,
    confidence: float,
    source: str,
) -> str:
    return (
        f"Expected accuracy {expected_acc*100:.1f}% from holdout baselines "
        f"(daily {daily_acc*100:.1f}%, intraday {intraday_acc*100:.1f}%) "
        f"adjusted by live confidence ({confidence*100:.1f}%). "
        f"Intraday signal source: {source}."
    )


def _synthetic_candidates(top_n: int = 15, note: str = "") -> List[Dict]:
    symbols = settings.default_us_universe + settings.default_india_universe
    out: list[dict] = []
    for symbol in symbols:
        up_prob = float(np.clip(0.5 + (_stable_unit_interval(f"{symbol}:bias") - 0.5) * 0.46, 0.1, 0.9))
        confidence = float(np.clip(abs(up_prob - 0.5) * 2.0, 0.08, 0.92))
        atr_pct = 0.8 + _stable_unit_interval(f"{symbol}:atr") * 3.0
        entry = _synthetic_price(symbol)
        direction = "LONG" if up_prob >= 0.5 else "SHORT"
        stop, target = _trade_levels(entry, atr_pct, direction)
        rr, loss_pct, tp_pct = _risk_profile(entry, stop, target)
        expected_acc = float(np.clip(0.49 + confidence * 0.22, 0.45, 0.75))
        out.append(
            {
                "symbol": symbol,
                "market": market_from_symbol(symbol),
                "direction": direction,
                "next_day_up_prob": round(up_prob, 4),
                "intraday_up_prob": round(up_prob, 4),
                "combined_up_prob": round(up_prob, 4),
                "combined_confidence": round(confidence, 4),
                "expected_profit_pct": round(max(0.2, tp_pct * max(0.35, up_prob)), 2),
                "risk_score": round(35.0 + _stable_unit_interval(f"{symbol}:risk") * 40.0, 2),
                "entry_price": round(entry, 2),
                "stop_loss": stop,
                "target_price": target,
                "risk_reward": rr,
                "max_loss_pct": loss_pct,
                "take_profit_pct": tp_pct,
                "expected_accuracy": round(expected_acc, 4),
                "accuracy_reasoning": (
                    "Live market feed unavailable, so this signal uses deterministic synthetic priors "
                    "until fresh prices are received."
                ),
                "rationale": note
                or "Synthetic fallback signal used while waiting for live feed refresh.",
            }
        )
    out.sort(key=lambda x: (x["expected_profit_pct"], x["combined_confidence"]), reverse=True)
    return out[:top_n]


def _heuristic_candidates(top_n: int = 15) -> List[Dict]:
    symbols = settings.default_us_universe + settings.default_india_universe
    out: list[dict] = []
    for symbol in symbols:
        try:
            df = download_daily_history(symbol, start="2023-01-01", end="2030-01-01")
        except Exception:  # noqa: BLE001
            continue
        if len(df) < 60:
            continue
        feats = add_technical_features(df).dropna()
        if feats.empty:
            continue
        latest = feats.iloc[-1]
        momentum = float(np.tanh(latest["ret_5"] * 35 + latest["macd_hist"] * 3))
        prob_up = float(np.clip(0.5 + momentum * 0.2, 0.05, 0.95))
        direction = "LONG" if prob_up >= 0.5 else "SHORT"
        confidence = abs(prob_up - 0.5) * 2
        atr_pct = float(latest["atr_pct"])
        entry = float(latest["close"])
        stop, target = _trade_levels(entry, atr_pct, direction)
        rr, loss_pct, tp_pct = _risk_profile(entry, stop, target)
        expected_acc = float(np.clip(0.5 + (confidence * 0.08) - 0.02, 0.42, 0.66))
        out.append(
            {
                "symbol": symbol,
                "market": market_from_symbol(symbol),
                "direction": direction,
                "next_day_up_prob": round(prob_up, 4),
                "intraday_up_prob": round(prob_up, 4),
                "combined_up_prob": round(prob_up, 4),
                "combined_confidence": round(confidence, 4),
                "expected_profit_pct": round(confidence * atr_pct * 1.5, 2),
                "risk_score": round(min(100.0, latest["vol_20"] * 10000), 2),
                "entry_price": round(entry, 2),
                "stop_loss": stop,
                "target_price": target,
                "risk_reward": rr,
                "max_loss_pct": loss_pct,
                "take_profit_pct": tp_pct,
                "expected_accuracy": round(expected_acc, 4),
                "accuracy_reasoning": (
                    "Heuristic fallback estimate using momentum/volatility confidence; "
                    "model holdout metrics unavailable for this symbol/state."
                ),
                "rationale": "Heuristic mode: momentum + volatility proxy (train model to replace this).",
            }
        )
    if not out:
        return _synthetic_candidates(
            top_n=top_n,
            note="Synthetic fallback mode enabled because price feed calls returned no usable bars.",
        )
    if len(out) < top_n:
        existing = {item["symbol"] for item in out}
        for item in _synthetic_candidates(top_n=top_n * 2):
            if item["symbol"] in existing:
                continue
            out.append(item)
            existing.add(item["symbol"])
            if len(out) >= top_n:
                break
    out.sort(key=lambda x: (x["expected_profit_pct"], x["combined_confidence"]), reverse=True)
    return out[:top_n]


def get_ranked_candidates(top_n: int = 15) -> List[Dict]:
    cached = _get_candidate_cache(top_n=top_n)
    if cached:
        return cached

    daily_bundle, intraday_bundle = _ensure_bundles()
    if daily_bundle is None:
        fallback = _heuristic_candidates(top_n=top_n)
        _persist_prediction_snapshot(fallback)
        _set_candidate_cache(fallback)
        return fallback

    news_items = fetch_global_news(limit=200)
    news_daily = daily_news_sentiment_series(news_items)
    daily_acc = _metric_or_default(daily_bundle.metrics, "accuracy", 0.5)
    intraday_acc = _metric_or_default(intraday_bundle.metrics, "accuracy", daily_acc) if intraday_bundle else daily_acc

    symbols = list(daily_bundle.symbol_to_code.keys())
    live_symbol_cap = min(len(symbols), max(12, min(20, top_n)))
    intraday_budget = 8
    candidates = []

    for idx, symbol in enumerate(symbols[:live_symbol_cap]):
        try:
            code = daily_bundle.symbol_to_code[symbol]
            daily_row = _latest_feature_row_daily(symbol, code, news_daily)
            if daily_row is None:
                continue

            x_day = daily_bundle.scaler.transform(
                daily_row[daily_bundle.feature_columns].to_numpy(dtype=float)
            )
            day_prob = float(predict_proba(daily_bundle.model, x_day)[0])

            intraday_source = "daily-only-model"
            if intraday_bundle is None:
                intra_prob = day_prob
            elif idx >= intraday_budget:
                intra_prob = day_prob
                intraday_source = "daily-budget-proxy"
            else:
                intra_row = _latest_feature_row_intraday(symbol, code)
                intraday_source = "intraday"
                if intra_row is None:
                    # If intraday bars are unavailable from free feed, use latest daily feature row
                    # for a best-effort intraday estimate instead of dropping the symbol.
                    intra_row = daily_row.copy()
                    intraday_source = "daily-fallback"

                try:
                    x_intra = intraday_bundle.scaler.transform(
                        intra_row[intraday_bundle.feature_columns].to_numpy(dtype=float)
                    )
                    intra_prob = float(predict_proba(intraday_bundle.model, x_intra)[0])
                except Exception:  # noqa: BLE001
                    intra_prob = day_prob
                    intraday_source = "daily-proxy"

            combined_up = 0.55 * day_prob + 0.45 * intra_prob
            direction = "LONG" if combined_up >= 0.5 else "SHORT"
            confidence = float(abs(combined_up - 0.5) * 2.0)

            last = daily_row.iloc[-1]
            atr_pct = float(last["atr_pct"])
            entry = float(last["close"])
            stop, target = _trade_levels(entry, atr_pct, direction)
            rr, loss_pct, tp_pct = _risk_profile(entry, stop, target)
            expected = confidence * atr_pct * 1.9
            risk = min(100.0, float(last["vol_20"]) * 10000 + (1 - confidence) * 40)
            news_signal = float(last["news_sentiment"])

            if direction == "LONG":
                rationale = (
                    f"Bullish model blend; news={news_signal:+.2f}. "
                    f"Intraday signal source: {intraday_source}."
                )
            else:
                rationale = (
                    f"Bearish model blend; news={news_signal:+.2f}. "
                    f"Intraday signal source: {intraday_source}."
                )
            expected_acc = _expected_accuracy_value(confidence, daily_acc, intraday_acc)
            accuracy_reasoning = _expected_accuracy_reasoning(
                expected_acc,
                daily_acc,
                intraday_acc,
                confidence,
                intraday_source,
            )

            candidates.append(
                {
                    "symbol": symbol,
                    "market": market_from_symbol(symbol),
                    "direction": direction,
                    "next_day_up_prob": round(day_prob, 4),
                    "intraday_up_prob": round(intra_prob, 4),
                    "combined_up_prob": round(combined_up, 4),
                    "combined_confidence": round(confidence, 4),
                    "expected_profit_pct": round(expected, 2),
                    "risk_score": round(risk, 2),
                    "entry_price": round(entry, 2),
                    "stop_loss": stop,
                    "target_price": target,
                    "risk_reward": rr,
                    "max_loss_pct": loss_pct,
                    "take_profit_pct": tp_pct,
                    "expected_accuracy": round(expected_acc, 4),
                    "accuracy_reasoning": accuracy_reasoning,
                    "rationale": rationale,
                }
            )
        except Exception:  # noqa: BLE001
            continue

    candidates.sort(
        key=lambda x: (x["expected_profit_pct"] * x["combined_confidence"]) / max(1.0, x["risk_score"]),
        reverse=True,
    )
    candidates = candidates[:top_n]

    if len(candidates) < top_n:
        existing = {item["symbol"] for item in candidates}
        fallback = _synthetic_candidates(
            top_n=max(top_n * 2, 50),
            note="Synthetic supplement added while awaiting additional live-feed coverage.",
        )
        for item in fallback:
            if item["symbol"] in existing:
                continue
            item["rationale"] = f"{item['rationale']} Intraday feed fallback engaged."
            candidates.append(item)
            existing.add(item["symbol"])
            if len(candidates) >= top_n:
                break

    final_candidates = candidates[:top_n]
    if not final_candidates:
        final_candidates = _get_candidate_cache(top_n=top_n)
    if not final_candidates:
        final_candidates = _synthetic_candidates(
            top_n=top_n,
            note="Synthetic fallback mode enabled because model inference produced no live symbols.",
        )
    _persist_prediction_snapshot(final_candidates)
    _set_candidate_cache(final_candidates)
    return final_candidates


def _candidate_variant(candidate: Dict, direction: str) -> tuple[Dict, float]:
    item = dict(candidate)
    entry = float(item["entry_price"])
    loss_pct = float(item.get("max_loss_pct", 1.0))
    tp_pct = float(item.get("take_profit_pct", 1.8))
    up_prob = float(item.get("combined_up_prob", 0.5))
    side_prob = up_prob if direction == "LONG" else (1.0 - up_prob)
    side_conf = float(np.clip(abs(side_prob - 0.5) * 2.0, 0.0, 1.0))

    if direction == "LONG":
        stop = entry * (1 - loss_pct / 100.0)
        target = entry * (1 + tp_pct / 100.0)
    else:
        stop = entry * (1 + loss_pct / 100.0)
        target = entry * (1 - tp_pct / 100.0)

    item["direction"] = direction
    item["stop_loss"] = round(stop, 2)
    item["target_price"] = round(target, 2)
    item["combined_confidence"] = round(side_conf, 4)
    item["expected_profit_pct"] = round(max(0.05, side_prob * tp_pct), 2)
    item["expected_accuracy"] = round(float(np.clip(0.42 + side_prob * 0.5, 0.4, 0.9)), 4)
    item["accuracy_reasoning"] = (
        f"{direction} setup probability from combined model is {side_prob*100:.1f}%. "
        "Higher side probability improves expected directional hit-rate."
    )
    item["rationale"] = f"{direction} side ranking generated from current model probabilities."
    score = (side_prob * max(0.3, tp_pct) * (0.6 + side_conf)) / max(1.0, float(item.get("risk_score", 1.0)))
    return item, float(score)


def get_ranked_candidates_split(per_side: int = 15) -> Dict:
    base = get_ranked_candidates(top_n=max(60, per_side * 4))
    if not base:
        return {"longs": [], "shorts": []}

    long_pool: list[tuple[float, Dict]] = []
    short_pool: list[tuple[float, Dict]] = []
    for candidate in base:
        long_item, long_score = _candidate_variant(candidate, "LONG")
        short_item, short_score = _candidate_variant(candidate, "SHORT")
        long_pool.append((long_score, long_item))
        short_pool.append((short_score, short_item))

    long_pool.sort(key=lambda x: x[0], reverse=True)
    short_pool.sort(key=lambda x: x[0], reverse=True)

    longs = [item for _, item in long_pool[:per_side]]
    shorts = [item for _, item in short_pool[:per_side]]
    return {"longs": longs, "shorts": shorts}


def get_news_impact(limit: int = 20) -> List[Dict]:
    try:
        news_items = fetch_global_news(limit=140)
    except Exception:  # noqa: BLE001
        news_items = []
    valid_symbols = set(settings.default_us_universe + settings.default_india_universe)
    cutoff = datetime.now(timezone.utc) - timedelta(days=2)

    impact_map: dict[str, dict] = {}
    for item in news_items:
        published = item.get("published_at")
        if published is not None and published < cutoff:
            continue
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        text = f"{title}. {summary}".lower()
        if not text:
            continue
        sentiment = float(item.get("sentiment", 0.0))
        relevance = float(item.get("relevance", 0.0))
        sign = 1.0 if sentiment >= 0 else -1.0
        magnitude_base = (0.45 + relevance) * (0.35 + abs(sentiment))

        for rule in NEWS_THEME_RULES:
            if not any(keyword in text for keyword in rule["keywords"]):
                continue
            delta = sign * magnitude_base * float(rule["beta"])
            for symbol in rule["symbols"]:
                if symbol not in valid_symbols:
                    continue
                rec = impact_map.setdefault(
                    symbol,
                    {"impact_score": 0.0, "themes": set(), "drivers": []},
                )
                rec["impact_score"] += delta
                rec["themes"].add(rule["theme"])
                if title and len(rec["drivers"]) < 5 and title not in rec["drivers"]:
                    rec["drivers"].append(title)

    impacted: list[dict] = []
    for symbol, rec in impact_map.items():
        score = float(rec["impact_score"])
        impacted.append(
            {
                "symbol": symbol,
                "market": market_from_symbol(symbol),
                "impact_score": round(score, 4),
                "expected_move": "UP" if score >= 0 else "DOWN",
                "confidence": round(float(np.clip(abs(score) / 2.5, 0.05, 1.0)), 4),
                "themes": sorted(list(rec["themes"]))[:4],
                "drivers": rec["drivers"][:4],
            }
        )
    impacted.sort(key=lambda x: (abs(x["impact_score"]), x["confidence"]), reverse=True)
    if not impacted:
        fallback = _get_candidate_cache(top_n=min(max(limit, 8), 25))
        if not fallback:
            fallback = get_ranked_candidates(top_n=min(max(limit, 8), 25))
        for item in fallback:
            direction = item.get("direction", "LONG")
            score = float(item.get("combined_confidence", 0.25)) * (1.0 if direction == "LONG" else -1.0)
            impacted.append(
                {
                    "symbol": item["symbol"],
                    "market": item["market"],
                    "impact_score": round(score, 4),
                    "expected_move": "UP" if score >= 0 else "DOWN",
                    "confidence": round(float(np.clip(abs(score), 0.05, 1.0)), 4),
                    "themes": ["Fallback Signal"],
                    "drivers": ["Live headline impact unavailable; using model directional pressure."],
                }
            )
        impacted.sort(key=lambda x: (abs(x["impact_score"]), x["confidence"]), reverse=True)
    return impacted[:limit]


def _synthetic_chart(symbol: str, bars: int, timeframe: str) -> dict:
    tf = (timeframe or "5m").lower()
    freq_map = {
        "1m": "1min",
        "2m": "2min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "1h": "60min",
        "1d": "1D",
    }
    freq = freq_map.get(tf, "5min")
    point_count = int(np.clip(bars, 60, 1000))
    end_ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    idx = pd.date_range(end=end_ts, periods=point_count, freq=freq)
    seed = int(_stable_unit_interval(f"{symbol}:{tf}:chart") * 1_000_000)
    rng = np.random.default_rng(seed)
    base = _synthetic_price(symbol)
    volatility = 0.002 + _stable_unit_interval(f"{symbol}:{tf}:vol") * 0.012
    drift = (_stable_unit_interval(f"{symbol}:{tf}:drift") - 0.5) * 0.0015
    returns = rng.normal(loc=drift, scale=volatility, size=point_count)
    close = base * np.exp(np.cumsum(returns))
    open_px = np.roll(close, 1)
    open_px[0] = close[0]
    spread = np.maximum(0.002 * close, np.abs(rng.normal(0.0, 0.004 * close, size=point_count)))
    high = np.maximum(open_px, close) + spread
    low = np.minimum(open_px, close) - spread
    volume = np.maximum(10.0, rng.normal(loc=900000.0, scale=250000.0, size=point_count))
    frame = pd.DataFrame(
        {
            "open": open_px,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    frame["sma20"] = frame["close"].rolling(20).mean()
    frame["sma50"] = frame["close"].rolling(50).mean()
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    frame["vwap"] = (typical * frame["volume"]).cumsum() / frame["volume"].replace(0, np.nan).cumsum()

    points = []
    for ts, row in frame.iterrows():
        points.append(
            {
                "timestamp": ts.isoformat(),
                "open": round(float(row["open"]), 4),
                "high": round(float(row["high"]), 4),
                "low": round(float(row["low"]), 4),
                "close": round(float(row["close"]), 4),
                "volume": round(float(row["volume"]), 2),
                "sma20": None if pd.isna(row["sma20"]) else round(float(row["sma20"]), 4),
                "sma50": None if pd.isna(row["sma50"]) else round(float(row["sma50"]), 4),
                "vwap": None if pd.isna(row["vwap"]) else round(float(row["vwap"]), 4),
            }
        )
    return {
        "symbol": symbol,
        "points": points,
        "timeframe": tf,
        "source": "synthetic-fallback",
        "delayed": True,
    }


def get_chart_payload(symbol: str, bars: int = 250, timeframe: str = "5m") -> dict:
    tf = (timeframe or "5m").lower()
    intraday_map = {
        "1m": ("7d", "1m"),
        "2m": ("30d", "2m"),
        "5m": ("60d", "5m"),
        "15m": ("60d", "15m"),
        "30m": ("60d", "30m"),
        "60m": ("730d", "60m"),
        "1h": ("730d", "60m"),
    }

    source = "daily"
    delayed = False
    used_timeframe = tf

    if tf in intraday_map:
        period, interval = intraday_map[tf]
        df = download_intraday_history(symbol, period=period, interval=interval)
        source = "intraday"
        delayed = True
    else:
        df = download_daily_history(symbol, start="2022-01-01", end="2030-01-01")

    if df.empty and source == "intraday":
        df = download_daily_history(symbol, start="2022-01-01", end="2030-01-01")
        source = "daily-fallback"
        used_timeframe = "1d"

    if df.empty:
        return _synthetic_chart(symbol=symbol, bars=bars, timeframe=used_timeframe)

    frame = df.copy()
    frame["sma20"] = frame["close"].rolling(20).mean()
    frame["sma50"] = frame["close"].rolling(50).mean()
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    cum_vol = frame["volume"].replace(0, np.nan).cumsum()
    frame["vwap"] = (typical * frame["volume"]).cumsum() / cum_vol
    frame = frame.tail(bars)
    payload = []
    for idx, row in frame.iterrows():
        payload.append(
            {
                "timestamp": idx.isoformat(),
                "open": round(float(row["open"]), 4),
                "high": round(float(row["high"]), 4),
                "low": round(float(row["low"]), 4),
                "close": round(float(row["close"]), 4),
                "volume": float(row["volume"]),
                "sma20": None if pd.isna(row["sma20"]) else round(float(row["sma20"]), 4),
                "sma50": None if pd.isna(row["sma50"]) else round(float(row["sma50"]), 4),
                "vwap": None if pd.isna(row["vwap"]) else round(float(row["vwap"]), 4),
            }
        )
    return {
        "symbol": symbol,
        "points": payload,
        "timeframe": used_timeframe,
        "source": source,
        "delayed": delayed,
    }


def get_model_metrics() -> Dict:
    daily_bundle, intraday_bundle = _ensure_bundles()
    daily_metrics = daily_bundle.metrics if daily_bundle else {}
    intraday_metrics = intraday_bundle.metrics if intraday_bundle else {}

    daily_acc = _metric_or_default(daily_metrics, "accuracy", 0.5)
    daily_auc = _metric_or_default(daily_metrics, "auc", 0.5)
    intraday_acc = _metric_or_default(intraday_metrics, "accuracy", daily_acc)
    intraday_auc = _metric_or_default(intraday_metrics, "auc", daily_auc)
    live_hit, live_samples = _get_live_hit_rate()

    base = (daily_acc + intraday_acc + live_hit) / 3.0 if live_samples >= 20 else (daily_acc + intraday_acc) / 2.0
    low = int(max(40, (base - 0.03) * 100))
    high = int(min(88, (base + 0.07) * 100))
    band = f"{low}% - {high}%"

    reasoning = (
        "Band blends out-of-sample daily/intraday holdout quality with rolling live hit-rate from resolved signals. "
        "Higher confidence setups tend to sit near the upper band."
    )
    return {
        "daily_accuracy": round(daily_acc, 4),
        "daily_auc": round(daily_auc, 4),
        "intraday_accuracy": round(intraday_acc, 4),
        "intraday_auc": round(intraday_auc, 4),
        "expected_accuracy_band": band,
        "reasoning": f"{reasoning} Live hit-rate sample: {live_samples}.",
    }
