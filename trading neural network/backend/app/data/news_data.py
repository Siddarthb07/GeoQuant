from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import feedparser
import pandas as pd
import requests
from dateutil import parser as dt_parser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


RSS_FEEDS = [
    ("Reuters World", "https://feeds.reuters.com/Reuters/worldNews"),
    ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
]

RELEVANCE_KEYWORDS = {
    "war",
    "conflict",
    "missile",
    "sanction",
    "ceasefire",
    "geopolitical",
    "oil",
    "rates",
    "inflation",
    "federal reserve",
    "rbi",
    "china",
    "russia",
    "ukraine",
    "middle east",
    "taiwan",
    "tariff",
    "defense",
    "attack",
}


analyzer = SentimentIntensityAnalyzer()
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; GeoQuantTrader/1.0; +https://localhost)",
    "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
}

FALLBACK_NEWS_HEADLINES = [
    {
        "title": "Oil routes remain volatile as conflict risk keeps energy traders alert",
        "summary": "War risk around key shipping lanes pushed crude and defense headlines back into focus.",
        "source": "Fallback Market Wire",
    },
    {
        "title": "Federal Reserve commentary keeps rates-sensitive banking stocks active",
        "summary": "Liquidity, inflation, and bond yield updates continue to drive banking-sector positioning.",
        "source": "Fallback Market Wire",
    },
    {
        "title": "AI chip demand outlook supports semiconductor and cloud spending themes",
        "summary": "Data-center and GPU supply commentary lifted focus on technology and software names.",
        "source": "Fallback Market Wire",
    },
    {
        "title": "RBI policy signals and inflation prints shape Indian financial sector sentiment",
        "summary": "Rate guidance continues to affect major Indian banks and broader risk appetite.",
        "source": "Fallback Market Wire",
    },
    {
        "title": "Ceasefire uncertainty and defense contracts influence global industrial flows",
        "summary": "Defense procurement updates and geopolitical risks are impacting multi-sector positioning.",
        "source": "Fallback Market Wire",
    },
    {
        "title": "Consumer demand outlook mixed as retail and media ad trends diverge",
        "summary": "Advertising and discretionary spending updates create stock-specific opportunities.",
        "source": "Fallback Market Wire",
    },
]


def _to_datetime(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = dt_parser.parse(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def _score_relevance(text: str) -> tuple[float, list[str]]:
    lowered = text.lower()
    hits = [k for k in RELEVANCE_KEYWORDS if k in lowered]
    score = min(1.0, len(hits) / 4.0)
    return score, hits[:5]


def _score_sentiment(text: str) -> float:
    return float(analyzer.polarity_scores(text)["compound"])


def _download_feed(url: str, timeout_sec: int = 4) -> feedparser.FeedParserDict:
    try:
        resp = requests.get(url, timeout=timeout_sec, headers=REQUEST_HEADERS)
        resp.raise_for_status()
        return feedparser.parse(resp.content)
    except Exception:  # noqa: BLE001
        return feedparser.FeedParserDict(entries=[])


def _fallback_news(limit: int) -> List[Dict]:
    now = datetime.now(timezone.utc)
    items: list[dict] = []
    for idx, seed in enumerate(FALLBACK_NEWS_HEADLINES):
        title = seed["title"]
        summary = seed["summary"]
        body = f"{title}. {summary}"
        relevance, tags = _score_relevance(body)
        sentiment = _score_sentiment(body)
        published = now - timedelta(minutes=idx * 20)
        items.append(
            {
                "title": title,
                "summary": summary,
                "link": "#",
                "source": seed["source"],
                "published_at": published,
                "sentiment": sentiment,
                "relevance": relevance,
                "tags": tags,
            }
        )
        if len(items) >= limit:
            break
    return items


def fetch_global_news(limit: int = 80) -> List[Dict]:
    items: list[dict] = []
    for source, url in RSS_FEEDS:
        parsed = _download_feed(url)
        for entry in parsed.entries[: max(5, limit // len(RSS_FEEDS) + 3)]:
            title = entry.get("title", "")
            summary = entry.get("summary", "") or entry.get("description", "")
            body = f"{title}. {summary}"
            relevance, tags = _score_relevance(body)
            sentiment = _score_sentiment(body)
            published = _to_datetime(entry.get("published") or entry.get("updated"))
            items.append(
                {
                    "title": title.strip(),
                    "summary": summary.strip(),
                    "link": entry.get("link", ""),
                    "source": source,
                    "published_at": published,
                    "sentiment": sentiment,
                    "relevance": relevance,
                    "tags": tags,
                }
            )

    deduped: dict[str, dict] = {}
    for item in sorted(items, key=lambda x: x["published_at"], reverse=True):
        key = item["title"].strip().lower()
        if key and key not in deduped:
            deduped[key] = item
    merged = list(deduped.values())
    merged.sort(key=lambda x: (x["relevance"], x["published_at"]), reverse=True)
    if not merged:
        return _fallback_news(limit)
    return merged[:limit]


def daily_news_sentiment_series(news_items: List[Dict]) -> pd.Series:
    if not news_items:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(news_items)
    frame["date"] = pd.to_datetime(frame["published_at"]).dt.date
    # Relevance-weighted sentiment.
    frame["weighted"] = frame["sentiment"] * (0.6 + frame["relevance"])
    aggregated = frame.groupby("date")["weighted"].mean()
    return aggregated
