"""
External signals collector.

Collects signals from external sources that may indicate
upcoming traffic changes:
- Social media mentions and trending
- News coverage
- Competitor activity
- Search trends
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from src.utils.logging import get_logger

from .base import BaseCollector

logger = get_logger(__name__)


class ExternalSignalsCollector(BaseCollector):
    """
    Collector for external signals that may impact traffic.

    Monitors:
    - Social media mentions (Twitter/X, Reddit, etc.)
    - News coverage
    - Search trends
    - Competitor activity

    Each signal has a confidence score based on source reliability.
    """

    def __init__(
        self,
        service_name: str = "default",
        brand_keywords: list[str] | None = None,
        collection_interval: float = 600.0,  # 10 minutes
        twitter_bearer_token: str | None = None,
        newsapi_key: str | None = None,
    ) -> None:
        """
        Initialize external signals collector.

        Args:
            service_name: Name for metrics labeling
            brand_keywords: Keywords to monitor (brand names, product names)
            collection_interval: Seconds between collections
            twitter_bearer_token: Twitter API bearer token
            newsapi_key: NewsAPI.org API key
        """
        super().__init__(
            name=f"external-{service_name}",
            collection_interval=collection_interval,
        )

        self.service_name = service_name
        self.brand_keywords = brand_keywords or []

        # API credentials
        self.twitter_bearer_token = twitter_bearer_token
        self.newsapi_key = newsapi_key

        # HTTP client
        self._client: httpx.AsyncClient | None = None

        # Signal weights for confidence scoring
        self._source_confidence: dict[str, float] = {
            "twitter": 0.7,        # Social media can be noisy
            "reddit": 0.6,         # Even noisier
            "news_major": 0.9,     # Major news outlets are reliable
            "news_minor": 0.7,     # Smaller outlets less so
            "search_trends": 0.8,  # Search trends are predictive
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def collect(self) -> list[dict[str, Any]]:
        """
        Collect external signals from all configured sources.

        Returns:
            List of metric dictionaries representing signals
        """
        metrics = []
        timestamp = datetime.now(timezone.utc)

        # Collect from Twitter
        if self.twitter_bearer_token and self.brand_keywords:
            twitter_signals = await self._collect_twitter_signals()
            metrics.extend(self._signals_to_metrics(twitter_signals, timestamp))

        # Collect from News API
        if self.newsapi_key and self.brand_keywords:
            news_signals = await self._collect_news_signals()
            metrics.extend(self._signals_to_metrics(news_signals, timestamp))

        # Add aggregate signal strength
        total_signal_strength = sum(
            m["value"] * m["labels"].get("confidence", 1.0)
            for m in metrics
            if m["metric_name"] == "external_signal"
        )

        metrics.append({
            "timestamp": timestamp,
            "service_name": self.service_name,
            "metric_name": "external_signals_aggregate",
            "value": total_signal_strength,
            "labels": {"sources_count": len(metrics)},
        })

        logger.info(
            "External signals collection complete",
            signal_count=len(metrics),
            aggregate_strength=total_signal_strength,
        )

        return metrics

    async def _collect_twitter_signals(self) -> list[dict[str, Any]]:
        """
        Collect signals from Twitter/X.

        Monitors:
        - Recent tweet volume for brand keywords
        - Engagement rates
        - Sentiment (simplified)
        """
        signals = []

        if not self.twitter_bearer_token:
            return signals

        client = await self._get_client()

        for keyword in self.brand_keywords:
            try:
                # Twitter API v2 recent search
                response = await client.get(
                    "https://api.twitter.com/2/tweets/search/recent",
                    params={
                        "query": f"{keyword} -is:retweet lang:en",
                        "max_results": 100,
                        "tweet.fields": "public_metrics,created_at",
                    },
                    headers={
                        "Authorization": f"Bearer {self.twitter_bearer_token}",
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    tweets = data.get("data", [])
                    meta = data.get("meta", {})

                    # Calculate signal strength based on volume and engagement
                    tweet_count = meta.get("result_count", 0)
                    total_engagement = 0

                    for tweet in tweets:
                        metrics = tweet.get("public_metrics", {})
                        total_engagement += (
                            metrics.get("like_count", 0) +
                            metrics.get("retweet_count", 0) * 2 +
                            metrics.get("reply_count", 0) * 3
                        )

                    # Normalize signal (0-1 scale)
                    volume_signal = min(tweet_count / 100, 1.0)
                    engagement_signal = min(total_engagement / 1000, 1.0)
                    combined_signal = (volume_signal + engagement_signal) / 2

                    signals.append({
                        "source": "twitter",
                        "keyword": keyword,
                        "signal_strength": combined_signal,
                        "confidence": self._source_confidence["twitter"],
                        "metadata": {
                            "tweet_count": tweet_count,
                            "total_engagement": total_engagement,
                        },
                    })

                elif response.status_code == 429:
                    logger.warning("Twitter API rate limited")
                    break
                else:
                    logger.warning(
                        "Twitter API error",
                        status_code=response.status_code,
                        keyword=keyword,
                    )

            except Exception as e:
                logger.error("Error collecting Twitter signals", error=str(e))

        return signals

    async def _collect_news_signals(self) -> list[dict[str, Any]]:
        """
        Collect signals from news articles.

        Uses NewsAPI to find recent articles mentioning brand keywords.
        """
        signals = []

        if not self.newsapi_key:
            return signals

        client = await self._get_client()

        for keyword in self.brand_keywords:
            try:
                # Search for recent news
                from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

                response = await client.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": keyword,
                        "from": from_date,
                        "sortBy": "relevancy",
                        "language": "en",
                        "pageSize": 50,
                        "apiKey": self.newsapi_key,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    articles = data.get("articles", [])
                    total_results = data.get("totalResults", 0)

                    # Categorize sources
                    major_sources = [
                        "bbc", "cnn", "reuters", "bloomberg", "techcrunch",
                        "wsj", "nytimes", "washingtonpost", "forbes"
                    ]

                    major_count = 0
                    minor_count = 0

                    for article in articles:
                        source_name = article.get("source", {}).get("name", "").lower()
                        if any(major in source_name for major in major_sources):
                            major_count += 1
                        else:
                            minor_count += 1

                    # Calculate signal strength
                    volume_signal = min(total_results / 50, 1.0)
                    major_signal = min(major_count / 10, 1.0)

                    # Higher weight for major news coverage
                    combined_signal = (volume_signal * 0.4) + (major_signal * 0.6)

                    confidence = (
                        self._source_confidence["news_major"]
                        if major_count > 0
                        else self._source_confidence["news_minor"]
                    )

                    signals.append({
                        "source": "news",
                        "keyword": keyword,
                        "signal_strength": combined_signal,
                        "confidence": confidence,
                        "metadata": {
                            "total_articles": total_results,
                            "major_source_articles": major_count,
                            "minor_source_articles": minor_count,
                        },
                    })

                else:
                    logger.warning(
                        "NewsAPI error",
                        status_code=response.status_code,
                        keyword=keyword,
                    )

            except Exception as e:
                logger.error("Error collecting news signals", error=str(e))

        return signals

    def _signals_to_metrics(
        self,
        signals: list[dict[str, Any]],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        """Convert signals to metric dictionaries."""
        metrics = []

        for signal in signals:
            metrics.append({
                "timestamp": timestamp,
                "service_name": self.service_name,
                "metric_name": "external_signal",
                "value": signal["signal_strength"],
                "labels": {
                    "source": signal["source"],
                    "keyword": signal["keyword"],
                    "confidence": signal["confidence"],
                },
            })

        return metrics

    def add_keyword(self, keyword: str) -> None:
        """Add a keyword to monitor."""
        if keyword not in self.brand_keywords:
            self.brand_keywords.append(keyword)
            logger.info("Added monitoring keyword", keyword=keyword)

    def remove_keyword(self, keyword: str) -> None:
        """Remove a keyword from monitoring."""
        if keyword in self.brand_keywords:
            self.brand_keywords.remove(keyword)
            logger.info("Removed monitoring keyword", keyword=keyword)

    async def stop(self) -> None:
        """Stop collector and close HTTP client."""
        await super().stop()
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
